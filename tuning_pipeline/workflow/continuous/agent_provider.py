"""Pluggable structured-decision providers for the online tuning Controller.

Secrets are referenced by environment-variable name only and are never stored
in project configuration or Session artifacts.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_CODEX_PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass
class AgentResult:
    provider: str
    returncode: int
    stdout: str
    stderr: str


def resolve_agent_profile(
    agent_section: dict[str, Any] | None,
    *,
    legacy_command: str = "auto",
) -> dict[str, Any]:
    """Resolve one provider's settings without exposing credentials."""
    section = dict(agent_section or {})
    provider = str(section.get("provider", "codex"))
    profiles = dict(section.get("providers", {}))
    if provider in profiles:
        settings = dict(profiles[provider])
    else:
        settings = dict(
            section.get("settings", {"command": legacy_command})
        )
    if provider not in {
        "codex",
        "anthropic",
        "openai_compatible",
        "deepseek",
        "command",
    }:
        raise ValueError(f"Unsupported agent.provider={provider!r}")
    return {"provider": provider, "settings": settings}


def _extract_json(text: str) -> Any:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
                return value
            except json.JSONDecodeError:
                continue
        raise


def _prune_disallowed_metadata(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str = "$",
) -> tuple[Any, list[str]]:
    """Remove only object keys explicitly forbidden by a JSON Schema.

    Agent providers occasionally attach explanatory metadata next to an
    otherwise valid structured decision.  Values, required fields and types
    are never coerced here; ordinary schema validation remains authoritative.
    """
    removed: list[str] = []
    if isinstance(value, dict) and (
        schema.get("type") == "object" or isinstance(schema.get("properties"), dict)
    ):
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if key in properties:
                normalized[key], child_removed = _prune_disallowed_metadata(
                    item, properties[key], path=child_path
                )
                removed.extend(child_removed)
            elif additional is False:
                removed.append(child_path)
            else:
                normalized[key] = item
        return normalized, removed
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        normalized_items: list[Any] = []
        for index, item in enumerate(value):
            normalized, child_removed = _prune_disallowed_metadata(
                item, schema["items"], path=f"{path}[{index}]"
            )
            normalized_items.append(normalized)
            removed.extend(child_removed)
        return normalized_items, removed
    return value, removed


def _fallback_validate_json_schema(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str = "$",
) -> None:
    """Validate the strict schema subset used by Controller decisions.

    Production prefers ``jsonschema``. This fallback keeps the contract strict
    in minimal/offline controller environments instead of silently accepting an
    invalid Agent value when that optional package is absent.
    """

    def json_equal(left: Any, right: Any) -> bool:
        return type(left) is type(right) and left == right

    def matches(candidate: Any, branch: dict[str, Any]) -> bool:
        try:
            _fallback_validate_json_schema(candidate, branch, path=path)
            return True
        except ValueError:
            return False

    if "if" in schema and matches(value, schema["if"]):
        _fallback_validate_json_schema(value, schema.get("then", {}), path=path)
    for branch in schema.get("allOf", []):
        _fallback_validate_json_schema(value, branch, path=path)
    if "anyOf" in schema and not any(
        matches(value, branch) for branch in schema["anyOf"]
    ):
        raise ValueError(f"{path} does not match any allowed schema")
    if "const" in schema and not json_equal(value, schema["const"]):
        raise ValueError(f"{path} must equal {schema['const']!r}")
    if "enum" in schema and not any(json_equal(value, item) for item in schema["enum"]):
        raise ValueError(f"{path}={value!r} is outside the allowed enum")

    declared_types = schema.get("type")
    if declared_types is not None:
        names = [declared_types] if isinstance(declared_types, str) else declared_types
        checks = {
            "object": lambda item: isinstance(item, dict),
            "array": lambda item: isinstance(item, list),
            "string": lambda item: isinstance(item, str),
            "boolean": lambda item: isinstance(item, bool),
            "null": lambda item: item is None,
            "number": lambda item: isinstance(item, (int, float))
            and not isinstance(item, bool),
            "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        }
        if not any(checks.get(name, lambda _item: False)(value) for name in names):
            raise ValueError(f"{path} has invalid type {type(value).__name__}; expected {names}")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{path} is missing required property {key!r}")
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                raise ValueError(f"{path} has forbidden properties {extras}")
        for key, child in value.items():
            if key in properties:
                _fallback_validate_json_schema(
                    child, properties[key], path=f"{path}.{key}"
                )
    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            raise ValueError(f"{path} has fewer than minItems")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise ValueError(f"{path} has more than maxItems")
        if isinstance(schema.get("items"), dict):
            for index, child in enumerate(value):
                _fallback_validate_json_schema(
                    child, schema["items"], path=f"{path}[{index}]"
                )
    if isinstance(value, str) and len(value) < int(schema.get("minLength", 0)):
        raise ValueError(f"{path} is shorter than minLength")
    if isinstance(value, str) and "pattern" in schema:
        if re.search(str(schema["pattern"]), value) is None:
            raise ValueError(f"{path} does not match the required pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} is above maximum")


def _validate_and_write(value: Any, schema_path: Path, output_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    normalized, removed = _prune_disallowed_metadata(value, schema)
    try:
        import jsonschema

        jsonschema.validate(normalized, schema)
    except ImportError:
        _fallback_validate_json_schema(normalized, schema)
    output_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if removed:
        output_path.with_name(output_path.name + ".normalization.json").write_text(
            json.dumps(
                {
                    "policy": "prune_schema_forbidden_metadata_v1",
                    "removed_paths": removed,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _required_secret(settings: dict[str, Any]) -> str:
    name = str(settings.get("api_key_env", "")).strip()
    if not name:
        raise RuntimeError("API provider requires agent.settings.api_key_env")
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"Required Agent credential environment variable is unset: {name}")
    return value


def _codex_profile(settings: dict[str, Any]) -> str:
    """Return one safe, explicit Codex profile name or an empty string.

    An explicit profile is the opt-in boundary for loading Codex user config.
    The ordinary ``codex`` route keeps its historical deterministic behavior
    and continues to pass ``--ignore-user-config``.
    """
    profile = os.environ.get("VLLMTKB_CODEX_PROFILE", "").strip() or str(
        settings.get("profile", "")
    ).strip()
    if profile and not _CODEX_PROFILE_RE.fullmatch(profile):
        raise RuntimeError(
            "Codex profile must contain only letters, digits, '.', '_' or '-'"
        )
    return profile


def _codex_environment(settings: dict[str, Any]) -> dict[str, str]:
    """Build the Codex subprocess environment without copying any secret value."""
    environment = os.environ.copy()
    for setting_name, environment_name in (
        ("codex_home", "CODEX_HOME"),
        ("tmp_dir", "TMPDIR"),
    ):
        configured = str(settings.get(setting_name, "")).strip()
        if not configured:
            continue
        directory = Path(configured).expanduser()
        if not directory.is_absolute() or not directory.is_dir():
            raise RuntimeError(
                f"Codex {setting_name} must be an existing absolute directory"
            )
        environment[environment_name] = str(directory)
    return environment


def _codex_output_wait_seconds(settings: dict[str, Any]) -> int:
    wait_seconds = int(settings.get("output_wait_seconds", 60))
    if not 1 <= wait_seconds <= 300:
        raise RuntimeError("Codex output_wait_seconds must be between 1 and 300")
    return wait_seconds


def _output_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _normalize_codex_output(
    *,
    output_path: Path,
    schema_path: Path,
    previous_signature: tuple[int, int] | None,
    wait_seconds: int,
) -> None:
    """Wait for Codex's last-message file and normalize it to schema-valid JSON.

    Some Codex model providers finish the CLI process before the last-message
    writer has made its final non-empty update.  Other providers wrap an
    otherwise valid JSON value in prose or a Markdown fence.  Do not let either
    transport detail surface as a Controller JSONDecodeError.
    """
    deadline = time.monotonic() + wait_seconds
    last_error: Exception | None = None
    while True:
        signature = _output_signature(output_path)
        if signature is not None and signature != previous_signature:
            raw = output_path.read_text(encoding="utf-8", errors="replace")
            if raw.strip():
                try:
                    value = _extract_json(raw)
                    # Keep the provider's exact response for diagnosis while
                    # making the documented decision path machine-readable.
                    output_path.with_name(output_path.name + ".raw.txt").write_text(
                        raw, encoding="utf-8"
                    )
                    _validate_and_write(value, schema_path, output_path)
                    return
                except Exception as exc:  # Schema/format may still be mid-write.
                    last_error = exc
        if time.monotonic() >= deadline:
            detail = (
                f"{type(last_error).__name__}: {last_error}"
                if last_error is not None
                else "last-message file was missing, unchanged, or empty"
            )
            raise RuntimeError(
                f"Codex did not produce schema-valid JSON within {wait_seconds}s: "
                + detail
            )
        time.sleep(0.25)


def validate_agent_credentials(agent: dict[str, Any]) -> None:
    """Fail at Controller startup instead of after a costly benchmark round."""
    provider = str(agent.get("provider", "codex")).lower()
    settings = dict(agent.get("settings", {}))
    if provider == "codex":
        _codex_profile(settings)
        _codex_environment(settings)
        _codex_output_wait_seconds(settings)
        requested = os.environ.get("VLLMTKB_CODEX_COMMAND", "").strip() or str(
            settings.get("command", "auto")
        )
        executable = (
            shutil.which("codex.cmd") or shutil.which("codex")
            if requested.lower() == "auto"
            else (
                requested
                if Path(requested).expanduser().is_file()
                else shutil.which(requested)
            )
        )
        if not executable:
            raise RuntimeError(
                "Codex CLI was not found. Install/login Codex, or set "
                "VLLMTKB_CODEX_COMMAND."
            )
    elif provider in {"anthropic", "openai_compatible", "deepseek"}:
        _required_secret(settings)
    elif provider == "command" and not settings.get("command"):
        raise RuntimeError("command provider requires agent.settings.command")


def _http_json(url: str, headers: dict[str, str], body: dict[str, Any], timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Agent API returned HTTP {exc.code}: {detail[-2000:]}") from exc


def _run_deepseek(
    settings: dict[str, Any],
    *,
    schema_prompt: str,
    schema_path: Path,
    output_path: Path,
    timeout: int,
) -> AgentResult:
    """Call DeepSeek with bounded transport/format recovery.

    JSON mode guarantees syntactic JSON, not conformance to our decision
    schema.  The frozen local schema remains authoritative and is checked on
    every attempt before a response is accepted.
    """
    base = str(settings.get("base_url", "https://api.deepseek.com")).rstrip("/")
    model = str(settings.get("model", "deepseek-v4-flash")).strip()
    if not model:
        raise RuntimeError("DeepSeek provider requires agent.settings.model")
    attempts = int(settings.get("max_api_retries", 3)) + 1
    if not 1 <= attempts <= 6:
        raise RuntimeError("DeepSeek max_api_retries must be between 0 and 5")
    thinking = str(settings.get("thinking", "enabled")).lower()
    if thinking not in {"enabled", "disabled"}:
        raise RuntimeError("DeepSeek thinking must be enabled or disabled")
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": schema_prompt}],
        "response_format": {"type": "json_object"},
        "thinking": {"type": thinking},
        "max_tokens": int(settings.get("max_tokens", 16384)),
    }
    if thinking == "enabled":
        body["reasoning_effort"] = str(settings.get("reasoning_effort", "high"))
    else:
        body["temperature"] = float(settings.get("temperature", 0))

    failures: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            payload = _http_json(
                base + "/chat/completions",
                {"Authorization": "Bearer " + _required_secret(settings)},
                body,
                timeout,
            )
            text = str(payload["choices"][0]["message"].get("content") or "").strip()
            if not text:
                raise RuntimeError("DeepSeek returned empty content")
            value = _extract_json(text)
            _validate_and_write(value, schema_path, output_path)
            return AgentResult("deepseek", 0, text, "")
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            urllib.error.URLError,
            TimeoutError,
            RuntimeError,
        ) as exc:
            failures.append(f"attempt {attempt}/{attempts}: {type(exc).__name__}: {exc}")
            if attempt == attempts:
                break
            delay = min(60, 5 * (4 ** (attempt - 1)))
            time.sleep(delay)
    return AgentResult("deepseek", 1, "", "\n".join(failures)[-8000:])


def run_structured_agent(
    agent: dict[str, Any],
    *,
    prompt: str,
    schema_path: Path,
    output_path: Path,
    cwd: Path,
    allowed_dir: Path,
    timeout: int = 1800,
) -> AgentResult:
    provider = str(agent.get("provider", "codex")).lower()
    settings = dict(agent.get("settings", {}))
    if provider == "codex":
        profile = _codex_profile(settings)
        codex_environment = _codex_environment(settings)
        output_wait_seconds = _codex_output_wait_seconds(settings)
        use_user_config = bool(settings.get("use_user_config", False))
        requested = os.environ.get("VLLMTKB_CODEX_COMMAND", "").strip() or str(
            settings.get("command", "auto")
        )
        if requested.lower() == "auto":
            executable = shutil.which("codex.cmd") or shutil.which("codex")
        else:
            candidate = Path(requested).expanduser()
            executable = str(candidate) if candidate.is_file() else shutil.which(requested)
        if not executable:
            raise RuntimeError(
                "Codex CLI was not found. Install/login Codex, or set "
                "VLLMTKB_CODEX_COMMAND."
            )
        command = [
            executable,
            "exec",
        ]
        if profile:
            # Profiles live in CODEX_HOME and may select a custom model
            # provider (for example a DeepSeek-compatible gateway). Loading
            # user config is allowed only through this explicit opt-in.
            command.extend(["--profile", profile])
        elif use_user_config:
            # Explicitly load the base config from CODEX_HOME. This supports a
            # server-managed Codex installation whose tested model provider is
            # selected at the top level rather than in a named profile.
            pass
        else:
            command.append("--ignore-user-config")
        if bool(settings.get("ephemeral", False)):
            command.append("--ephemeral")
        command.extend(
            [
                "-C",
                str(cwd),
                "--add-dir",
                str(allowed_dir),
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "--json",
                "-",
            ]
        )
        previous_signature = _output_signature(output_path)
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            input=prompt,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            env=codex_environment,
            timeout=timeout,
            check=False,
        )
        if completed.returncode == 0:
            try:
                _normalize_codex_output(
                    output_path=output_path,
                    schema_path=schema_path,
                    previous_signature=previous_signature,
                    wait_seconds=output_wait_seconds,
                )
            except Exception as exc:
                return AgentResult(
                    provider,
                    1,
                    completed.stdout,
                    (completed.stderr + "\n" + f"Structured output error: {exc}").strip(),
                )
        return AgentResult(provider, completed.returncode, completed.stdout, completed.stderr)

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_prompt = (
        prompt
        + "\n\nReturn one JSON value only. It must validate against this JSON Schema:\n"
        + json.dumps(schema, ensure_ascii=False)
    )
    if provider == "deepseek":
        return _run_deepseek(
            settings,
            schema_prompt=schema_prompt,
            schema_path=schema_path,
            output_path=output_path,
            timeout=timeout,
        )
    if provider == "anthropic":
        payload = _http_json(
            str(settings.get("base_url", "https://api.anthropic.com/v1/messages")),
            {
                "x-api-key": _required_secret(settings),
                "anthropic-version": str(settings.get("api_version", "2023-06-01")),
            },
            {
                "model": str(settings.get("model", "claude-sonnet-4-6")),
                "max_tokens": int(settings.get("max_tokens", 8192)),
                "messages": [{"role": "user", "content": schema_prompt}],
            },
            timeout,
        )
        text = "\n".join(
            str(item.get("text", ""))
            for item in payload.get("content", [])
            if item.get("type") == "text"
        )
    elif provider == "openai_compatible":
        base = str(settings.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        payload = _http_json(
            base + "/chat/completions",
            {"Authorization": "Bearer " + _required_secret(settings)},
            {
                "model": str(settings["model"]),
                "messages": [{"role": "user", "content": schema_prompt}],
                "temperature": 0,
            },
            timeout,
        )
        text = str(payload["choices"][0]["message"]["content"])
    elif provider == "command":
        command = shlex.split(str(settings.get("command", "")), posix=os.name != "nt")
        if not command:
            raise RuntimeError("command provider requires agent.settings.command")
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            input=schema_prompt,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            return AgentResult(provider, completed.returncode, completed.stdout, completed.stderr)
        text = completed.stdout
    else:
        raise RuntimeError(
            f"Unsupported Agent provider {provider!r}; choose codex, anthropic, "
            "openai_compatible, deepseek, or command"
        )

    value = _extract_json(text)
    _validate_and_write(value, schema_path, output_path)
    return AgentResult(provider, 0, text, "")
