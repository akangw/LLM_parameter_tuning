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
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    if provider not in {"codex", "anthropic", "openai_compatible", "command"}:
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


def _validate_and_write(value: Any, schema_path: Path, output_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        import jsonschema

        jsonschema.validate(value, schema)
    except ImportError:
        pass
    output_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _required_secret(settings: dict[str, Any]) -> str:
    name = str(settings.get("api_key_env", "")).strip()
    if not name:
        raise RuntimeError("API provider requires agent.settings.api_key_env")
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"Required Agent credential environment variable is unset: {name}")
    return value


def validate_agent_credentials(agent: dict[str, Any]) -> None:
    """Fail at Controller startup instead of after a costly benchmark round."""
    provider = str(agent.get("provider", "codex")).lower()
    settings = dict(agent.get("settings", {}))
    if provider == "codex":
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
    elif provider in {"anthropic", "openai_compatible"}:
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
            "--ignore-user-config",
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
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            input=prompt,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return AgentResult(provider, completed.returncode, completed.stdout, completed.stderr)

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_prompt = (
        prompt
        + "\n\nReturn one JSON value only. It must validate against this JSON Schema:\n"
        + json.dumps(schema, ensure_ascii=False)
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
            "openai_compatible, or command"
        )

    value = _extract_json(text)
    _validate_and_write(value, schema_path, output_path)
    return AgentResult(provider, 0, text, "")
