"""Pluggable scheduler/executor boundary for the continuous Controller.

The production ``ktp`` and ``ktp_lab`` paths remain implemented by the
Controller for backwards compatibility.  New schedulers use the versioned
JSON bridge defined here, so replacing the resource manager does not grant the
adapter authority over candidate selection, measurement assessment, or Session
state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


EXECUTOR_ADAPTER_API_VERSION = "vllmtkb-executor-adapter/v1"
REQUIRED_ACTIONS = frozenset(
    {
        "prepare",
        "check_ready",
        "submit",
        "snapshot",
        "stop",
        "wait_for_release",
    }
)
OPTIONAL_ACTIONS = frozenset({"start_benchmark", "stop_partial"})


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _reject_inline_secrets(value: Any, path: str = "executor_adapter.config") -> None:
    """Keep archived adapter configuration free of credentials."""

    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.lower()
            secret_like = any(
                marker in normalized
                for marker in ("password", "secret", "api_key", "token", "credential")
            )
            env_reference = normalized.endswith("_env") or normalized.endswith(
                "_env_var"
            )
            if secret_like and not env_reference and child not in (None, ""):
                raise ValueError(
                    f"{path}.{key} looks like an inline secret; store only an "
                    "environment-variable name in a *_env field"
                )
            _reject_inline_secrets(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_inline_secrets(child, f"{path}[{index}]")


def _validated_bridge_path(
    setting: dict[str, Any], project_root: Path
) -> tuple[Path, list[str]]:
    bridge_value = str(setting.get("bridge_path", "")).strip()
    bridge = Path(bridge_value)
    if not bridge_value or bridge.is_absolute() or ".." in bridge.parts:
        raise ValueError(
            "executor_adapter.bridge_path must be a project-relative path"
        )
    if bridge.suffix.lower() != ".py":
        raise ValueError("executor_adapter.bridge_path must identify a .py file")

    roots = [str(item).strip() for item in setting.get("allowlisted_roots", [])]
    if not roots:
        raise ValueError("executor_adapter.allowlisted_roots must not be empty")
    root_paths: list[Path] = []
    for value in roots:
        root = Path(value)
        if not value or root.is_absolute() or ".." in root.parts:
            raise ValueError(
                "executor_adapter.allowlisted_roots entries must be project-relative"
            )
        root_paths.append((project_root / root).resolve())

    resolved = (project_root / bridge).resolve()
    if not resolved.is_file() or not resolved.is_relative_to(project_root.resolve()):
        raise ValueError(
            "executor_adapter.bridge_path does not exist inside the project"
        )
    if not any(resolved == root or resolved.is_relative_to(root) for root in root_paths):
        raise ValueError(
            "executor_adapter.bridge_path is outside executor_adapter.allowlisted_roots"
        )
    return resolved, roots


class CommandExecutorAdapter:
    """Call an operator-owned scheduler bridge through a strict JSON protocol."""

    def __init__(
        self,
        *,
        bridge_path: Path,
        setting: dict[str, Any],
        identity: dict[str, Any],
    ) -> None:
        self.bridge_path = bridge_path
        self.setting = copy.deepcopy(setting)
        self.identity = copy.deepcopy(identity)
        self.timeout_seconds = int(setting.get("timeout_seconds", 300))
        if not 1 <= self.timeout_seconds <= 86400:
            raise ValueError("executor_adapter.timeout_seconds must be 1..86400")
        self.python_command = str(setting.get("python_command") or sys.executable)
        if not self.python_command.strip():
            raise ValueError("executor_adapter.python_command must not be empty")
        capabilities = setting.get("capabilities", {})
        if capabilities and not isinstance(capabilities, dict):
            raise ValueError("executor_adapter.capabilities must be a mapping")
        self.capabilities = {
            action: bool((capabilities or {}).get(action, action in REQUIRED_ACTIONS))
            for action in REQUIRED_ACTIONS | OPTIONAL_ACTIONS
        }
        missing = sorted(
            action for action in REQUIRED_ACTIONS if not self.capabilities[action]
        )
        if missing:
            raise ValueError(
                "executor_adapter disables required capabilities: " + ", ".join(missing)
            )

    def supports(self, action: str) -> bool:
        return bool(self.capabilities.get(action, False))

    def invoke(
        self,
        action: str,
        *,
        context: dict[str, Any],
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if action not in REQUIRED_ACTIONS | OPTIONAL_ACTIONS:
            raise ValueError(f"Unknown executor adapter action: {action}")
        if not self.supports(action):
            raise NotImplementedError(
                f"Executor adapter {self.identity['name']!r} does not support {action}"
            )
        request = {
            "api_version": EXECUTOR_ADAPTER_API_VERSION,
            "action": action,
            "context": copy.deepcopy(context),
            "payload": copy.deepcopy(payload or {}),
            # Adapter configuration is frozen in session_config.yaml.  Secrets
            # must be referenced by environment-variable name, never embedded.
            "adapter_config": copy.deepcopy(self.setting.get("config", {})),
        }
        environment = os.environ.copy()
        environment["VLLMTKB_EXECUTOR_ACTION"] = action
        completed = subprocess.run(
            [self.python_command, str(self.bridge_path)],
            input=json.dumps(request, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=self.timeout_seconds,
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            raise RuntimeError(
                f"Executor adapter action {action!r} failed with exit code "
                f"{completed.returncode}: {(stderr or stdout)[-4000:]}"
            )
        stdout = completed.stdout.strip()
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Executor adapter action {action!r} did not return one JSON object"
            ) from exc
        if not isinstance(result, dict):
            raise RuntimeError(
                f"Executor adapter action {action!r} returned a non-object JSON value"
            )
        if result.get("api_version") != EXECUTOR_ADAPTER_API_VERSION:
            raise RuntimeError(
                f"Executor adapter action {action!r} returned an incompatible api_version"
            )
        if result.get("ok") is not True:
            raise RuntimeError(
                f"Executor adapter action {action!r} rejected the request: "
                f"{str(result.get('error') or result.get('message') or 'unknown error')[:4000]}"
            )
        return result


def resolve_executor_adapter(
    config: dict[str, Any], project_root: Path
) -> tuple[CommandExecutorAdapter | None, dict[str, Any]]:
    """Resolve and fingerprint the selected execution backend.

    Legacy modes intentionally bypass the external bridge so their command,
    state, and timing behavior remains byte-for-byte compatible.
    """

    mode = str(config.get("execution_mode", "ktp"))
    if mode in {"ktp", "ktp_lab"}:
        return None, {
            "api_version": EXECUTOR_ADAPTER_API_VERSION,
            "kind": "legacy_builtin",
            "name": mode,
            "sha256": _canonical_sha256({"kind": "legacy_builtin", "name": mode}),
        }
    if mode != "executor_adapter":
        raise ValueError(f"Unsupported execution_mode={mode!r}")

    setting = config.get("executor_adapter")
    if not isinstance(setting, dict):
        raise ValueError(
            "executor_adapter mapping is required when execution_mode=executor_adapter"
        )
    if setting.get("kind", "command_v1") != "command_v1":
        raise ValueError("Only executor_adapter.kind=command_v1 is supported")
    bridge_path, allowlisted_roots = _validated_bridge_path(setting, project_root)
    _reject_inline_secrets(setting.get("config", {}))
    bridge_bytes = bridge_path.read_bytes()
    frozen_setting = copy.deepcopy(setting)
    frozen_setting["allowlisted_roots"] = allowlisted_roots
    identity_source = {
        "api_version": EXECUTOR_ADAPTER_API_VERSION,
        "kind": "command_v1",
        "name": str(setting.get("name") or bridge_path.stem),
        "bridge_path": _portable_path(bridge_path, project_root),
        "bridge_sha256": hashlib.sha256(bridge_bytes).hexdigest(),
        "python_command": str(setting.get("python_command") or sys.executable),
        "timeout_seconds": int(setting.get("timeout_seconds", 300)),
        "config": copy.deepcopy(setting.get("config", {})),
        "capabilities": copy.deepcopy(setting.get("capabilities", {})),
    }
    identity = copy.deepcopy(identity_source)
    identity["sha256"] = _canonical_sha256(identity_source)
    adapter = CommandExecutorAdapter(
        bridge_path=bridge_path,
        setting=frozen_setting,
        identity=identity,
    )
    return adapter, identity


def validate_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize the platform-independent task state returned by an adapter."""

    snapshot = result.get("snapshot")
    if not isinstance(snapshot, dict):
        raise RuntimeError("Executor snapshot response must contain snapshot object")
    for field in ("terminal", "partial_failure"):
        if not isinstance(snapshot.get(field), bool):
            raise RuntimeError(f"Executor snapshot.{field} must be boolean")
    active = snapshot.get("active_pods")
    if active is not None and (not isinstance(active, int) or active < 0):
        raise RuntimeError("Executor snapshot.active_pods must be null or non-negative int")
    normalized = {
        "status": snapshot.get("status"),
        "active_pods": active,
        "terminal": snapshot["terminal"],
        "partial_failure": snapshot["partial_failure"],
    }
    for key, value in snapshot.items():
        if key not in normalized:
            normalized[key] = value
    return normalized
