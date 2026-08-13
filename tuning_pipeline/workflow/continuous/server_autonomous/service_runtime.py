#!/usr/bin/env python3
"""Service-manager guard and portable config renderer.

Exit code 78 means that automatic recovery is intentionally blocked. Both
generated service definitions treat it as a clean, non-restartable outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

BLOCKED_EXIT = 78
COMPLETED_STATUSES = {"completed_by_agent", "tuning_complete", "dry_run_complete"}
ARCHIVED_RESUME_STATUSES = {
    "stopped_after_current_round",
    "stopped_after_failed_round",
}
RECOVERABLE_CONTROLLER_STATUSES = {"recovering_controller_error"}
LEGACY_RECOVERABLE_AGENT_MARKERS = (
    "Agent analysis failed",
    "Structured output error",
    "schema-valid JSON",
)


def _merge(base: dict[str, object], overlay: dict[str, object]) -> dict[str, object]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_config(path: Path, seen: set[Path] | None = None) -> dict[str, object]:
    resolved = path.resolve()
    visited = set() if seen is None else seen
    if resolved in visited:
        raise ValueError(f"Recursive base_config reference: {resolved}")
    visited.add(resolved)
    value = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {resolved}")
    base_setting = value.pop("base_config", None)
    if not base_setting:
        return value
    base_path = Path(str(base_setting))
    if not base_path.is_absolute():
        base_path = resolved.parent / base_path
    return _merge(_load_config(base_path, visited), value)


def resolve_lease_name(runtime_root: Path, config_path: Path) -> str:
    """Prefer the frozen Session lease, then use the merged configuration."""
    state_path = runtime_root / "state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        lease_name = str(state.get("lease_name", "")).strip()
        if lease_name:
            return lease_name
    config = _load_config(config_path)
    lab = config.get("lab", {})
    if not isinstance(lab, dict) or not str(lab.get("lease_name", "")).strip():
        raise ValueError("No Lease name exists in Session state or merged configuration")
    return str(lab["lease_name"]).strip()


def process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def terminal_artifact_exists(state: dict[str, object]) -> bool:
    session_dir = Path(str(state.get("session_dir", "")))
    if not state.get("session_dir"):
        return False
    try:
        index = int(state.get("round_index", 0))
    except (TypeError, ValueError):
        return False
    label = str(state.get("round_label", ""))
    round_dir = session_dir / f"round_{index:03d}_{label}"
    results = round_dir / "05_results"
    return (results / "metrics.json").is_file() or (results / "failure.yaml").is_file()


def decide(runtime_root: Path, requested_mode: str) -> tuple[str, str]:
    stop_marker = runtime_root / "STOP_REQUESTED"
    if stop_marker.exists():
        return "blocked", f"graceful-stop marker exists: {stop_marker}"

    lock_path = runtime_root / "controller.lock"
    if lock_path.exists():
        try:
            pid = int(lock_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            pid = -1
        if process_is_running(pid):
            return "blocked", f"controller PID {pid} already owns the runtime lock"

    state_path = runtime_root / "state.json"
    if requested_mode == "new":
        return "--start", "explicit new Session requested"
    if requested_mode == "resume" and not state_path.is_file():
        return "blocked", f"resume requested but state is absent: {state_path}"
    if not state_path.is_file():
        return "--start", "no prior state; starting a new Session"

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        backup_path = state_path.with_name(state_path.name + ".previous")
        try:
            state = json.loads(backup_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as backup_exc:
            return "blocked", (
                "state and last-known-good backup cannot be read safely: "
                f"primary={exc}; backup={backup_exc}"
            )
        if not isinstance(state, dict):
            return "blocked", "last-known-good state backup is not a JSON object"

    status = str(state.get("status", ""))
    if status in COMPLETED_STATUSES:
        return "complete", f"Session is already terminal: {status}"
    if status in RECOVERABLE_CONTROLLER_STATUSES and terminal_artifact_exists(state):
        return "--resume", f"retrying bounded recoverable Controller error: {status}"
    if (
        status == "paused_controller_error"
        and terminal_artifact_exists(state)
        and any(
            marker in str(state.get("controller_error", ""))
            for marker in LEGACY_RECOVERABLE_AGENT_MARKERS
        )
    ):
        return "--resume", "migrating legacy Agent protocol pause into recovery"
    if status.startswith("paused_"):
        return "blocked", f"Session requires operator review: {status}"

    has_task = bool(state.get("active_task_id"))
    has_run = bool(state.get("active_run_id"))
    if isinstance(state.get("pending_submission"), dict):
        return "--resume", "recovering an interrupted submission transaction"
    if has_task and has_run:
        return "--resume", f"recovering active task/run from status {status or 'unknown'}"
    if has_task != has_run:
        return "blocked", "state contains only one of active_task_id/active_run_id"
    if status in ARCHIVED_RESUME_STATUSES and terminal_artifact_exists(state):
        return "--resume", f"resuming archived terminal round from status {status}"
    if (
        status == "running"
        and state.get("analysis_status") == "ready"
        and terminal_artifact_exists(state)
    ):
        return "--resume", "recovering validated analysis before candidate submission"
    return "blocked", f"state is not safely auto-resumable: status={status or 'missing'}"


def render(repo_root: Path, env_file: Path, output_root: Path) -> list[Path]:
    autonomous = repo_root / "tuning_pipeline/workflow/continuous/server_autonomous"
    runner = autonomous / "run_foreground.sh"
    output_root.mkdir(parents=True, exist_ok=True)
    socket_tag = hashlib.sha256(str(repo_root).encode("utf-8")).hexdigest()[:12]
    supervisor_socket = repo_root.parent / f".vllmtkb-{socket_tag}.sock"
    if len(os.fsencode(supervisor_socket)) >= 104:
        raise ValueError(
            "Approved workspace parent is too long for a safe Supervisor AF_UNIX socket: "
            f"{supervisor_socket}"
        )
    values = {
        "@REPO_ROOT@": str(repo_root),
        "@RUNNER@": str(runner),
        "@ENV_FILE@": str(env_file),
        "@SERVICE_ROOT@": str(output_root),
        "@SUPERVISOR_SOCKET@": str(supervisor_socket),
    }
    template_root = autonomous / "service_templates"
    outputs = []
    for source_name, target_name in (
        ("vllmtkb-server-autonomous.service.in", "vllmtkb-server-autonomous.service"),
        ("supervisord.conf.in", "supervisord.conf"),
    ):
        content = (template_root / source_name).read_text(encoding="utf-8")
        for token, value in values.items():
            content = content.replace(token, value)
        target = output_root / target_name
        target.write_text(content, encoding="utf-8", newline="\n")
        outputs.append(target)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    guard = sub.add_parser("decide")
    guard.add_argument("--runtime-root", type=Path, required=True)
    guard.add_argument("--mode", choices=("auto", "new", "resume"), default="auto")
    renderer = sub.add_parser("render")
    renderer.add_argument("--repo-root", type=Path, required=True)
    renderer.add_argument("--env-file", type=Path, required=True)
    renderer.add_argument("--output-root", type=Path, required=True)
    lease = sub.add_parser("lease-name")
    lease.add_argument("--runtime-root", type=Path, required=True)
    lease.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "decide":
        action, reason = decide(args.runtime_root, args.mode)
        print(action)
        print(reason, file=os.sys.stderr)
        return BLOCKED_EXIT if action == "blocked" else 0
    if args.command == "lease-name":
        print(resolve_lease_name(args.runtime_root, args.config))
        return 0
    for path in render(
        args.repo_root.resolve(), args.env_file.resolve(), args.output_root.resolve()
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
