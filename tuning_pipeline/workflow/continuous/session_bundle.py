#!/usr/bin/env python3
"""Portable, integrity-checked export/import for immutable Session evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "vllmtkb-session-bundle/v1"
TERMINAL_STATUSES = {
    "stopped_after_current_round",
    "completed_by_agent",
    "stopped_after_failed_round",
    "paused_for_human",
    "tuning_complete",
    "paused_after_repeated_infrastructure_failure",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def inspect_bundle(bundle: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    with zipfile.ZipFile(bundle, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or not all(_safe_member(name) for name in names):
            raise ValueError("Bundle contains duplicate or unsafe paths")
        if "manifest.json" not in names or "state.json" not in names:
            raise ValueError("Bundle is missing manifest.json or state.json")
        files = {name: archive.read(name) for name in names if not name.endswith("/")}
    manifest = json.loads(files["manifest.json"])
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported Session bundle schema: {manifest.get('schema')!r}")
    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise ValueError("Bundle manifest files must be a mapping")
    actual_names = set(files) - {"manifest.json"}
    if actual_names != set(expected):
        raise ValueError("Bundle members do not match its manifest")
    for name, identity in expected.items():
        data = files[name]
        if identity != {"sha256": _sha256(data), "size": len(data)}:
            raise ValueError(f"Bundle integrity check failed: {name}")
    return manifest, files


def export_session(
    runtime_root: Path,
    output: Path,
    *,
    session_dir: Path | None = None,
    allow_active_snapshot: bool = False,
) -> dict[str, Any]:
    state_path = runtime_root / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Controller state does not exist: {state_path}")
    state = _json(state_path)
    status = str(state.get("status", "unknown"))
    if status not in TERMINAL_STATUSES and not allow_active_snapshot:
        raise RuntimeError(
            f"Session status {status!r} is active or unknown; pass "
            "--allow-active-snapshot for an explicitly non-final snapshot"
        )
    source = (session_dir or Path(str(state.get("session_dir", "")))).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Session directory does not exist: {source}")
    session_id = str(state.get("session_id") or source.name)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", session_id):
        raise ValueError(f"Unsafe Session id: {session_id!r}")

    payloads: dict[str, bytes] = {
        "state.json": state_path.read_bytes(),
    }
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Session bundle refuses symlinks: {path}")
        if path.is_file():
            relative = path.relative_to(source).as_posix()
            payloads[f"session/{relative}"] = path.read_bytes()
    manifest = {
        "schema": SCHEMA,
        "session_id": session_id,
        "source_status": status,
        "active_snapshot": status not in TERMINAL_STATUSES,
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": {
            name: {"sha256": _sha256(data), "size": len(data)}
            for name, data in sorted(payloads.items())
        },
    }
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
            for name, data in sorted(payloads.items()):
                archive.writestr(name, data)
        os.replace(temporary, output)
    except Exception:
        # Preserve the failed temporary artifact for diagnosis; this tool never
        # deletes operator data or partially written evidence.
        raise
    return manifest


def import_session(bundle: Path, runtime_root: Path, *, activate: bool = False) -> Path:
    manifest, files = inspect_bundle(bundle)
    session_id = str(manifest.get("session_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", session_id):
        raise ValueError(f"Unsafe Session id: {session_id!r}")
    destination = (runtime_root / "experiments" / session_id).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing Session: {destination}")
    if activate and (runtime_root / "state.json").exists():
        raise FileExistsError("Refusing to overwrite existing controller state")
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in sorted(files.items()):
        if not name.startswith("session/"):
            continue
        relative = PurePosixPath(name).relative_to("session")
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    if activate:
        state = json.loads(files["state.json"])
        state["session_dir"] = str(destination)
        runtime_root.mkdir(parents=True, exist_ok=True)
        (runtime_root / "state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--runtime-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--session-dir", type=Path)
    export.add_argument("--allow-active-snapshot", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("bundle", type=Path)
    restore = commands.add_parser("import")
    restore.add_argument("bundle", type=Path)
    restore.add_argument("--runtime-root", type=Path, required=True)
    restore.add_argument("--activate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "export":
        result = export_session(
            args.runtime_root,
            args.output,
            session_dir=args.session_dir,
            allow_active_snapshot=args.allow_active_snapshot,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "verify":
        result, _ = inspect_bundle(args.bundle)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        destination = import_session(args.bundle, args.runtime_root, activate=args.activate)
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
