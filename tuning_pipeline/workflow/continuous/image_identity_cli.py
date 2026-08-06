#!/usr/bin/env python3
"""Validate or atomically approve a probed runtime-image identity."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "remote" / "image_version_manifest.yaml"
DEFAULT_ACTIVATION = HERE / "activation.approved.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def validate_pair(manifest_path: Path, activation_path: Path) -> None:
    try:
        from .continuous_tuning import validate_activation_approval
    except ImportError:
        from continuous_tuning import validate_activation_approval
    validate_activation_approval(
        load_yaml(manifest_path), approval_path=activation_path
    )


def _require(value: Any, label: str, pattern: str | None = None) -> str:
    text = str(value or "").strip()
    if not text or (pattern and not re.fullmatch(pattern, text)):
        raise ValueError(f"Probe has invalid {label}")
    return text


def validate_probe(probe: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    image = _require(probe.get("image"), "image")
    if ":" not in image.rsplit("/", 1)[-1] or "@" in image:
        raise ValueError("Probe image must be a repository:tag reference")
    repository, tag = image.rsplit(":", 1)
    digest = _require(probe.get("digest"), "digest", r"sha256:[0-9a-f]{64}")
    try:
        size_bytes = int(probe["size_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Probe has invalid size_bytes") from exc
    if size_bytes <= 0:
        raise ValueError("Probe size_bytes must be positive")
    versions: dict[str, dict[str, str]] = {}
    for component in ("vllm", "vllm_ascend"):
        value = probe.get(component)
        if not isinstance(value, dict):
            raise ValueError(f"Probe is missing {component}")
        versions[component] = {
            "package": _require(value.get("package"), f"{component}.package"),
            "commit": _require(value.get("commit"), f"{component}.commit", r"[0-9a-f]{40}"),
        }
    portrait = existing.get("parameter_portrait", {})
    for component in ("vllm", "vllm_ascend"):
        expected = portrait.get(f"{component}_commit")
        if expected and versions[component]["commit"] != expected:
            raise ValueError(
                f"Probe {component} commit differs from parameter portrait; "
                "migrate the knowledge artifacts before approval"
            )
    evidence = probe.get("evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("Probe evidence must be a mapping")
    source_probe = _require(evidence.get("source_commit_probe"), "evidence.source_commit_probe")
    platform = probe.get("platform", {})
    if not isinstance(platform, dict):
        raise ValueError("Probe platform must be a mapping")
    return {
        "image": image,
        "repository": repository,
        "tag": tag,
        "digest": digest,
        "size_bytes": size_bytes,
        "versions": versions,
        "source_commit_probe": source_probe,
        "platform": platform,
    }


def build_documents(
    probe: dict[str, Any], existing: dict[str, Any], approved_by: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = validate_probe(probe, existing)
    approved_by = _require(approved_by, "approved_by")
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest = copy.deepcopy(existing)
    manifest["verified_at"] = timestamp
    manifest["platform"] = identity["platform"] or manifest.get("platform", {})
    manifest["source_image"] = {
        "reference": identity["image"],
        "pulled_digest": identity["digest"],
    }
    manifest["target_image"] = {
        "repository": identity["repository"],
        "tag": identity["tag"],
        "digest": identity["digest"],
        "size_bytes": identity["size_bytes"],
    }
    manifest["versions"] = identity["versions"]
    manifest["verification"] = {
        "package_versions_match": True,
        "source_commits_match": True,
        "remote_pull_verified": True,
        "model_catalog_pairing_verified": True,
        "portrait_reuse_approved": True,
    }
    activation = {
        "approved": True,
        "approved_at": timestamp,
        "approved_by": approved_by,
        "target": {
            "image": identity["image"],
            "image_digest": identity["digest"],
            "vllm_commit": identity["versions"]["vllm"]["commit"],
            "vllm_ascend_commit": identity["versions"]["vllm_ascend"]["commit"],
        },
        "evidence": {
            "package_versions": {
                name: value["package"] for name, value in identity["versions"].items()
            },
            "source_commit_probe": identity["source_commit_probe"],
        },
    }
    return manifest, activation


def _stage_yaml(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, allow_unicode=True, sort_keys=False)
    return Path(name)


def load_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.probe_json:
        value = json.loads(args.probe_json.read_text(encoding="utf-8"))
    else:
        command = json.loads(args.probe_command_file.read_text(encoding="utf-8"))
        if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
            raise ValueError("Probe command file must contain a non-empty JSON argv list")
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "Image probe command failed")
        value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("Image probe output must be a JSON object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    validate.add_argument("--activation", type=Path, default=DEFAULT_ACTIVATION)
    approve = commands.add_parser("approve")
    probes = approve.add_mutually_exclusive_group(required=True)
    probes.add_argument("--probe-json", type=Path)
    probes.add_argument("--probe-command-file", type=Path)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    approve.add_argument("--activation", type=Path, default=DEFAULT_ACTIVATION)
    approve.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "validate":
        validate_pair(args.manifest, args.activation)
        print("Image identity: OK")
        return 0
    manifest, activation = build_documents(
        load_probe(args), load_yaml(args.manifest), args.approved_by
    )
    if args.dry_run:
        print(yaml.safe_dump_all([manifest, activation], allow_unicode=True, sort_keys=False))
        return 0
    staged_manifest = _stage_yaml(args.manifest, manifest)
    staged_activation = _stage_yaml(args.activation, activation)
    os.replace(staged_manifest, args.manifest)
    os.replace(staged_activation, args.activation)
    validate_pair(args.manifest, args.activation)
    print("Image identity approved and validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
