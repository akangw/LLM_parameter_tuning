#!/usr/bin/env python3
"""CI-safe configuration, portability and high-confidence secret checks."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = (
    "scripts/controller.sh",
    "docker/controller/Dockerfile",
    "docker-compose.controller.yml",
    "tuning_pipeline/workflow/continuous/topology_profiles.yaml",
    "tuning_pipeline/workflow/continuous/runtime_profiles.yaml",
    "tuning_pipeline/workflow/continuous/executor_profiles.yaml",
    "tuning_pipeline/workflow/continuous/remote/image_version_manifest.yaml",
)
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "OpenAI key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "GitHub token": re.compile(r"\bgh[oprsu]_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}
ASSIGNMENT = re.compile(
    r"(?im)^\s*(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*"
    r"[\"']?([^\s#\"']{12,})"
)
PLACEHOLDER_MARKERS = (
    "${",
    "env",
    "getenv",
    "settings.",
    "replace",
    "configure",
    "example",
    "placeholder",
    "your-",
)


def repository_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line]


def main() -> int:
    errors: list[str] = []
    files = repository_files()
    relative_names = {path.relative_to(ROOT).as_posix() for path in files}
    for required in REQUIRED:
        if required not in relative_names:
            errors.append(f"missing required portability file: {required}")
    if "tuning_pipeline/workflow/continuous/config.local.yaml" in relative_names:
        errors.append("config.local.yaml must not be committed")

    for path in files:
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(ROOT).as_posix()
        suffix = path.suffix.lower()
        try:
            if suffix in {".yaml", ".yml"}:
                list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
            elif suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError, json.JSONDecodeError) as exc:
            errors.append(f"invalid configuration {relative}: {exc}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"possible {label} in {relative}")
        for match in ASSIGNMENT.finditer(text):
            value = match.group(1).lower()
            if not any(marker in value for marker in PLACEHOLDER_MARKERS):
                errors.append(f"possible assigned secret in {relative}")
                break

    sys.path.insert(0, str(ROOT / "tuning_pipeline"))
    try:
        from workflow.continuous.continuous_tuning import validate_activation_approval
        from workflow.continuous.runtime_profile import resolve_runtime_profile
        from workflow.continuous.topology_profile import resolve_topology_profile

        continuous = ROOT / "tuning_pipeline" / "workflow" / "continuous"
        config = yaml.safe_load((continuous / "config.yaml").read_text(encoding="utf-8"))
        config, _ = resolve_runtime_profile(config, ROOT / "tuning_pipeline")
        resolve_topology_profile(config, ROOT / "tuning_pipeline")
        manifest = yaml.safe_load(
            (continuous / "remote" / "image_version_manifest.yaml").read_text(encoding="utf-8")
        )
        validate_activation_approval(
            manifest, approval_path=continuous / "activation.approved.yaml"
        )
    except Exception as exc:
        errors.append(f"cross-file validation failed: {exc}")

    if errors:
        print("Repository validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Repository validation: OK ({len(files)} files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
