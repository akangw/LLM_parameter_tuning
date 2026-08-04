"""Self-contained version-to-portrait migration entry point.

The command never replaces accepted production portraits. Each version pair is
built under an isolated, commit-qualified run directory and can be audited
before activation.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
PORTRAIT = ROOT / "portrait_pipeline"
BASE_CONTEXT = PORTRAIT / "build" / "target-context.snapshot.yaml"


def run(command: list[str], *, cwd: Path = ROOT) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode:
        raise SystemExit(f"Command failed ({completed.returncode}): {subprocess.list2cmdline(command)}")


def capture(command: list[str]) -> str:
    completed = subprocess.run(
        command, cwd=ROOT, text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=False,
    )
    if completed.returncode:
        raise SystemExit(completed.stderr.strip() or "Git command failed")
    return completed.stdout.strip()


def checkout(target: Path, url: str, ref: str) -> tuple[Path, str]:
    """Resolve a ref in a migration-only checkout, never the active source tree."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_dir():
        run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(target)])
    run(["git", "-C", str(target), "fetch", "--filter=blob:none", "origin", ref])
    commit = capture(["git", "-C", str(target), "rev-parse", "FETCH_HEAD"])
    run(["git", "-C", str(target), "checkout", "--detach", commit])
    return target, commit


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Fetch two source versions and build an isolated parameter-portrait migration"
    )
    result.add_argument("--vllm", required=True, help="vLLM tag, branch, or commit")
    result.add_argument("--vllm-ascend", required=True, help="vLLM-Ascend tag, branch, or commit")
    result.add_argument("--provider", choices=["codex", "anthropic"], default="codex")
    result.add_argument("--prepare-only", action="store_true", help="prepare/audit the queue without LLM execution")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--concurrency", type=int, default=8)
    result.add_argument("--run-root", type=Path, default=PORTRAIT / "build" / "version_migrations")
    return result


def main() -> int:
    args = parser().parse_args()
    source_root = args.run_root.resolve() / "_source_checkouts"
    vllm_path, vllm_commit = checkout(
        source_root / "vllm", "https://github.com/vllm-project/vllm.git", args.vllm
    )
    ascend_path, ascend_commit = checkout(
        source_root / "vllm-ascend",
        "https://github.com/vllm-project/vllm-ascend.git",
        args.vllm_ascend,
    )
    run_id = f"vllm-{vllm_commit[:12]}_ascend-{ascend_commit[:12]}"
    run_root = args.run_root.resolve() / run_id
    extract_dir = run_root / "01_extract"
    candidates_dir = run_root / "02_migration"
    queue_dir = run_root / "03_portrait_queue"
    run_root.mkdir(parents=True, exist_ok=True)

    context = copy.deepcopy(yaml.safe_load(BASE_CONTEXT.read_text(encoding="utf-8")))
    context["context_id"] = run_id
    context["release"]["vllm_commit"] = vllm_commit
    context["release"]["vllm_ascend_commit"] = ascend_commit
    context_path = run_root / "target-context.yaml"
    context_path.write_text(
        yaml.safe_dump(context, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "schema_version": "version-migration-run/v1",
        "requested_refs": {"vllm": args.vllm, "vllm_ascend": args.vllm_ascend},
        "resolved_commits": {"vllm": vllm_commit, "vllm_ascend": ascend_commit},
        "provider": args.provider,
        "activation_status": "isolated_not_activated",
    }
    (run_root / "run-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    migration = [
        sys.executable, "-m", "build.migration_pipeline",
        "--vllm-src", str(vllm_path), "--vllm-ascend-src", str(ascend_path),
        "--legacy-dir", str(PORTRAIT / "outputs" / "ParameterYAML"),
        "--extract-output", str(extract_dir), "--output", str(candidates_dir),
        "--target-context", str(context_path), "--concurrency", str(args.concurrency),
    ]
    if args.resume:
        migration.append("--resume")
    if args.provider == "codex" or args.prepare_only:
        migration.append("--dry-run")
    run(migration, cwd=PORTRAIT)

    if args.provider == "codex":
        run([
            sys.executable, "-m", "build.codex_portrait_pipeline", "--run-dir", str(queue_dir),
            "prepare", "--extraction", str(extract_dir / "parameters.structured.json"),
            "--manifest", str(candidates_dir / "reports" / "migration-manifest.json"),
        ], cwd=PORTRAIT)
        if not args.prepare_only:
            run([
                sys.executable, "-m", "build.codex_portrait_pipeline.supervisor",
                "--run-dir", str(queue_dir),
            ], cwd=PORTRAIT)
            run([
                sys.executable, "-m", "build.codex_portrait_pipeline", "--run-dir", str(queue_dir), "audit"
            ], cwd=PORTRAIT)
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
