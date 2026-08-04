"""Build Codex tags, audit them, and stop after compiling Search Limits."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STATUS = ROOT / "build-status.json"
TAG_OUTPUT = ROOT / "tag_params" / "output"
SEARCH_OUTPUT = ROOT / "search_limits"
PIPELINE_LOGS = TAG_OUTPUT / "logs" / "pipeline"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(state: str, **extra: object) -> None:
    value = {"state": state, "updated_at": utc_now(), **extra}
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(STATUS)


def run_stage(
    name: str,
    command: list[str],
    stdout_name: str,
    stderr_name: str,
    *,
    check: bool = True,
) -> int:
    write_status(name, command=command)
    PIPELINE_LOGS.mkdir(parents=True, exist_ok=True)
    with (
        (PIPELINE_LOGS / stdout_name).open("a", encoding="utf-8") as stdout,
        (PIPELINE_LOGS / stderr_name).open("a", encoding="utf-8") as stderr,
    ):
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    if check and result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    return result.returncode


def main() -> None:
    try:
        # A whole-queue retry is deliberately separate from the per-parameter
        # retries in codex_tagger. This survives transient Codex/network failure
        # without discarding any completed checkpoints.
        for queue_pass in range(1, 6):
            return_code = run_stage(
                "tagging",
                [
                    sys.executable,
                    "-m",
                    "tag_params",
                    "--retry-errors",
                    "--max-attempts",
                    "3",
                    "--workers",
                    "8",
                ],
                "pipeline-tagging.stdout.log",
                "pipeline-tagging.stderr.log",
                check=False,
            )
            if return_code == 0:
                break
            if queue_pass == 5:
                raise RuntimeError(
                    "tagging remained incomplete after 5 resumable queue passes"
                )
            delay = 30 * queue_pass
            write_status(
                "tagging_retry_wait",
                queue_pass=queue_pass,
                next_queue_pass=queue_pass + 1,
                delay_seconds=delay,
            )
            time.sleep(delay)
        run_stage(
            "auditing_tags",
            [sys.executable, "-m", "tag_params.audit"],
            "pipeline-audit.stdout.log",
            "pipeline-audit.stderr.log",
        )
        manifest = SEARCH_OUTPUT / "manifest.json"
        if not manifest.is_file():
            if SEARCH_OUTPUT.exists():
                raise RuntimeError(
                    f"Incomplete Search Limits directory already exists: {SEARCH_OUTPUT}"
                )
            run_stage(
                "compiling_search_limits",
                [
                    sys.executable,
                    "-m",
                    "workflow.search_space_compiler",
                    "--knowledge-dir",
                    str(TAG_OUTPUT / "params"),
                    "--scenario",
                    str(ROOT / "workflow/search_space_compiler/scenario.glm52-a3-aligned-l1.yaml"),
                    "--registry",
                    str(ROOT / "workflow/search_space_compiler/registry.yaml"),
                    "--policy",
                    str(ROOT / "workflow/search_space_compiler/policy.yaml"),
                    "--output",
                    str(SEARCH_OUTPUT),
                ],
                "pipeline-search-limits.stdout.log",
                "pipeline-search-limits.stderr.log",
            )
        write_status(
            "search_limits_complete",
            tagged_params=str(TAG_OUTPUT / "params"),
            tag_audit=str(TAG_OUTPUT / "audit.json"),
            search_limits=str(SEARCH_OUTPUT),
            finished_at=utc_now(),
        )
    except Exception as exc:
        write_status("paused_on_error", error=str(exc))
        raise


if __name__ == "__main__":
    main()
