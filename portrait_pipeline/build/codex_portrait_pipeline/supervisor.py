"""Continuously drain the portrait queue with the authenticated Codex CLI."""
from __future__ import annotations

import json
import argparse
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
RUN = Path(__file__).resolve().parent / "run"
INDEX = RUN / "index.json"
LOGS = RUN / "worker_logs"
CODEX_CMD = ""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_index() -> dict:
    return json.loads(INDEX.read_text(encoding="utf-8"))


def save_index(index: dict) -> None:
    counts = {
        "pending": 0, "in_progress": 0, "completed": 0,
        "skipped": 0, "error": 0,
    }
    for task in index["tasks"]:
        counts[task["status"]] = counts.get(task["status"], 0) + 1
    index["summary"] = {"total": len(index["tasks"]), **counts}
    INDEX.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def recover_interrupted_tasks() -> int:
    """Return interrupted/non-final tasks to the resumable queue on startup."""
    index = load_index()
    recovered = 0
    for row in index["tasks"]:
        if row["status"] not in {"in_progress", "error"}:
            continue
        row["status"] = "pending"
        row["last_error"] = None
        row["updated_at"] = now()
        recovered += 1
    if recovered:
        save_index(index)
    return recovered


def next_pending() -> dict | None:
    for row in load_index()["tasks"]:
        if row["status"] == "pending":
            return row
    return None


def write_status(value: dict) -> None:
    (RUN / "supervisor-status.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def worker_prompt(task_id: str) -> str:
    relative_run = RUN.resolve().relative_to(ROOT.resolve()).as_posix()
    roots = load_index().get("inputs", {}).get("source_roots", {})
    return f"""You are one worker in the offline ParameterYAML portrait queue.
Read build/codex_portrait_pipeline/AGENT_INSTRUCTIONS.md.
Process only task {task_id}.
The draft file already exists as a placeholder. Overwrite that exact file using
PowerShell/.NET file writing if apply_patch is unavailable in the Windows sandbox.
Claim the task, inspect its task/context and these exact pinned source repositories:
vLLM: {roots.get('vllm')}
vLLM-Ascend: {roots.get('vllm_ascend')}
write {relative_run}/drafts/{task_id}.yaml, then run this accept command and fix
every validation error:
python -m build.codex_portrait_pipeline --run-dir {relative_run} accept {task_id} {relative_run}/drafts/{task_id}.yaml
Do not process any other task. Do not
edit build/parse_params, ../tuning_pipeline, or the pinned source trees. Finish
only after accept succeeds.
"""


def run_one(row: dict) -> int:
    task_id = row["task_id"]
    LOGS.mkdir(parents=True, exist_ok=True)
    draft = RUN / "drafts" / f"{task_id}.yaml"
    if not draft.exists():
        draft.write_text(
            "# Codex worker placeholder; replace with the validated portrait.\n",
            encoding="utf-8",
        )
    prompt_path = LOGS / f"{task_id}.prompt.txt"
    out_path = LOGS / f"{task_id}.out.log"
    err_path = LOGS / f"{task_id}.err.log"
    prompt_path.write_text(worker_prompt(task_id), encoding="utf-8")
    write_status({
        "state": "running", "task_id": task_id, "name": row["name"],
        "sequence": row["sequence"], "started_at": now(),
    })
    args = [
        *(["cmd.exe", "/d", "/c"] if os.name == "nt" and CODEX_CMD.lower().endswith((".cmd", ".bat")) else []),
        CODEX_CMD, "exec", "--ephemeral",
        "--skip-git-repo-check", "--sandbox", "workspace-write",
        "-C", str(ROOT), "-",
    ]
    with (
        prompt_path.open("r", encoding="utf-8") as stdin,
        out_path.open("w", encoding="utf-8") as stdout,
        err_path.open("w", encoding="utf-8") as stderr,
    ):
        result = subprocess.run(
            args, cwd=ROOT, stdin=stdin, stdout=stdout, stderr=stderr,
            text=True, encoding="utf-8", errors="replace",
        )
    current = next(
        item for item in load_index()["tasks"] if item["task_id"] == task_id
    )
    success = current["status"] in {"completed", "skipped"}
    write_status({
        "state": "completed" if success else "worker_failed",
        "task_id": task_id, "name": row["name"], "sequence": row["sequence"],
        "worker_exit_code": result.returncode, "task_status": current["status"],
        "finished_at": now(),
    })
    return 0 if success else 1


def main() -> None:
    global RUN, INDEX, LOGS, CODEX_CMD
    parser = argparse.ArgumentParser(description="Drain an offline portrait queue with Codex")
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--codex-command", default="auto")
    args = parser.parse_args()
    RUN = args.run_dir.resolve()
    INDEX = RUN / "index.json"
    LOGS = RUN / "worker_logs"
    requested = os.environ.get("VLLMTKB_CODEX_COMMAND", "").strip() or args.codex_command
    CODEX_CMD = (
        shutil.which("codex.cmd") or shutil.which("codex") or ""
        if requested == "auto"
        else (str(Path(requested).expanduser()) if Path(requested).expanduser().is_file() else shutil.which(requested) or "")
    )
    if not CODEX_CMD:
        raise SystemExit("Codex CLI not found; login/install it or set VLLMTKB_CODEX_COMMAND")
    recovered = recover_interrupted_tasks()
    if recovered:
        write_status({
            "state": "recovered_after_restart",
            "recovered_tasks": recovered,
            "recovered_at": now(),
        })
    consecutive_failures = 0
    while True:
        row = next_pending()
        if row is None:
            write_status({"state": "queue_complete", "finished_at": now()})
            return
        result = run_one(row)
        if result == 0:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                write_status({
                    "state": "paused_after_failures",
                    "consecutive_failures": consecutive_failures,
                    "task_id": row["task_id"], "name": row["name"],
                    "paused_at": now(),
                })
                return
        time.sleep(2)


if __name__ == "__main__":
    main()
