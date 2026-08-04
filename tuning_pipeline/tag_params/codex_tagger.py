"""Resumable five-dimensional tag generation using Codex CLI authentication."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from threading import Lock

import yaml

from .schema import Tags, generate_tags_schema_text
from .utils import sanitize_filename


MODULE_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = MODULE_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
PORTRAIT_ROOT = PROJECT_ROOT / "portrait_pipeline"
DEFAULT_INPUT = PORTRAIT_ROOT / "outputs" / "ParameterYAML"
DEFAULT_OUTPUT = MODULE_DIR / "output"
SCHEMA_PATH = MODULE_DIR / "tags-output.schema.json"
TAGS_DEFINITION_PATH = MODULE_DIR / "resources" / "tags.yaml"
TARGET_CONTEXT = PORTRAIT_ROOT / "build" / "target-context.snapshot.yaml"
VLLM_COMMIT = "418bd6273c03bf48d5066733769e0a74bdc51694"
VLLM_ASCEND_COMMIT = "32c8cf190f596b47f0d0b965e64aea9f2b789ad4"
TAGGER_VERSION = "codex-five-dimension/v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_parameter_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"ParameterYAML input directory not found: {input_dir}")
    files = sorted(input_dir.glob("*.yaml"))
    if not files:
        raise RuntimeError(f"No ParameterYAML files found in: {input_dir}")

    logical_names: set[str] = set()
    output_names: set[str] = set()
    for path in files:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not value.get("name"):
            raise ValueError(f"Invalid ParameterYAML: {path}")
        logical_name = str(value["name"])
        output_name = f"{sanitize_filename(logical_name)}.yaml"
        if logical_name in logical_names:
            raise ValueError(f"Duplicate logical parameter name: {logical_name}")
        if output_name in output_names:
            raise ValueError(f"Sanitized output filename collision: {output_name}")
        logical_names.add(logical_name)
        output_names.add(output_name)
    return files


def load_progress(path: Path, files: list[Path]) -> dict[str, Any]:
    existing: dict[str, Any] = {}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
    old_items = existing.get("items", {})
    items: dict[str, Any] = {}

    for source in files:
        digest = file_sha256(source)
        item = dict(old_items.get(source.name, {}))
        if item.get("input_sha256") != digest:
            item = {
                "status": "pending",
                "name": None,
                "output_file": None,
                "attempts": 0,
                "error": None,
            }
        if item.get("status") == "in_progress":
            item["status"] = "pending"
            item["error"] = "recovered after interrupted Codex call"
        item["input_sha256"] = digest
        item["updated_at"] = utc_now()
        items[source.name] = item

    progress = {
        "schema_version": "codex-tag-progress/v2",
        "tagger_version": TAGGER_VERSION,
        "input_dir": str(files[0].parent),
        "output_dir": str(path.parent / "params"),
        "items": items,
    }
    refresh_summary(progress)
    return progress


def refresh_summary(progress: dict[str, Any]) -> None:
    counts = {"pending": 0, "in_progress": 0, "completed": 0, "error": 0}
    tagged_params: list[str] = []
    error_params: list[str] = []
    for item in progress["items"].values():
        status = item.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
        if status == "completed" and item.get("name"):
            tagged_params.append(str(item["name"]))
        elif status == "error" and item.get("name"):
            error_params.append(str(item["name"]))
    progress["summary"] = {"total": len(progress["items"]), **counts}
    # Compatibility with the original query/compiler partial-output whitelist.
    progress["tagged_params"] = sorted(tagged_params)
    progress["error_params"] = sorted(error_params)
    progress["updated_at"] = utc_now()


def resolve_codex() -> str:
    for name in ("codex.cmd", "codex"):
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError("Codex CLI was not found on PATH.")


def command_for_executable(executable: str, arguments: list[str]) -> list[str]:
    command = [executable, *arguments]
    if executable.lower().endswith((".cmd", ".bat")):
        return ["cmd.exe", "/d", "/c", *command]
    return command


def codex_version(codex: str) -> str:
    result = subprocess.run(
        command_for_executable(codex, ["--version"]),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=PIPELINE_ROOT,
        timeout=30,
    )
    return (result.stdout or result.stderr).strip() or "unknown"


def build_prompt(parameter_text: str) -> str:
    tag_schema = generate_tags_schema_text(TAGS_DEFINITION_PATH)
    return f"""You are an expert in vLLM performance tuning on Huawei Ascend NPU.
Assign the existing knowledge-base five-dimensional tags to exactly one
ParameterYAML record.

Tag schema:

{tag_schema}

Rules:

1. Judge every allowed tag value independently. A value applies when the
   parameter matters to a deployment matching that value.
2. The environment dimensions `model`, `hardware`, and `deploy_topology`
   default to broad applicability unless the ParameterYAML proves a
   restriction. For example, when no hardware restriction exists, use both
   `a2` and `a3`; when no topology restriction exists, use both `single_node`
   and `multi_node`.
3. The goal dimensions `optimize_target` and `deploy_scenario` require positive
   evidence that the parameter helps or materially affects that goal. Empty
   lists are valid for these dimensions when no such evidence exists.
4. Multiple values may apply. Use every applicable exact value and never invent
   values outside the schema.
5. Map generic latency evidence to `ttft` when prefill-focused and `tpot` when
   decode-focused. Use `throughput` for scheduling, batching, parallelism,
   communication, kernel, or compilation behavior that changes serving rate.
6. Base the decision on the full record, especially `impact_detail`,
   `constraints`, `tuning_advice`, `performance_scope`, `usage_locations`, and
   `related_parameters`.
7. Return every one of the five dimensions. The structured-output schema is
   authoritative. Return only the required JSON object.

ParameterYAML:
```yaml
{parameter_text}
```
"""


def prompt_sha256() -> str:
    return sha256_bytes(build_prompt("<PARAMETER_YAML>").encode("utf-8"))


def codex_command(codex: str, output_path: Path) -> list[str]:
    return command_for_executable(codex, [
        "exec", "--ephemeral", "--skip-git-repo-check",
        "--sandbox", "read-only", "-C", str(PIPELINE_ROOT),
        "--output-schema", str(SCHEMA_PATH),
        "--output-last-message", str(output_path), "-",
    ])


def tag_one(
    codex: str,
    source: Path,
    output_dir: Path,
    logs_dir: Path,
    attempt: int,
) -> tuple[dict[str, list[str]], str, str]:
    parameter_text = source.read_text(encoding="utf-8")
    parameter = yaml.safe_load(parameter_text)
    if not isinstance(parameter, dict) or not parameter.get("name"):
        raise ValueError(f"Invalid ParameterYAML: {source}")

    logical_name = str(parameter["name"])
    safe_name = sanitize_filename(logical_name)
    response_path = logs_dir / f"{safe_name}.response.json"
    stdout_path = logs_dir / f"{safe_name}.attempt-{attempt}.stdout.log"
    stderr_path = logs_dir / f"{safe_name}.attempt-{attempt}.stderr.log"
    response_path.unlink(missing_ok=True)
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        result = subprocess.run(
            codex_command(codex, response_path),
            input=build_prompt(parameter_text),
            stdout=stdout,
            stderr=stderr,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=PIPELINE_ROOT,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Codex exited with {result.returncode}; see {stderr_path.name}"
        )
    if not response_path.is_file():
        raise RuntimeError("Codex returned success without a structured response")
    raw = json.loads(response_path.read_text(encoding="utf-8"))
    tags_value = raw.get("tags", raw)
    tags = Tags(**tags_value).model_dump()

    tagged = dict(parameter)
    tagged["tags"] = tags
    output_name = f"{safe_name}.yaml"
    destination = output_dir / output_name
    destination.write_text(
        yaml.safe_dump(tagged, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )
    return tags, logical_name, output_name


def write_manifest(
    output_root: Path,
    input_dir: Path,
    progress: dict[str, Any],
    cli_version: str,
) -> None:
    distribution: dict[str, dict[str, int]] = {}
    for item in progress["items"].values():
        if item.get("status") != "completed" or not item.get("output_file"):
            continue
        path = output_root / "params" / str(item["output_file"])
        if not path.is_file():
            continue
        record = yaml.safe_load(path.read_text(encoding="utf-8"))
        for dimension, values in (record.get("tags") or {}).items():
            bucket = distribution.setdefault(dimension, {})
            for value in values or []:
                bucket[value] = bucket.get(value, 0) + 1
    manifest = {
        "schema_version": "codex-tag-manifest/v2",
        "build_timestamp": utc_now(),
        "builder": "Codex CLI (authenticated local agent)",
        "tagger_version": TAGGER_VERSION,
        "codex_cli_version": cli_version,
        "prompt_sha256": prompt_sha256(),
        "output_schema_sha256": file_sha256(SCHEMA_PATH),
        "tag_definition_sha256": file_sha256(TAGS_DEFINITION_PATH),
        "input_params_dir": str(input_dir),
        "output_params_dir": str(output_root / "params"),
        "tagging_results": progress["summary"],
        "tag_distribution": distribution,
        "source_commits": {
            "vllm": VLLM_COMMIT,
            "vllm-ascend": VLLM_ASCEND_COMMIT,
        },
        "target_context": str(TARGET_CONTEXT),
    }
    (output_root / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the original five-dimensional tags with Codex CLI."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-params", type=int, default=0)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of independent Codex CLI calls (each parameter remains one agent task).",
    )
    parser.add_argument(
        "--allow-incomplete-portraits",
        action="store_true",
        help="Allow a partial tagging test before the portrait queue completes.",
    )
    return parser.parse_args()


def ensure_portrait_queue_complete(input_dir: Path, allow_incomplete: bool) -> None:
    expected = PORTRAIT_ROOT / "outputs" / "ParameterYAML"
    if input_dir.resolve() != expected.resolve() or allow_incomplete:
        return
    index_path = PORTRAIT_ROOT / "build" / "codex_portrait_pipeline" / "run" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    summary = index.get("summary", {})
    unfinished = sum(int(summary.get(key, 0)) for key in ("pending", "in_progress", "error"))
    if unfinished:
        raise RuntimeError(
            f"Portrait queue is not complete ({unfinished} unfinished). Tagging was not started."
        )


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_attempts <= 10:
        raise ValueError("--max-attempts must be between 1 and 10")
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be between 1 and 8")
    input_dir = args.input.resolve()
    output_root = args.output.resolve()
    ensure_portrait_queue_complete(input_dir, args.allow_incomplete_portraits)
    files = load_parameter_files(input_dir)
    output_dir = output_root / "params"
    logs_dir = output_root / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.json"
    progress = load_progress(progress_path, files)
    if args.retry_errors:
        for item in progress["items"].values():
            if item.get("status") == "error":
                item["status"] = "pending"
    refresh_summary(progress)
    atomic_json(progress_path, progress)
    codex = resolve_codex()
    cli_version = codex_version(codex)

    eligible = []
    for source in files:
        status = progress["items"][source.name]["status"]
        if status == "completed" or (status == "error" and not args.retry_errors):
            continue
        eligible.append(source)
    if args.max_params:
        eligible = eligible[: args.max_params]

    state_lock = Lock()

    def persist() -> None:
        refresh_summary(progress)
        atomic_json(progress_path, progress)

    def process_source(source: Path) -> tuple[str, Exception | None]:
        item = progress["items"][source.name]
        last_error: Exception | None = None
        for local_attempt in range(1, args.max_attempts + 1):
            with state_lock:
                item["attempts"] = int(item.get("attempts", 0)) + 1
                attempt_number = int(item["attempts"])
                item.update(status="in_progress", error=None, updated_at=utc_now())
                persist()
            try:
                tags, name, output_file = tag_one(
                    codex, source, output_dir, logs_dir, attempt_number
                )
                with state_lock:
                    item.update(
                        status="completed",
                        name=name,
                        output_file=output_file,
                        tags=tags,
                        error=None,
                        updated_at=utc_now(),
                    )
                    persist()
                return source.name, None
            except Exception as exc:
                last_error = exc
                with state_lock:
                    item.update(
                        status="pending" if local_attempt < args.max_attempts else "error",
                        error=str(exc),
                        updated_at=utc_now(),
                    )
                    persist()
                print(
                    f"[attempt {local_attempt}/{args.max_attempts}] {source.name}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if local_attempt < args.max_attempts:
                    time.sleep(min(30, 2 ** local_attempt))
        return source.name, last_error

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_source, source) for source in eligible]
        for future in as_completed(futures):
            source_name, last_error = future.result()
            if last_error is not None:
                print(f"[error] {source_name}: {last_error}", file=sys.stderr, flush=True)
            with state_lock:
                persist()
                write_manifest(output_root, input_dir, progress, cli_version)
                summary = dict(progress["summary"])
            print(
                f"{summary['completed']}/{summary['total']} completed, "
                f"{summary['error']} errors",
                flush=True,
            )

    refresh_summary(progress)
    atomic_json(progress_path, progress)
    write_manifest(output_root, input_dir, progress, cli_version)
    summary = progress["summary"]
    if not args.max_params and any(summary[key] for key in ("pending", "in_progress", "error")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
