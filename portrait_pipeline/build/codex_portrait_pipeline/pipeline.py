"""Prepare, inspect, validate, and track Codex-authored parameter portraits.

This module performs no model API calls.  Codex reads a task package, examines
the pinned source trees, authors a draft YAML, and hands it back to ``accept``.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

MODULE_DIR = Path(__file__).resolve().parent
BUILD_ROOT = MODULE_DIR.parent
ROOT = BUILD_ROOT.parent
if str(BUILD_ROOT) not in sys.path:
    sys.path.insert(0, str(BUILD_ROOT))

from parse_params.schema import ParameterYAML, SkippedParamYAML
from parse_params.utils import sanitize_filename

DEFAULT_EXTRACTION = BUILD_ROOT / "extracted_parameters" / "parameters.structured.json"
DEFAULT_MANIFEST = BUILD_ROOT / "migration_candidates" / "reports" / "migration-manifest.json"
DEFAULT_RUN = MODULE_DIR / "run"
DEFAULT_PARAMETER_OUTPUT = ROOT / "outputs" / "ParameterYAML"
DEFAULT_SKIPPED_OUTPUT = ROOT / "outputs" / "skipped"
VLLM_ROOT = ROOT / "sources" / "vllm"
ASCEND_ROOT = ROOT / "sources" / "vllm-ascend"


def _parameter_output_dir(run_dir: Path) -> Path:
    if run_dir.resolve() == DEFAULT_RUN.resolve():
        return DEFAULT_PARAMETER_OUTPUT
    return run_dir / "params"


def _skipped_output_dir(run_dir: Path) -> Path:
    if run_dir.resolve() == DEFAULT_RUN.resolve():
        return DEFAULT_SKIPPED_OUTPUT
    return run_dir / "skipped"


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise SystemExit(f"Required input not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_filename(task_id: str) -> str:
    return f"{task_id}.json"


def prepare(extraction_path: Path, manifest_path: Path, run_dir: Path) -> None:
    extraction = _read_json(extraction_path)
    manifest = _read_json(manifest_path)
    rich_by_name = {
        str(item["name"]): item for item in extraction.get("parameters", [])
    }
    plans = manifest.get("candidate_plan", [])
    if not plans:
        raise SystemExit("Migration manifest has no candidate_plan")

    tasks_dir = run_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    filenames: dict[str, str] = {}
    for sequence, plan in enumerate(plans, 1):
        name = str(plan["name"])
        parameter = rich_by_name.get(name)
        if parameter is None:
            raise SystemExit(f"Manifest parameter missing from structured extraction: {name}")
        task_id = str(parameter["id"])
        output_filename = f"{sanitize_filename(name)}.yaml"
        previous = filenames.get(output_filename)
        if previous and previous != name:
            raise SystemExit(
                f"ParameterYAML filename collision: {previous!r} and {name!r} -> {output_filename}"
            )
        filenames[output_filename] = name
        task = {
            "schema_version": "codex-portrait-task/v1",
            "task_id": task_id,
            "sequence": sequence,
            "parameter": parameter,
            "migration": {
                "class": plan["migration_class"],
                "legacy_files": plan.get("legacy_files", []),
                "legacy_profiles": plan.get("legacy_profiles", []),
            },
            "target": {
                "output_schema": "parse_params.ParameterYAML",
                "output_filename": output_filename,
                "source_commits": extraction.get("sources", {}),
            },
        }
        _write_json(tasks_dir / _task_filename(task_id), task)
        rows.append({
            "sequence": sequence,
            "task_id": task_id,
            "name": name,
            "migration_class": plan["migration_class"],
            "output_filename": output_filename,
            "status": "pending",
            "attempts": 0,
            "last_error": None,
            "updated_at": _utc_now(),
        })

    index = {
        "schema_version": "codex-portrait-index/v1",
        "created_at": _utc_now(),
        "inputs": {
            "structured_extraction": str(extraction_path.resolve()),
            "migration_manifest": str(manifest_path.resolve()),
            "extraction_hash": extraction.get("extraction_hash"),
            "source_roots": {
                "vllm": extraction.get("sources", {}).get("vllm", {}).get("path"),
                "vllm_ascend": extraction.get("sources", {}).get("vllm_ascend", {}).get("path"),
            },
        },
        "summary": {"total": len(rows), "pending": len(rows), "completed": 0, "skipped": 0, "error": 0},
        "tasks": rows,
    }
    _write_json(run_dir / "index.json", index)
    (run_dir / "drafts").mkdir(exist_ok=True)
    _parameter_output_dir(run_dir).mkdir(parents=True, exist_ok=True)
    _skipped_output_dir(run_dir).mkdir(parents=True, exist_ok=True)
    (run_dir / "contexts").mkdir(exist_ok=True)
    print(f"Prepared {len(rows)} tasks in {run_dir}")


def sync_prepared_run(run_dir: Path, prepared_run: Path) -> None:
    """Merge newly extracted tasks without resetting completed portrait work."""
    current = _load_index(run_dir)
    prepared = _load_index(prepared_run)
    current_by_id = {str(row["task_id"]): row for row in current["tasks"]}
    merged = []
    added = []
    for row in prepared["tasks"]:
        task_id = str(row["task_id"])
        retained = current_by_id.get(task_id)
        if retained is None:
            retained = dict(row)
            added.append(str(row["name"]))
        else:
            # Keep execution state, but refresh all plan metadata.  Migration
            # aliases can legitimately change A/B/CURRENT_ONLY classification
            # without changing the stable parameter task id.
            retained = {
                **retained,
                "sequence": row["sequence"],
                "name": row["name"],
                "migration_class": row["migration_class"],
                "output_filename": row["output_filename"],
            }
        merged.append(retained)
        task = _read_json(prepared_run / "tasks" / _task_filename(task_id))
        _write_json(run_dir / "tasks" / _task_filename(task_id), task)
    removed = sorted(set(current_by_id) - {str(row["task_id"]) for row in prepared["tasks"]})
    if removed:
        raise SystemExit(f"Prepared run removed existing tasks: {removed}")
    current["inputs"] = prepared["inputs"]
    current["tasks"] = merged
    _save_index(run_dir, current)
    print(json.dumps({"added": added, "summary": current["summary"]}, ensure_ascii=False, indent=2))


def _load_index(run_dir: Path) -> dict:
    return _read_json(run_dir / "index.json")


def _save_index(run_dir: Path, index: dict) -> None:
    counts = {"pending": 0, "in_progress": 0, "completed": 0, "skipped": 0, "error": 0}
    for task in index["tasks"]:
        counts[task["status"]] = counts.get(task["status"], 0) + 1
    index["summary"] = {"total": len(index["tasks"]), **counts}
    _write_json(run_dir / "index.json", index)


def list_tasks(run_dir: Path, limit: int, status: str) -> None:
    index = _load_index(run_dir)
    matches = [item for item in index["tasks"] if item["status"] == status]
    for item in matches[:limit]:
        print(
            f"{item['sequence']:03d}\t{item['task_id']}\t"
            f"{item['migration_class']}\t{item['name']}"
        )
    print(f"shown={min(limit, len(matches))} matching={len(matches)}")


def _rg_hits(root: Path, patterns: list[str], limit: int = 24) -> list[str]:
    hits: list[str] = []
    for pattern in dict.fromkeys(value for value in patterns if value):
        try:
            result = subprocess.run(
                ["rg", "-n", "-F", "--glob", "*.py", "--glob", "*.md",
                 "--glob", "!**/tests/**", "--glob", "!**/test/**",
                 "--glob", "!**/examples/**", pattern, str(root)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        for line in result.stdout.splitlines():
            if line not in hits:
                hits.append(line)
            if len(hits) >= limit:
                return hits
    return hits


def materialize_context(run_dir: Path, task_id: str) -> Path:
    task = _read_json(run_dir / "tasks" / _task_filename(task_id))
    parameter = task["parameter"]
    name = str(parameter["name"])
    leaf = name.lstrip("-").replace("-", "_").rsplit(".", 1)[-1]
    patterns = [name, leaf]
    evidence = []
    for source in parameter.get("source_locations", []):
        evidence.append({
            "repository": source.get("repository"),
            "path": source.get("path"),
            "line": source.get("line"),
            "symbol": source.get("symbol"),
            "excerpt": source.get("excerpt"),
        })
    vllm_root, ascend_root = _source_roots(run_dir)
    context = {
        "schema_version": "codex-portrait-context/v1",
        "task_id": task_id,
        "name": name,
        "definition_evidence": evidence,
        "search_hits": {
            "vllm_ascend": _rg_hits(ascend_root, patterns),
            "vllm": _rg_hits(vllm_root, patterns),
        },
        "instructions": [
            "Treat pinned source code as ground truth.",
            "Check vllm-ascend behavior before upstream behavior.",
            "Use the legacy portrait only as a migration hint.",
            "Write the unchanged parse_params.ParameterYAML schema.",
        ],
    }
    path = run_dir / "contexts" / f"{task_id}.json"
    _write_json(path, context)
    print(path)
    return path


def _find_row(index: dict, task_id: str) -> dict:
    for row in index["tasks"]:
        if row["task_id"] == task_id:
            return row
    raise SystemExit(f"Unknown task id: {task_id}")


def _validate_quality(raw: dict) -> None:
    """Enforce semantic requirements documented by the original prompt."""
    impact = raw.get("performance_impact")
    if impact == "none":
        if not str(raw.get("skip_reason", "")).strip():
            raise ValueError("skip_reason must be meaningful when performance_impact is none")
        return
    allowed_value_types = {
        "int", "float", "bool", "str", "list[int]", "list[float]",
        "list[str]", "dict", "unknown",
    }
    if raw.get("value_type") not in allowed_value_types:
        raise ValueError(
            f"value_type must use the unchanged ParameterYAML enum, got {raw.get('value_type')!r}"
        )
    allowed_performance_scopes = {"latency", "throughput", "memory"}
    invalid_scopes = set(raw.get("performance_scope") or []) - allowed_performance_scopes
    if invalid_scopes:
        raise ValueError(f"invalid performance_scope values: {sorted(invalid_scopes)}")
    if not str(raw.get("impact_detail", "")).strip():
        raise ValueError("impact_detail must contain current-source evidence")
    advice = raw.get("tuning_advice") or {}
    suggestions = advice.get("suggested_values") or []
    caveats = advice.get("caveats") or []
    if impact in {"high", "medium"} and not suggestions:
        raise ValueError("high/medium impact requires at least one suggested value")
    if impact in {"high", "medium"} and not caveats:
        raise ValueError("high/medium impact requires at least one caveat")
    for number, suggestion in enumerate(suggestions, 1):
        if suggestion.get("value") is None:
            raise ValueError(f"suggested_values[{number}].value must not be null")
        if not str(suggestion.get("reason", "")).strip():
            raise ValueError(f"suggested_values[{number}].reason must not be empty")


def _source_roots(run_dir: Path) -> tuple[Path, Path]:
    inputs = _load_index(run_dir).get("inputs", {})
    roots = inputs.get("source_roots", {})
    vllm = Path(roots.get("vllm") or VLLM_ROOT).resolve()
    ascend = Path(roots.get("vllm_ascend") or ASCEND_ROOT).resolve()
    if not (vllm / "vllm").is_dir() or not (ascend / "vllm_ascend").is_dir():
        raise ValueError(
            f"Portrait queue source roots are invalid: vllm={vllm}, ascend={ascend}"
        )
    return vllm, ascend


def _source_reference_candidates(run_dir: Path, reference: str) -> list[Path]:
    """Resolve a repo-relative ``path:line`` source reference."""
    relative = re.sub(r":\d+(?::\d+)?$", "", reference.replace("\\", "/")).lstrip("./")
    vllm_root, ascend_root = _source_roots(run_dir)
    if relative.startswith("vllm/"):
        return [vllm_root / relative]
    if relative.startswith("vllm_ascend/"):
        return [ascend_root / relative]
    if relative.startswith("docs/"):
        return [ascend_root / relative, vllm_root / relative]
    return [ascend_root / relative, vllm_root / relative]


def _validate_source_references(run_dir: Path, raw: dict) -> None:
    """Require every reported source/usage path to exist in a pinned repo."""
    references = list(raw.get("source_file") or [])
    references.extend(
        location.get("file", "")
        for location in (raw.get("usage_locations") or [])
        if isinstance(location, dict)
    )
    invalid = []
    for reference in references:
        if not reference or not any(
            path.is_file()
            for path in _source_reference_candidates(run_dir, str(reference))
        ):
            invalid.append(str(reference))
    if invalid:
        raise ValueError(
            "source references must be repo-relative paths that exist in the pinned repositories: "
            + ", ".join(repr(item) for item in invalid)
        )


def claim(run_dir: Path, task_id: str) -> None:
    index = _load_index(run_dir)
    row = _find_row(index, task_id)
    if row["status"] in {"completed", "skipped"}:
        raise SystemExit(f"Task already finalized: {task_id} ({row['status']})")
    row["status"] = "in_progress"
    row["attempts"] += 1
    row["updated_at"] = _utc_now()
    _save_index(run_dir, index)
    materialize_context(run_dir, task_id)


def accept(run_dir: Path, task_id: str, draft: Path) -> None:
    index = _load_index(run_dir)
    row = _find_row(index, task_id)
    raw = yaml.safe_load(draft.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("Draft must contain one YAML mapping")
    expected = row["name"]
    if raw.get("name") != expected:
        raise SystemExit(f"Draft name mismatch: expected {expected!r}, got {raw.get('name')!r}")
    try:
        _validate_quality(raw)
        if raw.get("performance_impact") == "none":
            model = SkippedParamYAML(**raw)
            destination = _skipped_output_dir(run_dir) / row["output_filename"]
            status = "skipped"
        else:
            _validate_source_references(run_dir, raw)
            raw.setdefault("analysis_date", date.today().isoformat())
            model = ParameterYAML(**raw)
            destination = _parameter_output_dir(run_dir) / row["output_filename"]
            status = "completed"
    except Exception as exc:
        row["status"] = "error"
        row["last_error"] = str(exc)
        row["updated_at"] = _utc_now()
        _save_index(run_dir, index)
        raise SystemExit(f"Schema validation failed: {exc}") from exc
    destination.write_text(
        yaml.safe_dump(
            model.model_dump(exclude_none=False), allow_unicode=True,
            sort_keys=False, width=120,
        ),
        encoding="utf-8",
    )
    row["status"] = status
    row["last_error"] = None
    row["updated_at"] = _utc_now()
    _save_index(run_dir, index)
    print(f"{status}: {destination}")


def audit(run_dir: Path) -> None:
    index = _load_index(run_dir)
    errors: list[str] = []
    for row in index["tasks"]:
        if row["status"] not in {"completed", "skipped"}:
            continue
        output_dir = (
            _parameter_output_dir(run_dir)
            if row["status"] == "completed"
            else _skipped_output_dir(run_dir)
        )
        path = output_dir / row["output_filename"]
        if not path.is_file():
            errors.append(f"missing {row['task_id']}: {path}")
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if row["status"] == "completed":
                ParameterYAML(**data)
                _validate_source_references(run_dir, data)
            else:
                SkippedParamYAML(**data)
        except Exception as exc:
            errors.append(f"invalid {row['task_id']}: {exc}")
    print(json.dumps({"summary": index["summary"], "errors": errors}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Codex Agent ParameterYAML workflow")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--extraction", type=Path, default=DEFAULT_EXTRACTION)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p = sub.add_parser("sync")
    p.add_argument("--prepared-run", type=Path, required=True)
    p = sub.add_parser("list")
    p.add_argument("--status", default="pending",
                   choices=["pending", "in_progress", "completed", "skipped", "error"])
    p.add_argument("--limit", type=int, default=10)
    p = sub.add_parser("context")
    p.add_argument("task_id")
    p = sub.add_parser("claim")
    p.add_argument("task_id")
    p = sub.add_parser("accept")
    p.add_argument("task_id")
    p.add_argument("draft", type=Path)
    sub.add_parser("audit")
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_dir = args.run_dir.resolve()
    if args.command == "prepare":
        prepare(args.extraction.resolve(), args.manifest.resolve(), run_dir)
    elif args.command == "sync":
        sync_prepared_run(run_dir, args.prepared_run.resolve())
    elif args.command == "list":
        list_tasks(run_dir, args.limit, args.status)
    elif args.command == "context":
        materialize_context(run_dir, args.task_id)
    elif args.command == "claim":
        claim(run_dir, args.task_id)
    elif args.command == "accept":
        accept(run_dir, args.task_id, args.draft.resolve())
    elif args.command == "audit":
        audit(run_dir)
