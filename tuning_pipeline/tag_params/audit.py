"""Deterministic audit for the completed Codex Tagged YAML knowledge base."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .codex_tagger import DEFAULT_INPUT, DEFAULT_OUTPUT
from .schema import Tags


PIPELINE_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_PATH = (
    PIPELINE_ROOT
    / "workflow"
    / "search_space_compiler"
    / "scenario.glm52-a3-aligned-l1.yaml"
)
REGISTRY_PATH = (
    PIPELINE_ROOT / "workflow" / "search_space_compiler" / "registry.yaml"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def exact_tag_match(record: dict[str, Any], dimension: str, wanted: list[str]) -> bool:
    actual = {str(value).lower() for value in record["tags"].get(dimension, [])}
    return any(str(value).lower() in actual for value in wanted)


def audit(input_dir: Path, output_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    progress_path = output_root / "progress.json"
    if not progress_path.is_file():
        raise FileNotFoundError(f"Tag progress is missing: {progress_path}")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    summary = progress.get("summary", {})
    if any(int(summary.get(key, 0)) for key in ("pending", "in_progress", "error")):
        errors.append(f"tag queue is incomplete: {summary}")

    sources: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(input_dir.glob("*.yaml")):
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        sources[path.name] = (path, value)

    output_dir = output_root / "params"
    completed_records: list[dict[str, Any]] = []
    for source_name, (source_path, source_value) in sources.items():
        item = progress.get("items", {}).get(source_name)
        if not isinstance(item, dict) or item.get("status") != "completed":
            errors.append(f"missing completed progress item: {source_name}")
            continue
        output_name = item.get("output_file")
        output_path = output_dir / str(output_name)
        if not output_path.is_file():
            errors.append(f"missing tagged YAML: {output_path}")
            continue
        tagged = yaml.safe_load(output_path.read_text(encoding="utf-8"))
        if not isinstance(tagged, dict):
            errors.append(f"tagged YAML is not an object: {output_path}")
            continue
        tags_value = tagged.get("tags")
        try:
            tags = Tags(**tags_value).model_dump()
        except Exception as exc:
            errors.append(f"invalid tags in {output_path.name}: {exc}")
            continue
        without_tags = dict(tagged)
        without_tags.pop("tags", None)
        if without_tags != source_value:
            errors.append(f"ParameterYAML fields changed during tagging: {source_path.name}")
        completed_records.append(tagged)

    actual_outputs = list(output_dir.glob("*.yaml"))
    if len(actual_outputs) != len(sources):
        errors.append(
            f"tagged file count {len(actual_outputs)} != input count {len(sources)}"
        )

    distribution: dict[str, dict[str, int]] = {}
    for record in completed_records:
        for dimension, values in record["tags"].items():
            bucket = distribution.setdefault(dimension, {})
            for value in values:
                bucket[value] = bucket.get(value, 0) + 1

    scenario = yaml.safe_load(SCENARIO_PATH.read_text(encoding="utf-8"))
    recall_tags = scenario["recall_tags"]
    recalled = []
    for record in completed_records:
        if str(record.get("performance_impact", "")).lower() != "high":
            continue
        dimensions = ("hardware", "model", "deploy_topology", "optimize_target")
        if all(
            not recall_tags.get(dimension)
            or exact_tag_match(record, dimension, recall_tags[dimension])
            for dimension in dimensions
        ):
            recalled.append(str(record["name"]))

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    knowledge_names = {str(record["name"]) for record in completed_records}
    registry_matches: dict[str, str | None] = {}
    for entry in registry["parameters"]:
        matched = next(
            (
                str(alias)
                for alias in entry.get("knowledge_names", [])
                if str(alias) in knowledge_names
            ),
            None,
        )
        registry_matches[str(entry["canonical_name"])] = matched
        if matched is None:
            errors.append(
                f"registry parameter has no current portrait: {entry['canonical_name']}"
            )

    report = {
        "schema_version": "tag-audit/v1",
        "audited_at": utc_now(),
        "input_parameters": len(sources),
        "tagged_parameters": len(completed_records),
        "tag_distribution": distribution,
        "scenario_high_impact_recall_count": len(recalled),
        "scenario_high_impact_recalled_names": recalled,
        "registry_matches": registry_matches,
        "errors": errors,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit completed Tagged YAML files.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_root = args.output.resolve()
    report = audit(args.input.resolve(), output_root)
    report_path = output_root / "audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
