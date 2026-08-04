"""Classify old ParameterYAML portraits against a newly extracted parameter set."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def _normal(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower().lstrip("-"))


def _value(value: object) -> str:
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "null":
            value = "none"
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _legacy_profiles(directory: Path) -> list[tuple[Path, dict]]:
    profiles = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if isinstance(data, dict) and isinstance(data.get("name"), str):
            profiles.append((path, data))
    return profiles


_STRUCTURAL_PREFIXES = {
    "attention": "AttentionConfig",
    "compilation": "CompilationConfig",
    "ec_transfer": "ECTransferConfig",
    "eplb": "EPLBConfig",
    "kernel": "KernelConfig",
    "kv_events": "KVEventsConfig",
    "kv_transfer": "KVTransferConfig",
    "kv_transfer_config": "KVTransferConfig",
    "pooler": "PoolerConfig",
    "profiler": "ProfilerConfig",
    "speculative": "SpeculativeConfig",
    "structured_outputs": "StructuredOutputsConfig",
    "weight_transfer": "WeightTransferConfig",
}

_EXPLICIT_LEGACY_REPLACEMENTS = {
    # The 0706 extractor interpreted this documented additional_config key as
    # a CLI flag.  The pinned implementation exposes only the nested key.
    "--SLO_limits_for_dynamic_batch": "additional_config.SLO_limits_for_dynamic_batch",
}


def _structural_alias(name: str) -> str | None:
    """Map the old flattened config namespace to current class identities."""
    if "." not in name:
        return None
    prefix, suffix = name.split(".", 1)
    current = _STRUCTURAL_PREFIXES.get(prefix)
    return f"{current}.{suffix}" if current else None


def classify(current: list[dict], selected: list[dict], legacy_dir: Path) -> dict:
    """Build a deterministic migration manifest.

    C and D describe legacy portraits. Candidate plans deliberately use only
    A/B/CURRENT_ONLY so every selected current parameter has one action.
    """
    full = {str(p["name"]): p for p in current}
    selected_names = {str(p["name"]) for p in selected}
    replacement_targets = {
        str(p["replaces_deprecated"]): str(p["name"])
        for p in current
        if p.get("replaces_deprecated")
    }
    normalized: dict[str, list[str]] = defaultdict(list)
    for name in full:
        normalized[_normal(name)].append(name)

    rows, by_target = [], defaultdict(list)
    for path, old in _legacy_profiles(legacy_dir):
        old_name = str(old["name"])
        candidates = [old_name]
        if old_name.startswith("--no-"):
            candidates.append("--" + old_name[5:])
        if not old_name.startswith("--"):
            candidates.append("--" + old_name)
        target = next((name for name in candidates if name in full), None)
        method = "direct"
        structural = _structural_alias(old_name)
        if target is None and structural in full:
            target, method = structural, "structural_alias"
        explicit = _EXPLICIT_LEGACY_REPLACEMENTS.get(old_name)
        if target is None and explicit in full:
            target, method = explicit, "corrected_legacy_interface"
        if target is None and len(normalized[_normal(old_name)]) == 1:
            target, method = normalized[_normal(old_name)][0], "normalized"
        if target is None:
            grade, reason = "D", "not_present_in_current_extraction"
        elif target not in selected_names:
            grade, reason = "C", "excluded_by_current_stage1"
        else:
            new = full[target]
            same_type = old.get("type") == new.get("type")
            same_scope = old.get("scope") in {new.get("scope"), None, "both", "all"}
            same_default = _value(old.get("default")) == _value(new.get("default"))
            grade = "A" if method == "direct" and same_type and same_scope and same_default else "B"
            reason = None
        row = {
            "legacy_file": path.name, "legacy_name": old_name,
            "target_name": target, "grade": grade, "mapping_method": method if target else None,
            "skip_reason": reason,
            "differences": {
                "type": target is not None and old.get("type") != full[target].get("type"),
                "scope": target is not None and old.get("scope") not in {full[target].get("scope"), None, "both", "all"},
                "default": target is not None and _value(old.get("default")) != _value(full[target].get("default")),
            },
        }
        rows.append(row)
        if grade in {"A", "B"} and target:
            by_target[target].append({"profile": old, "row": row})
        replacement_target = replacement_targets.get(old_name)
        if replacement_target in selected_names:
            by_target[replacement_target].append({
                "profile": old,
                "row": {
                    **row,
                    "target_name": replacement_target,
                    "grade": "B",
                    "mapping_method": "deprecated_replacement",
                    "skip_reason": None,
                },
            })

    plan = []
    for param in selected:
        name = str(param["name"])
        matches = by_target.get(name, [])
        migration_class = "CURRENT_ONLY" if not matches else (
            "B" if any(item["row"]["grade"] == "B" for item in matches) else "A"
        )
        plan.append({
            "name": name, "migration_class": migration_class,
            "legacy_profiles": [item["profile"] for item in matches],
            "legacy_files": [item["row"]["legacy_file"] for item in matches],
        })
    return {
        "schema_version": "portrait-migration-manifest/v1",
        "legacy_directory": str(legacy_dir.resolve()),
        "summary": {
            "legacy_profiles": len(rows), "current_candidates": len(selected),
            "legacy_grade_counts": dict(Counter(row["grade"] for row in rows)),
            "candidate_plan_counts": dict(Counter(row["migration_class"] for row in plan)),
        },
        "profiles": rows, "candidate_plan": plan,
    }


def add_migration_context(params: list[dict], manifest: dict) -> list[dict]:
    """Attach private migration data consumed only by the isolated prompt builder."""
    plans = {item["name"]: item for item in manifest["candidate_plan"]}
    enriched = []
    for param in params:
        item = dict(param)
        item["_migration"] = plans[str(param["name"])]
        enriched.append(item)
    return enriched
