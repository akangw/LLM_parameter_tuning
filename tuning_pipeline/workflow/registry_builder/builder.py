from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
CONTINUOUS_DIR = MODULE_DIR.parent / "continuous"


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be an object: {path}")
    return value


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_snapshot_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.yaml")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def dotted_get(value: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def exact_tag_match(
    param: dict[str, Any], dimension: str, wanted: Iterable[str]
) -> bool:
    actual = dotted_get(param, f"tags.{dimension}", [])
    if not isinstance(actual, list):
        return False
    actual_set = {str(item).lower() for item in actual}
    return any(str(item).lower() in actual_set for item in wanted)


def recalled_for_scenario(
    param: dict[str, Any], scenario: dict[str, Any], impacts: set[str]
) -> bool:
    tags = scenario.get("recall_tags", {})
    for dimension in (
        "hardware",
        "model",
        "deploy_topology",
        "optimize_target",
        "deploy_scenario",
    ):
        wanted = tags.get(dimension, [])
        if wanted and not exact_tag_match(param, dimension, wanted):
            return False
    return str(param.get("performance_impact", "")).lower() in impacts


def canonical_name(name: str, parameter_type: str) -> str:
    value = name.strip()
    if parameter_type == "cli":
        value = value.removeprefix("--")
    value = value.replace("-", "_")
    value = re.sub(r"[^A-Za-z0-9_.]+", "_", value).strip("_")
    return value or "unnamed_parameter"


def stable_unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    rendered: set[str] = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if key not in rendered:
            rendered.add(key)
            result.append(value)
    return result


def coerce_scalar(value: Any, value_type: str) -> Any | None:
    if value is None:
        return None
    if value_type == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
        return None
    if value_type == "int":
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        text = str(value).strip()
        return int(text) if re.fullmatch(r"-?\d+", text) else None
    if value_type == "float":
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None
    if value_type == "str":
        if isinstance(value, (list, dict)):
            return None
        text = str(value).strip()
        if text.lower() in {"none", "null"}:
            return None
        return text if text else None
    if value_type.startswith("list["):
        if isinstance(value, list):
            return value
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text.replace("'", '"'))
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, list) else None
    if value_type == "dict" and isinstance(value, dict):
        return value
    return None


def inferred_injection(
    param: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    name = str(param.get("name", ""))
    parameter_type = str(param.get("type", ""))
    value_type = str(param.get("value_type", "unknown"))
    reasons: list[str] = []
    if parameter_type == "cli" and name.startswith("--"):
        kind = "cli_bool_flag" if value_type == "bool" else "cli_value"
        return {"kind": kind, "flag": name, "confidence": "source_declared"}, reasons
    if parameter_type == "env" and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        kind = "env_bool" if value_type == "bool" else "env_value"
        return {"kind": kind, "name": name, "confidence": "source_declared"}, reasons
    if parameter_type == "nested" and "." in name:
        reasons.append("generic_nested_path_requires_controller_adapter")
        return {
            "kind": "nested_field",
            "path": name,
            "confidence": "portrait_inferred",
        }, reasons
    reasons.append("parameter_entrypoint_not_inferable")
    return None, reasons


def candidate_values(param: dict[str, Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    value_type = str(param.get("value_type", "unknown"))
    raw: list[Any] = []
    sources: list[dict[str, Any]] = []

    raw_default = param.get("default")
    default = coerce_scalar(raw_default, value_type)
    if raw_default is None or default is not None:
        raw.append(default)
        sources.append({"source": "portrait_default", "values": [default]})

    choices = param.get("valid_choices")
    parsed_choices: list[Any] = []
    if isinstance(choices, list):
        for choice in choices:
            parsed = coerce_scalar(choice, value_type)
            if choice is None or parsed is not None:
                parsed_choices.append(parsed)
    if parsed_choices:
        raw.extend(parsed_choices)
        sources.append({"source": "portrait_valid_choices", "values": parsed_choices})

    advice = param.get("tuning_advice")
    suggestions = advice.get("suggested_values", []) if isinstance(advice, dict) else []
    parsed_suggestions: list[Any] = []
    if isinstance(suggestions, list):
        for suggestion in suggestions:
            if not isinstance(suggestion, dict) or "value" not in suggestion:
                continue
            raw_value = suggestion["value"]
            parsed = coerce_scalar(raw_value, value_type)
            if raw_value is None or parsed is not None:
                parsed_suggestions.append(parsed)
    if parsed_suggestions:
        raw.extend(parsed_suggestions)
        sources.append(
            {
                "source": "portrait_suggested_values",
                "values": stable_unique(parsed_suggestions),
            }
        )

    if value_type == "bool":
        raw.extend([False, True])
        sources.append({"source": "boolean_domain", "values": [False, True]})
    return stable_unique(raw), sources


def infer_risk(param: dict[str, Any]) -> tuple[str, list[str]]:
    category = str(param.get("category", "other"))
    constraints = " ".join(str(value).lower() for value in param.get("constraints", []))
    high_categories = {"parallelism", "communication", "model", "hardware"}
    severe_tokens = (
        "system failure",
        "out of memory",
        "oom",
        "raises valueerror",
        "must match",
    )
    if category in high_categories or any(
        token in constraints for token in severe_tokens
    ):
        return "high", ["category_or_constraint_requires_conservative_review"]
    if str(param.get("performance_impact", "")).lower() == "high":
        return "medium", ["high_performance_impact"]
    return "medium", ["default_fail_closed_risk"]


class RegistryBuilder:
    """Build a non-executable registry proposal from tagged knowledge only."""

    def __init__(self, *, knowledge_dir: Path, scenario_path: Path, policy_path: Path):
        self.knowledge_dir = knowledge_dir.resolve()
        self.scenario_path = scenario_path.resolve()
        self.policy_path = policy_path.resolve()
        self.scenario = read_yaml(self.scenario_path)
        self.policy = read_yaml(self.policy_path)

    def _load_parameters(self) -> list[dict[str, Any]]:
        if not self.knowledge_dir.is_dir():
            raise FileNotFoundError(
                f"Knowledge directory not found: {self.knowledge_dir}"
            )
        allowed_names: set[str] | None = None
        progress_path = self.knowledge_dir.parent / "progress.json"
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            tagged = progress.get("tagged_params")
            if isinstance(tagged, list):
                allowed_names = {str(name) for name in tagged}
        by_name: dict[str, dict[str, Any]] = {}
        for path in sorted(self.knowledge_dir.glob("*.yaml")):
            try:
                value = yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError:
                continue
            if not isinstance(value, dict) or not value.get("name"):
                continue
            if allowed_names is not None and str(value["name"]) not in allowed_names:
                continue
            value["_knowledge_file"] = portable_path(path)
            by_name[str(value["name"])] = value
        return list(by_name.values())

    def build(self) -> dict[str, Any]:
        parameters = self._load_parameters()
        impacts = {
            str(value).lower()
            for value in self.policy.get("recall", {}).get(
                "performance_impacts", ["high"]
            )
        }
        recalled = [
            param
            for param in parameters
            if recalled_for_scenario(param, self.scenario, impacts)
        ]
        generated: list[dict[str, Any]] = []
        review: list[dict[str, Any]] = []
        unsupported: list[dict[str, Any]] = []
        names: dict[str, str] = {}

        for param in recalled:
            source_name = str(param["name"])
            parameter_type = str(param.get("type", "unknown"))
            canonical = canonical_name(source_name, parameter_type)
            if canonical in names and names[canonical] != source_name:
                unsupported.append(
                    {
                        "canonical_name": canonical,
                        "knowledge_name": source_name,
                        "reason": "canonical_name_collision",
                        "collides_with": names[canonical],
                    }
                )
                continue
            names[canonical] = source_name
            injection, injection_reasons = inferred_injection(param)
            values, value_sources = candidate_values(param)
            risk, risk_reasons = infer_risk(param)
            reasons: list[str] = []
            if param.get("deprecated"):
                reasons.append("deprecated_parameter")
            if injection is None:
                reasons.extend(injection_reasons)
            if str(param.get("value_type", "unknown")) in {"unknown", "dict"}:
                reasons.append("complex_or_unknown_value_type")
            if len(values) < 2:
                reasons.append("fewer_than_two_discrete_candidate_values")

            record = {
                "canonical_name": canonical,
                "knowledge_names": [source_name],
                "source_type": parameter_type,
                "value_type": param.get("value_type"),
                "category": param.get("category"),
                "scope": param.get("scope"),
                "performance_impact": param.get("performance_impact"),
                "candidate_values": values,
                "value_sources": value_sources,
                "risk": risk,
                "risk_reasons": risk_reasons,
                "injection": injection,
                "constraints_evidence": param.get("constraints", []),
                "related_parameters": param.get("related_parameters", []),
                "source_files": param.get("source_file", []),
                "knowledge_file": param.get("_knowledge_file"),
                "generation_status": "proposal_only",
                "availability": "unverified_for_current_image",
                "approval": "human_required",
            }
            if reasons:
                record["review_reasons"] = stable_unique(reasons + injection_reasons)
                if (
                    injection is None
                    or "complex_or_unknown_value_type" in reasons
                    or "deprecated_parameter" in reasons
                ):
                    unsupported.append(record)
                else:
                    review.append(record)
            else:
                if injection_reasons:
                    record["review_reasons"] = injection_reasons
                    review.append(record)
                else:
                    generated.append(record)

        return {
            "schema_version": 1,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "mode": "proposal_only_not_connected_to_mainflow",
            "inputs": {
                "knowledge_dir": portable_path(self.knowledge_dir),
                "scenario": portable_path(self.scenario_path),
                "policy": portable_path(self.policy_path),
                "scenario_id": self.scenario.get("scenario_id"),
                "recall_tags": self.scenario.get("recall_tags", {}),
                "performance_impacts": sorted(impacts),
                "knowledge_snapshot_sha256": directory_snapshot_sha256(
                    self.knowledge_dir
                ),
                "scenario_sha256": file_sha256(self.scenario_path),
                "policy_sha256": file_sha256(self.policy_path),
            },
            "summary": {
                "knowledge_parameters": len(parameters),
                "tag_recalled_parameters": len(recalled),
                "generated_candidates": len(generated),
                "review_required_candidates": len(review),
                "unsupported_candidates": len(unsupported),
                "existing_registry_dependency": False,
                "connected_to_mainflow": False,
            },
            "generated_candidates": generated,
            "review_queue": review,
            "unsupported": unsupported,
            "safety": {
                "executable": False,
                "requires_image_capability_verification": True,
                "requires_controller_injection_validation": True,
                "does_not_modify_existing_registry": True,
            },
        }


def write_outputs(result: dict[str, Any], output_dir: Path) -> list[Path]:
    output = output_dir.resolve()
    if output == CONTINUOUS_DIR.resolve() or CONTINUOUS_DIR.resolve() in output.parents:
        raise ValueError(
            "Registry proposal output cannot be written under workflow/continuous"
        )
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)
    files: list[Path] = []
    payloads = {
        "registry.generated.yaml": {
            "schema_version": result["schema_version"],
            "mode": result["mode"],
            "parameters": result["generated_candidates"],
        },
        "review_queue.yaml": {
            "schema_version": result["schema_version"],
            "parameters": result["review_queue"],
        },
        "unsupported.yaml": {
            "schema_version": result["schema_version"],
            "parameters": result["unsupported"],
        },
    }
    for filename, payload in payloads.items():
        path = output / filename
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=120),
            encoding="utf-8",
        )
        files.append(path)
    audit = output / "audit.json"
    audit.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    files.append(audit)
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": result["created_at"],
                "files": {path.name: file_sha256(path) for path in files},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    files.append(manifest)
    return files
