from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from .history import analyze_history


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
CONTINUOUS_DIR = MODULE_DIR.parent / "continuous"


def portable_path(path: Path | None) -> str | None:
    """Store repository-relative provenance when an input belongs to this clone."""
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be an object: {path}")
    return value


def read_structured(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_previous_active_names(path: Path) -> list[str]:
    value = read_structured(path)
    if not isinstance(value, dict):
        raise ValueError("Previous selection must be a compiled result object")
    active = value.get("active_parameters")
    if isinstance(active, list):
        names = [
            item.get("canonical_name") if isinstance(item, dict) else item
            for item in active
        ]
        return [str(name) for name in names if name]
    limits = value.get("active_search_limits")
    if isinstance(limits, dict):
        return [str(name) for name in limits]
    raise ValueError("Previous selection has no active_parameters or active_search_limits")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dotted_get(value: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def canonical_token(name: str) -> str:
    """Produce a stable token for fallback alias comparison.

    Explicit registry aliases remain authoritative. This fallback only handles
    CLI spelling migrations such as ``--max-num-seqs`` vs ``max_num_seqs``.
    """
    return name.removeprefix("--").replace("-", "_").lower()


def load_knowledge_base(params_dir: Path) -> list[dict[str, Any]]:
    if not params_dir.is_dir():
        raise FileNotFoundError(f"Knowledge-base directory not found: {params_dir}")
    allowed_names: set[str] | None = None
    progress_path = params_dir.parent / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        tagged = progress.get("tagged_params")
        if isinstance(tagged, list):
            allowed_names = {str(name) for name in tagged}

    by_name: dict[str, dict[str, Any]] = {}
    for path in sorted(params_dir.glob("*.yaml")):
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if not isinstance(value, dict) or not value.get("name"):
            continue
        logical_name = str(value["name"])
        if allowed_names is not None and logical_name not in allowed_names:
            continue
        value["_knowledge_file"] = portable_path(path)
        by_name[logical_name] = value
    return list(by_name.values())


def exact_tag_match(param: dict[str, Any], dimension: str, wanted: Iterable[str]) -> bool:
    actual = dotted_get(param, f"tags.{dimension}", [])
    if not isinstance(actual, list):
        return False
    actual_set = {str(item).lower() for item in actual}
    return any(str(item).lower() in actual_set for item in wanted)


def recalled_for_scenario(
    param: dict[str, Any],
    scenario: dict[str, Any],
    impacts: set[str],
) -> bool:
    tags = scenario["recall_tags"]
    for dimension in (
        "hardware",
        "model",
        "deploy_topology",
        "optimize_target",
    ):
        wanted = tags.get(dimension, [])
        if wanted and not exact_tag_match(param, dimension, wanted):
            return False
    deploy_scenario = tags.get("deploy_scenario", [])
    if deploy_scenario and not exact_tag_match(param, "deploy_scenario", deploy_scenario):
        return False
    return str(param.get("performance_impact", "")).lower() in impacts


def evaluate_condition(condition: dict[str, Any], scenario: dict[str, Any]) -> bool:
    actual = dotted_get(scenario, str(condition["path"]))
    expected = condition.get("value")
    op = condition.get("op", "eq")
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op == "in":
        return actual in condition.get("values", [])
    if op == "not_in":
        return actual not in condition.get("values", [])
    if op == "gt":
        return actual is not None and actual > expected
    if op == "gte":
        return actual is not None and actual >= expected
    if op == "lt":
        return actual is not None and actual < expected
    if op == "lte":
        return actual is not None and actual <= expected
    if op == "truthy":
        return bool(actual)
    if op == "falsy":
        return not bool(actual)
    raise ValueError(f"Unsupported prerequisite operator: {op}")


def coerce_scalar(value: Any) -> Any | None:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null", "auto"}:
        return None if lower in {"none", "null"} else "auto"
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", text):
        number = float(text)
        return number if math.isfinite(number) else None
    if (
        len(text) >= 2
        and text[0] in "[{"
        and text[-1] in "]}"
    ):
        try:
            return json.loads(text.replace("'", '"'))
        except json.JSONDecodeError:
            return None
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", text):
        return text
    return None


def suggested_values(param: dict[str, Any]) -> list[Any]:
    advice = param.get("tuning_advice")
    suggestions = advice.get("suggested_values", []) if isinstance(advice, dict) else []
    values: list[Any] = []
    if isinstance(suggestions, list):
        for item in suggestions:
            if not isinstance(item, dict) or "value" not in item:
                continue
            parsed = coerce_scalar(item["value"])
            if parsed is not None and parsed not in values:
                values.append(parsed)
    return values


def stable_unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    rendered: set[str] = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if key not in rendered:
            rendered.add(key)
            result.append(value)
    return result


def machine_constraints() -> list[dict[str, Any]]:
    return [
        {
            "id": "long_prefill_within_batch_budget",
            "kind": "lte_or_disabled",
            "left": "long_prefill_token_threshold",
            "right": "max_num_batched_tokens",
            "disabled_value": 0,
        },
        {
            "id": "mtp_scheduler_budget",
            "kind": "product_lte",
            "factors": ["max_num_seqs", {"add": ["num_speculative_tokens", 1]}],
            "right": "max_num_batched_tokens",
        },
        {
            "id": "mtp_factor_divides_tensor_parallel",
            "kind": "divides_or_disabled",
            "divisor": {"add": ["num_speculative_tokens", 1]},
            "dividend_scenario": "topology.tensor_parallel_size",
            "disabled_parameter": "num_speculative_tokens",
            "disabled_value": 0,
        },
        {
            "id": "decode_context_parallel_divides_tensor_parallel",
            "kind": "divides",
            "divisor": "decode_context_parallel_size",
            "dividend_scenario": "topology.tensor_parallel_size",
            "only_if_present": True,
        },
        {
            "id": "physical_parallelism_within_available_devices",
            "kind": "product_lte_scenario",
            "factors": [
                "topology.data_parallel_size",
                "topology.tensor_parallel_size",
                "topology.pipeline_parallel_size",
                "prefill_context_parallel_size",
            ],
            "right_scenario": "topology.total_npu",
            "only_if_present": True,
            "required_candidate_parameters": ["prefill_context_parallel_size"],
        },
        {
            "id": "cudagraph_sizes_match_speculation_factor",
            "kind": "all_multiples",
            "values": "cudagraph_capture_sizes",
            "factor": {"add": ["num_speculative_tokens", 1]},
            "only_if_present": True,
        },
    ]


def _resolve_operand(
    operand: Any,
    candidate: dict[str, Any],
    scenario: dict[str, Any],
) -> Any:
    if isinstance(operand, dict) and "add" in operand:
        values = [
            _resolve_operand(item, candidate, scenario)
            for item in operand["add"]
        ]
        if any(value is None for value in values):
            return None
        return sum(values)
    if isinstance(operand, str):
        if operand in candidate:
            return candidate[operand]
        return dotted_get(scenario, operand)
    return operand


def validate_candidate(
    candidate: dict[str, Any],
    scenario: dict[str, Any],
    constraints: list[dict[str, Any]] | None = None,
) -> list[str]:
    violations: list[str] = []
    for rule in constraints or machine_constraints():
        kind = rule["kind"]
        required_names = [
            value
            for value in (
                rule.get("left"),
                rule.get("right"),
                rule.get("divisor"),
                rule.get("values"),
                rule.get("disabled_parameter"),
            )
            if isinstance(value, str) and "." not in value
        ]
        required_names.extend(
            str(name) for name in rule.get("required_candidate_parameters", [])
        )
        if rule.get("only_if_present") and any(name not in candidate for name in required_names):
            continue
        if kind == "lte_or_disabled":
            left = candidate.get(rule["left"])
            right = candidate.get(rule["right"])
            if left != rule["disabled_value"] and left is not None and right is not None and left > right:
                violations.append(rule["id"])
        elif kind == "product_lte":
            factors = [_resolve_operand(item, candidate, scenario) for item in rule["factors"]]
            right = candidate.get(rule["right"])
            if right is not None and all(isinstance(v, (int, float)) for v in factors):
                if math.prod(factors) > right:
                    violations.append(rule["id"])
        elif kind == "product_lte_scenario":
            factors = [
                _resolve_operand(item, candidate, scenario)
                for item in rule["factors"]
            ]
            right = dotted_get(scenario, str(rule["right_scenario"]))
            if isinstance(right, (int, float)) and all(
                isinstance(value, (int, float)) for value in factors
            ):
                if math.prod(factors) > right:
                    violations.append(rule["id"])
        elif kind in {"divides", "divides_or_disabled"}:
            if kind == "divides_or_disabled":
                disabled = candidate.get(rule["disabled_parameter"])
                if disabled == rule["disabled_value"]:
                    continue
            divisor = _resolve_operand(rule["divisor"], candidate, scenario)
            dividend = dotted_get(scenario, rule["dividend_scenario"])
            if not isinstance(divisor, int) or divisor <= 0 or not isinstance(dividend, int) or dividend % divisor:
                violations.append(rule["id"])
        elif kind == "all_multiples":
            values = candidate.get(rule["values"])
            factor = _resolve_operand(rule["factor"], candidate, scenario)
            if isinstance(values, list) and isinstance(factor, int) and factor > 0:
                if any(not isinstance(value, int) or value % factor for value in values):
                    violations.append(rule["id"])
        else:
            raise ValueError(f"Unsupported machine constraint: {kind}")
    return violations


class SearchSpaceCompiler:
    """Compile an offline, auditable search-space proposal.

    This class intentionally has no SSH, subprocess, platform, or controller
    dependencies. Its output is advisory until a separate integration step is
    explicitly implemented.
    """

    def __init__(
        self,
        *,
        knowledge_dir: Path,
        scenario_path: Path,
        registry_path: Path | None = None,
        policy_path: Path | None = None,
        history_path: Path | None = None,
        previous_selection_path: Path | None = None,
    ):
        self.knowledge_dir = knowledge_dir.resolve()
        self.scenario_path = scenario_path.resolve()
        self.registry_path = (registry_path or MODULE_DIR / "registry.yaml").resolve()
        self.policy_path = (policy_path or MODULE_DIR / "policy.yaml").resolve()
        self.history_path = history_path.resolve() if history_path else None
        self.previous_selection_path = (
            previous_selection_path.resolve() if previous_selection_path else None
        )
        self.scenario = read_yaml(self.scenario_path)
        self.registry = read_yaml(self.registry_path)
        self.policy = read_yaml(self.policy_path)
        self.parameters = load_knowledge_base(self.knowledge_dir)
        self.by_name = {str(param["name"]): param for param in self.parameters}
        self.by_token: dict[str, list[dict[str, Any]]] = {}
        for param in self.parameters:
            self.by_token.setdefault(canonical_token(str(param["name"])), []).append(param)

    def _find_knowledge(self, entry: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        for alias in entry.get("knowledge_names", []):
            if alias in self.by_name:
                return self.by_name[alias], alias
        for alias in entry.get("knowledge_names", []):
            matches = self.by_token.get(canonical_token(alias), [])
            if len(matches) == 1:
                return matches[0], str(matches[0]["name"])
        return None, None

    def _candidate_values(
        self,
        entry: dict[str, Any],
        knowledge: dict[str, Any],
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        canonical = entry["canonical_name"]
        baseline = dotted_get(self.scenario, f"baseline.{canonical}")
        configured = entry.get("candidate_values", [])
        learned = suggested_values(knowledge) if entry.get("accept_safe_suggestions") else []
        sources: list[dict[str, Any]] = []
        raw: list[Any] = []
        if baseline is not None:
            raw.append(baseline)
            sources.append({"source": "scenario_baseline", "values": [baseline]})
        if configured:
            raw.extend(configured)
            sources.append({"source": "registry_policy", "values": configured})
        if learned:
            raw.extend(learned)
            sources.append({"source": "knowledge_suggested_values", "values": learned})
        values = stable_unique(raw)
        allowed = entry.get("allowed_values")
        if isinstance(allowed, list):
            allowed_keys = {
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                for value in allowed
            }
            values = [
                value
                for value in values
                if json.dumps(value, ensure_ascii=False, sort_keys=True) in allowed_keys
            ]
        return values, sources

    def _prune_values(
        self,
        canonical: str,
        values: list[Any],
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        baseline = dict(self.scenario.get("baseline", {}))
        accepted: list[Any] = []
        rejected: list[dict[str, Any]] = []
        for value in values:
            candidate = {**baseline, canonical: value}
            violations = validate_candidate(candidate, self.scenario)
            if violations:
                rejected.append({"value": value, "violations": violations})
            else:
                accepted.append(value)
        return accepted, rejected

    def _availability(self, canonical: str, entry: dict[str, Any]) -> dict[str, Any]:
        capabilities = self.scenario.get("capabilities", {})
        verified = set(capabilities.get("verified_canonical_parameters", []))
        if canonical in verified:
            return {
                "status": "verified",
                "source": capabilities.get("source", "scenario_capability_snapshot"),
            }
        if entry.get("integration_status") == "existing":
            return {
                "status": "verified_by_existing_mainflow_contract",
                "source": "existing_controller_injection_contract",
            }
        return {
            "status": "unverified_for_current_image",
            "source": "registry_and_knowledge_only",
        }

    def compile(self) -> dict[str, Any]:
        impacts = {
            str(value).lower()
            for value in self.policy["recall"]["performance_impacts"]
        }
        recalled_names = {
            str(param["name"])
            for param in self.parameters
            if recalled_for_scenario(param, self.scenario, impacts)
        }
        supplemental = set(self.policy["recall"].get("supplemental_canonical_names", []))
        fixed: list[dict[str, Any]] = []
        eligible: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for entry in self.registry["parameters"]:
            canonical = str(entry["canonical_name"])
            knowledge, matched_name = self._find_knowledge(entry)
            if knowledge is None:
                rejected.append(
                    {"canonical_name": canonical, "reason": "knowledge_parameter_not_found"}
                )
                continue
            recall_source = (
                "tag_recall"
                if matched_name in recalled_names
                else ("policy_supplement" if canonical in supplemental else None)
            )
            if recall_source is None:
                rejected.append(
                    {
                        "canonical_name": canonical,
                        "knowledge_name": matched_name,
                        "reason": "not_recalled_for_scenario",
                    }
                )
                continue
            if knowledge.get("deprecated") and not entry.get("allow_deprecated", False):
                rejected.append(
                    {
                        "canonical_name": canonical,
                        "knowledge_name": matched_name,
                        "reason": "deprecated_parameter",
                    }
                )
                continue
            failed_prerequisites = [
                condition
                for condition in entry.get("prerequisites", [])
                if not evaluate_condition(condition, self.scenario)
            ]
            if failed_prerequisites:
                rejected.append(
                    {
                        "canonical_name": canonical,
                        "knowledge_name": matched_name,
                        "reason": "prerequisites_not_met",
                        "failed_prerequisites": failed_prerequisites,
                    }
                )
                continue
            if not isinstance(entry.get("injection"), dict):
                rejected.append(
                    {
                        "canonical_name": canonical,
                        "knowledge_name": matched_name,
                        "reason": "missing_injection_contract",
                    }
                )
                continue

            role = entry.get("role", "tunable")
            values, value_sources = self._candidate_values(entry, knowledge)
            values, pruned = self._prune_values(canonical, values)
            record = {
                "canonical_name": canonical,
                "knowledge_name": matched_name,
                "aliases": entry.get("knowledge_names", []),
                "category": knowledge.get("category"),
                "performance_impact": knowledge.get("performance_impact"),
                "risk": entry.get("risk", "medium"),
                "role": role,
                "recall_source": recall_source,
                "baseline": dotted_get(self.scenario, f"baseline.{canonical}"),
                "values": values,
                "value_sources": value_sources,
                "pruned_values": pruned,
                "injection": entry["injection"],
                "integration_status": entry.get("integration_status", "planned"),
                "availability": self._availability(canonical, entry),
                "knowledge_file": knowledge.get("_knowledge_file"),
                "constraints_evidence": knowledge.get("constraints", []),
            }
            if role != "tunable":
                record["reason"] = entry.get("fixed_reason", role)
                fixed.append(record)
                continue
            if len(values) < 2:
                rejected.append(
                    {
                        **record,
                        "reason": "fewer_than_two_safe_values_after_pruning",
                    }
                )
                continue
            risk = record["risk"]
            availability_verified = str(record["availability"]["status"]).startswith(
                "verified"
            )
            record["approval"] = (
                "auto_approved"
                if (
                    risk in self.policy["approval"]["auto_approve_risks"]
                    and availability_verified
                )
                else "human_required"
            )
            record["approval_reasons"] = []
            if risk not in self.policy["approval"]["auto_approve_risks"]:
                record["approval_reasons"].append(f"risk_is_{risk}")
            if not availability_verified:
                record["approval_reasons"].append(
                    "parameter_not_verified_for_current_image"
                )
            score = int(self.policy["scoring"]["impact"].get(
                str(record["performance_impact"]), 0
            ))
            score += int(self.policy["scoring"]["risk"].get(str(risk), 0))
            score += int(self.policy["scoring"]["integration"].get(
                str(record["integration_status"]), 0
            ))
            score += int(self.policy["scoring"]["category"].get(
                str(record["category"]), 0
            ))
            if record["baseline"] is not None:
                score += int(self.policy["scoring"].get("baseline_known_bonus", 0))
            record["base_activation_score"] = score
            record["history_score_adjustment"] = 0.0
            record["activation_score"] = float(score)
            eligible.append(record)

        activation = self.policy["activation"]
        target = min(
            int(activation["target_count"]),
            int(activation["maximum_count"]),
        )
        by_canonical = {item["canonical_name"]: item for item in eligible}

        def select_ranked(score_key: str, force_core: bool) -> list[dict[str, Any]]:
            ordered = sorted(
                eligible,
                key=lambda item: (
                    -float(item[score_key]),
                    str(item["canonical_name"]),
                ),
            )
            chosen: list[dict[str, Any]] = []
            names: set[str] = set()
            categories: dict[str, int] = {}
            if force_core:
                for canonical in activation.get("core_parameters", []):
                    item = by_canonical.get(canonical)
                    if item and canonical not in names:
                        chosen.append(item)
                        names.add(canonical)
                        category = str(item["category"])
                        categories[category] = categories.get(category, 0) + 1
            for item in ordered:
                if len(chosen) >= target:
                    break
                canonical = str(item["canonical_name"])
                category = str(item["category"])
                if canonical in names:
                    continue
                if categories.get(category, 0) >= int(
                    activation["max_per_category"]
                ):
                    continue
                chosen.append(item)
                names.add(canonical)
                categories[category] = categories.get(category, 0) + 1
            if len(chosen) < int(activation["minimum_count"]):
                for item in ordered:
                    if len(chosen) >= int(activation["minimum_count"]):
                        break
                    if item["canonical_name"] not in names:
                        chosen.append(item)
                        names.add(str(item["canonical_name"]))
            return chosen

        base_active = select_ranked("base_activation_score", True)
        history_analysis: dict[str, Any] | None = None
        rotation_audit: dict[str, Any] = {
            "enabled": False,
            "reason": "no_history_input",
            "swaps": [],
        }
        if self.history_path:
            history_analysis = analyze_history(
                self.history_path,
                baseline_params=dict(self.scenario.get("baseline", {})),
                candidate_values={
                    str(item["canonical_name"]): item["values"] for item in eligible
                },
                policy=self.policy.get("history_selection", {}),
            )
            for item in eligible:
                stats = history_analysis["parameters"][item["canonical_name"]]
                item["history_evidence"] = stats
                quarantined_keys = {
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    for value in stats.get("quarantined_values", [])
                }
                if quarantined_keys:
                    retained_values = [
                        value
                        for value in item["values"]
                        if json.dumps(
                            value, ensure_ascii=False, sort_keys=True
                        )
                        not in quarantined_keys
                    ]
                    item["history_pruned_values"] = stats["quarantined_values"]
                    if len(retained_values) >= 2:
                        item["values"] = retained_values
                    else:
                        item["history_evidence"]["reasons"].append(
                            "quarantine_not_applied_because_it_would_exhaust_dimension"
                        )
                item["history_score_adjustment"] = stats["score_adjustment"]
                item["activation_score"] = round(
                    float(item["base_activation_score"])
                    + float(stats["score_adjustment"]),
                    4,
                )

        active = list(base_active)
        if history_analysis and history_analysis["attributed_trial_count"] > 0:
            desired = select_ranked("activation_score", False)
            previous_names = (
                load_previous_active_names(self.previous_selection_path)
                if self.previous_selection_path
                else [str(item["canonical_name"]) for item in base_active]
            )
            current_names = [
                name for name in previous_names if name in by_canonical
            ][:target]
            for item in base_active:
                name = str(item["canonical_name"])
                if len(current_names) >= target:
                    break
                if name not in current_names:
                    current_names.append(name)
            desired_names = {str(item["canonical_name"]) for item in desired}
            rotation_policy = self.policy["history_selection"]["rotation"]
            maximum_swaps = int(rotation_policy["maximum_swaps_per_session"])
            minimum_margin = float(rotation_policy["minimum_score_margin"])
            minimum_core = int(rotation_policy["minimum_core_parameters_retained"])
            core = set(activation.get("core_parameters", []))
            incoming = sorted(
                [
                    item
                    for item in desired
                    if item["canonical_name"] not in current_names
                ],
                key=lambda item: (
                    -float(item["activation_score"]),
                    str(item["canonical_name"]),
                ),
            )
            swaps: list[dict[str, Any]] = []
            for candidate_in in incoming:
                if len(swaps) >= maximum_swaps:
                    break
                outgoing = sorted(
                    [
                        by_canonical[name]
                        for name in current_names
                        if name not in desired_names
                    ],
                    key=lambda item: (
                        float(item["activation_score"]),
                        str(item["canonical_name"]),
                    ),
                )
                for candidate_out in outgoing:
                    out_name = str(candidate_out["canonical_name"])
                    in_name = str(candidate_in["canonical_name"])
                    category_counts = {
                        category: sum(
                            str(by_canonical[name]["category"]) == category
                            for name in current_names
                        )
                        for category in {
                            str(by_canonical[name]["category"])
                            for name in current_names
                        }
                    }
                    in_category = str(candidate_in["category"])
                    out_category = str(candidate_out["category"])
                    projected_category_count = category_counts.get(
                        in_category, 0
                    ) + (0 if in_category == out_category else 1)
                    if projected_category_count > int(
                        activation["max_per_category"]
                    ):
                        continue
                    retained_core = sum(
                        name in core
                        for name in current_names
                        if name != out_name
                    ) + (in_name in core)
                    if retained_core < minimum_core:
                        continue
                    margin = float(candidate_in["activation_score"]) - float(
                        candidate_out["activation_score"]
                    )
                    if margin < minimum_margin:
                        continue
                    index = current_names.index(out_name)
                    current_names[index] = in_name
                    swaps.append(
                        {
                            "out": out_name,
                            "in": in_name,
                            "score_margin": round(margin, 4),
                            "out_reasons": candidate_out.get(
                                "history_evidence", {}
                            ).get("reasons", []),
                            "in_reasons": candidate_in.get(
                                "history_evidence", {}
                            ).get("reasons", []),
                        }
                    )
                    break
            active = [by_canonical[name] for name in current_names]
            rotation_audit = {
                "enabled": True,
                "reason": "history_contains_attributed_trials",
                "previous_selection_source": (
                    str(self.previous_selection_path)
                    if self.previous_selection_path
                    else "cold_start_base_selection"
                ),
                "desired_selection": [
                    item["canonical_name"] for item in desired
                ],
                "maximum_swaps_per_session": maximum_swaps,
                "minimum_score_margin": minimum_margin,
                "swaps": swaps,
            }
        elif history_analysis:
            rotation_audit = {
                "enabled": False,
                "reason": "history_has_no_attributable_parameter_changes",
                "swaps": [],
            }

        eligible.sort(
            key=lambda item: (
                -float(item["activation_score"]),
                str(item["canonical_name"]),
            )
        )
        selected = {str(item["canonical_name"]) for item in active}

        reserves = [
            {**item, "reserve_reason": "activation_budget_or_category_diversity"}
            for item in eligible
            if item["canonical_name"] not in selected
        ]
        active_limits = {
            str(item["canonical_name"]): item["values"]
            for item in active
        }
        naive_combination_count = math.prod(
            len(values) for values in active_limits.values()
        )
        mainflow_ready = [
            item["canonical_name"]
            for item in active
            if item["integration_status"] == "existing"
        ]
        availability_verified = [
            item
            for item in eligible
            if str(item["availability"]["status"]).startswith("verified")
        ]
        return {
            "schema_version": 2,
            "compiler_mode": "compiled_offline_and_integrated_at_session_creation",
            "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "inputs": {
                "scenario": portable_path(self.scenario_path),
                "scenario_sha256": file_sha256(self.scenario_path),
                "registry": portable_path(self.registry_path),
                "registry_sha256": file_sha256(self.registry_path),
                "policy": portable_path(self.policy_path),
                "policy_sha256": file_sha256(self.policy_path),
                "knowledge_dir": portable_path(self.knowledge_dir),
                "history": portable_path(self.history_path),
                "history_sha256": (
                    file_sha256(self.history_path) if self.history_path else None
                ),
                "previous_selection": (
                    portable_path(self.previous_selection_path)
                ),
                "previous_selection_sha256": (
                    file_sha256(self.previous_selection_path)
                    if self.previous_selection_path
                    else None
                ),
            },
            "scenario_id": self.scenario["scenario_id"],
            "summary": {
                "knowledge_parameters": len(self.parameters),
                "tag_recalled_parameters": len(recalled_names),
                "registry_parameters": len(self.registry["parameters"]),
                "eligible_tunable_parameters": len(eligible),
                "active_parameters": len(active),
                "reserve_parameters": len(reserves),
                "fixed_parameters": len(fixed),
                "rejected_parameters": len(rejected),
                "mainflow_ready_active_parameters": len(mainflow_ready),
                "availability_verified_tunable_parameters": len(
                    availability_verified
                ),
                "human_approval_active_parameters": sum(
                    item["approval"] == "human_required" for item in active
                ),
                "naive_active_combinations": naive_combination_count,
                "history_trials": (
                    history_analysis["trial_count"] if history_analysis else 0
                ),
                "attributed_history_trials": (
                    history_analysis["attributed_trial_count"]
                    if history_analysis
                    else 0
                ),
                "rotation_swaps": len(rotation_audit["swaps"]),
            },
            "active_search_limits": active_limits,
            "search_space": {
                "naive_discrete_combinations": naive_combination_count,
                "materialized": False,
                "trial_validation_required": True,
                "note": (
                    "The compiler keeps constraints executable instead of "
                    "enumerating the Cartesian product. Every future trial "
                    "must pass validate_candidate before execution."
                ),
            },
            "active_parameters": active,
            "reserve_candidates": reserves,
            "history_analysis": history_analysis,
            "rotation_audit": rotation_audit,
            "fixed_parameters": fixed,
            "rejected_parameters": rejected,
            "machine_constraints": machine_constraints(),
            "approval_queue": [
                item
                for item in active
                if item["approval"] == "human_required"
            ],
            "integration": {
                "connected_to_mainflow": False,
                "mainflow_ready_active_parameters": mainflow_ready,
                "note": (
                    "No generated value is consumed by workflow/continuous. "
                    "A future explicit adapter and user approval are required."
                ),
            },
        }


def write_outputs(result: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir = output_dir.resolve()
    continuous = CONTINUOUS_DIR.resolve()
    if output_dir == continuous or continuous in output_dir.parents:
        raise ValueError(
            "Refusing to write compiler output inside workflow/continuous; "
            "the independent compiler must not modify the live controller area"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    agent_limits = output_dir / "agent_search_limits.yaml"
    compiled = output_dir / "search_space.compiled.yaml"
    audit = output_dir / "audit.json"
    approval = output_dir / "approval_queue.yaml"
    rotation = output_dir / "rotation_report.yaml"
    manifest = output_dir / "manifest.json"
    agent_limits.write_text(
        yaml.safe_dump(
            {"search_limits": result["active_search_limits"]},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    compiled.write_text(
        yaml.safe_dump(result, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    audit.write_text(
        json.dumps(
            {
                "summary": result["summary"],
                "fixed_parameters": result["fixed_parameters"],
                "rejected_parameters": result["rejected_parameters"],
                "history_analysis": result["history_analysis"],
                "rotation_audit": result["rotation_audit"],
                "integration": result["integration"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    approval.write_text(
        yaml.safe_dump(
            {"approval_queue": result["approval_queue"]},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    rotation.write_text(
        yaml.safe_dump(
            {
                "rotation_audit": result["rotation_audit"],
                "parameter_history": (
                    result["history_analysis"]["parameters"]
                    if result["history_analysis"]
                    else {}
                ),
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": result["generated_at"],
                "files": {
                    path.name: file_sha256(path)
                    for path in (agent_limits, compiled, audit, approval, rotation)
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [agent_limits, compiled, audit, approval, rotation, manifest]
