from __future__ import annotations

import copy
import fnmatch
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from workflow.search_space_compiler.compiler import dotted_get, evaluate_condition

from .builder import stable_unique


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_POLICY_PATH = MODULE_DIR / "compatibility_policy.yaml"


def _deep_merge_policy(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_policy(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def read_policy(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Compatibility policy must be an object: {path}")
    parent = value.pop("extends", None)
    if parent is not None:
        parent_path = (path.parent / str(parent)).resolve()
        if parent_path == path.resolve():
            raise ValueError("Compatibility policy cannot extend itself")
        value = _deep_merge_policy(read_policy(parent_path), value)
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported compatibility policy schema")
    return value


class CompatibilityValidator:
    """Deterministically turn source-legal values into scenario-safe values."""

    def __init__(
        self,
        *,
        scenario: dict[str, Any],
        policy_path: Path | None = None,
        policy: dict[str, Any] | None = None,
    ) -> None:
        self.scenario = scenario
        self.policy_path = (policy_path or DEFAULT_POLICY_PATH).resolve()
        self.policy = (
            copy.deepcopy(policy)
            if policy is not None
            else read_policy(self.policy_path)
        )
        if self.policy.get("schema_version") != 1:
            raise ValueError("Unsupported compatibility policy schema")
        normalization = self.policy.get("normalization", {})
        self.omit_tokens = {
            str(token).strip().lower() for token in normalization.get("omit_tokens", [])
        }
        self.placeholder_patterns = [
            re.compile(str(pattern), re.IGNORECASE)
            for pattern in normalization.get("placeholder_patterns", [])
        ]

    def _matching_rules(self, canonical: str) -> list[dict[str, Any]]:
        return [
            rule
            for rule in self.policy.get("parameter_rules", [])
            if fnmatch.fnmatchcase(
                canonical.lower(), str(rule.get("match", "")).lower()
            )
        ]

    def _is_placeholder(self, value: Any) -> bool:
        if isinstance(value, str):
            text = value.strip()
            return any(pattern.search(text) for pattern in self.placeholder_patterns)
        if isinstance(value, list):
            return any(self._is_placeholder(item) for item in value)
        if isinstance(value, dict):
            return any(self._is_placeholder(item) for item in value.values())
        return False

    def _normalize_value(
        self, value: Any, injection: dict[str, Any]
    ) -> tuple[Any, list[str]]:
        reasons: list[str] = []
        if isinstance(value, str) and value.strip().lower() in self.omit_tokens:
            if injection.get("kind") in {
                "env_value",
                "env_bool",
                "cli_value",
                "cli_list",
                "cli_bool_flag",
                "json_path",
            }:
                return None, ["normalized_omit_token_to_null_action"]
        if self._is_placeholder(value):
            raise ValueError("descriptive_placeholder_is_not_executable")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non_finite_number")
        return value, reasons

    def _baseline_for(self, canonical: str, rules: list[dict[str, Any]]) -> Any:
        baseline = dotted_get(self.scenario, f"baseline.{canonical}")
        if baseline is not None:
            return baseline
        for rule in rules:
            if rule.get("scenario_path"):
                value = dotted_get(self.scenario, str(rule["scenario_path"]))
                if value is not None:
                    return value
        return None

    def numeric_domain(
        self,
        canonical: str,
        baseline: Any,
        source_values: list[Any],
        *,
        include_source_values: bool = True,
    ) -> list[Any] | None:
        """Build a parameter-specific domain around an effective B0 anchor.

        ``None`` means the parameter has no numeric-domain policy and retains
        its source-derived values. Sequence capacity, batching, memory fraction
        and CUDAGraph sizes intentionally have different safe shapes.
        """
        rules = self.policy.get("numeric_candidate_domains", {})
        rule = rules.get(canonical) if isinstance(rules, dict) else None
        if not isinstance(rule, dict):
            return None
        if isinstance(baseline, bool) or not isinstance(baseline, (int, float)):
            return None
        kind = str(rule.get("kind", ""))
        if kind == "factor":
            values = [float(baseline) * float(factor) for factor in rule.get("factors", [])]
        elif kind == "offset":
            values = [float(baseline) + float(offset) for offset in rule.get("offsets", [])]
        elif kind == "fixed":
            values = list(rule.get("values", []))
        else:
            raise ValueError(
                f"Unsupported numeric candidate-domain kind for {canonical}: {kind!r}"
            )
        minimum = rule.get("minimum")
        maximum = rule.get("maximum")
        precision = rule.get("precision")
        integer = bool(rule.get("integer", False))
        normalized: list[Any] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Numeric candidate-domain value for {canonical} is not numeric: {value!r}"
                )
            if minimum is not None and value < float(minimum):
                continue
            if maximum is not None and value > float(maximum):
                continue
            if integer:
                value = int(round(value))
            elif precision is not None:
                value = round(float(value), int(precision))
            normalized.append(value)
        # B0 is always a legal starting point, even if a defensive bound in a
        # future policy would otherwise omit it.
        anchor: Any = int(baseline) if integer else baseline
        # Before B0, source values remain useful, source-verified provisional
        # evidence. After B0 the Controller disables this merge so the frozen
        # online domain is exclusively B0-anchored.
        source = (
            source_values
            if include_source_values and bool(rule.get("include_source_values", True))
            else []
        )
        return stable_unique([anchor, *normalized, *source])

    def _filter_special_domain(
        self, canonical: str, values: list[Any]
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        rejected: list[dict[str, Any]] = []
        if canonical == "speculative_config__attention_backend":
            # The upstream enum mixes CUDA, ROCm, XPU and CPU implementations.
            # For the pinned Ascend draft path, null keeps capability-aware
            # auto-selection and FLASH_ATTN is the only explicit experimental
            # selector with source evidence in the exact plugin tree.
            supported = {None, "FLASH_ATTN"}
            accepted = []
            for value in values:
                if value in supported:
                    accepted.append(value)
                else:
                    rejected.append(
                        {
                            "value": value,
                            "reason": "draft_attention_backend_not_supported_on_pinned_ascend",
                        }
                    )
            return stable_unique(accepted), rejected
        if canonical != "speculative_config__method":
            return values, rejected
        normalized: list[Any] = []
        mtp_enabled = bool(dotted_get(self.scenario, "features.mtp", False))
        repetitive = bool(
            dotted_get(self.scenario, "benchmark.repetitive_workload", False)
        )
        for value in values:
            effective = value
            if isinstance(value, str) and value.endswith("_mtp"):
                effective = "mtp"
            allowed = effective is None
            if effective == "mtp":
                allowed = mtp_enabled
            elif effective in {"ngram", "ngram_gpu", "suffix"}:
                allowed = repetitive
            if allowed:
                normalized.append(effective)
            else:
                rejected.append(
                    {
                        "value": value,
                        "reason": "speculative_method_not_proven_for_scenario",
                    }
                )
        return stable_unique(normalized), rejected

    def _candidate_domain_override(
        self,
        canonical: str,
        injection: dict[str, Any],
    ) -> tuple[list[Any] | None, list[dict[str, Any]]]:
        """Return a policy-owned executable domain for structured/non-numeric axes.

        Source extraction is intentionally conservative for list-valued parameters:
        it normally discovers the current list, but cannot invent useful alternative
        shapes.  Explicit overrides let the compatibility policy provide reviewed
        templates while preserving the same fail-closed normalization used for
        source-derived values.
        """
        overrides = self.policy.get("candidate_domain_overrides", {})
        raw_values = overrides.get(canonical) if isinstance(overrides, dict) else None
        if raw_values is None:
            return None, []
        if not isinstance(raw_values, list):
            raise ValueError(
                f"Candidate-domain override for {canonical} must be a list"
            )
        accepted: list[Any] = []
        rejected: list[dict[str, Any]] = []
        for raw in raw_values:
            try:
                value, _ = self._normalize_value(raw, injection)
            except ValueError as exc:
                rejected.append({"value": raw, "reason": str(exc)})
                continue
            accepted.append(value)
        return stable_unique(accepted), rejected

    def _filter_policy_values(
        self,
        values: list[Any],
        rules: list[dict[str, Any]],
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Apply scenario-gated value rules after executable normalization.

        A parameter can exist in both pinned source trees while only some of
        its values are safe for the selected model/topology.  Keeping this at
        value level avoids either admitting an invalid enum wholesale or
        permanently disabling a useful axis for future scenario profiles.
        """
        accepted: list[Any] = []
        rejected: list[dict[str, Any]] = []
        for value in values:
            rejection_reason: str | None = None
            value_key = json.dumps(value, ensure_ascii=False, sort_keys=True)
            for rule in rules:
                for value_rule in rule.get("value_rules", []):
                    gated_keys = {
                        json.dumps(item, ensure_ascii=False, sort_keys=True)
                        for item in value_rule.get("values", [])
                    }
                    if value_key not in gated_keys:
                        continue
                    condition = value_rule.get("require")
                    if isinstance(condition, dict) and not evaluate_condition(
                        condition, self.scenario
                    ):
                        rejection_reason = str(
                            value_rule.get(
                                "reason", "scenario_value_gate_not_met"
                            )
                        )
                        break
                if rejection_reason:
                    break
            if rejection_reason:
                rejected.append({"value": value, "reason": rejection_reason})
            else:
                accepted.append(value)
        return accepted, rejected

    def validate_parameter(self, parameter: dict[str, Any]) -> dict[str, Any]:
        canonical = str(parameter["canonical_name"])
        rules = self._matching_rules(canonical)
        for rule in rules:
            condition = rule.get("require")
            if isinstance(condition, dict) and not evaluate_condition(
                condition, self.scenario
            ):
                return {
                    "accepted": False,
                    "parameter": None,
                    "audit": {
                        "canonical_name": canonical,
                        "status": "excluded_fail_closed",
                        "reason": "scenario_feature_gate_not_met",
                        "condition": condition,
                    },
                }

        result = copy.deepcopy(parameter)
        source_canonical = canonical
        for rule in rules:
            if rule.get("rename_to"):
                canonical = str(rule["rename_to"])
                result["canonical_name"] = canonical
                break
        for rule in rules:
            if rule.get("risk_override"):
                result["risk"] = str(rule["risk_override"])
                break
        injection = result["injection"]
        accepted: list[Any] = []
        rejected_values: list[dict[str, Any]] = []
        normalization_events: list[dict[str, Any]] = []
        baseline = self._baseline_for(canonical, rules)
        for raw in result.get("candidate_values", []):
            try:
                value, reasons = self._normalize_value(raw, injection)
            except ValueError as exc:
                rejected_values.append({"value": raw, "reason": str(exc)})
                continue
            if reasons:
                normalization_events.append(
                    {"input": raw, "output": value, "reasons": reasons}
                )
            accepted.append(value)

        accepted, policy_rejections = self._filter_policy_values(accepted, rules)
        rejected_values.extend(policy_rejections)
        accepted, domain_rejections = self._filter_special_domain(canonical, accepted)
        rejected_values.extend(domain_rejections)
        numeric_domain = self.numeric_domain(canonical, baseline, accepted)
        if numeric_domain is not None:
            accepted = numeric_domain
        domain_override, override_rejections = self._candidate_domain_override(
            canonical, injection
        )
        if domain_override is not None:
            accepted = domain_override
            rejected_values.extend(override_rejections)
        if baseline is not None:
            accepted.insert(0, baseline)
        accepted = stable_unique(accepted)

        role = "tunable"
        fixed_reason: str | None = None
        for rule in rules:
            if rule.get("role"):
                role = str(rule["role"])
                fixed_reason = f"compatibility_policy:{role}"
                break
            role_when = rule.get("role_when")
            if (
                isinstance(role_when, dict)
                and isinstance(role_when.get("condition"), dict)
                and evaluate_condition(role_when["condition"], self.scenario)
            ):
                role = str(role_when["role"])
                fixed_reason = f"compatibility_policy:{role}"
                break
        if role != "tunable" and baseline is not None:
            accepted = [baseline]
        minimum_values = 2 if role == "tunable" else 1
        if len(accepted) < minimum_values:
            return {
                "accepted": False,
                "parameter": None,
                "audit": {
                    "canonical_name": canonical,
                    "status": "excluded_fail_closed",
                    "reason": "fewer_than_required_compatible_values",
                    "role": role,
                    "compatible_values": accepted,
                    "rejected_values": rejected_values,
                },
            }

        result["candidate_values"] = accepted
        result["integration_status"] = "generated_compatible"
        if role != "tunable":
            result["role"] = role
            result["fixed_reason"] = fixed_reason
        result.setdefault("generation", {})["compatibility"] = {
            "policy": self.policy_path.name,
            "status": "compatible",
            "role": role,
            "baseline": baseline,
            "source_canonical_name": source_canonical,
            "normalization_events": normalization_events,
            "rejected_values": rejected_values,
            "numeric_domain_policy_applied": numeric_domain is not None,
            "candidate_domain_override_applied": domain_override is not None,
        }
        return {
            "accepted": True,
            "parameter": result,
            "audit": {
                "canonical_name": canonical,
                "source_canonical_name": source_canonical,
                "status": "accepted",
                "role": role,
                "compatible_values": accepted,
                "normalization_events": normalization_events,
                "rejected_values": rejected_values,
                "numeric_domain_policy_applied": numeric_domain is not None,
                "candidate_domain_override_applied": domain_override is not None,
            },
        }

    def validate_combination(self, candidate: dict[str, Any]) -> list[str]:
        violations: list[str] = []
        for rule in self.policy.get("combination_constraints", []):
            kind = rule.get("kind")
            if kind == "require_if_active":
                value = candidate.get(str(rule["parameter"]))
                if value in rule.get("inactive_values", []):
                    continue
                if str(rule["parameter"]) not in candidate:
                    continue
                required_name = str(rule["required_parameter"])
                required = candidate.get(required_name)
                op = rule.get("required_op")
                expected = rule.get("required_value")
                if op == "gt":
                    valid = required is not None and required > expected
                elif op == "not_in":
                    valid = required not in expected
                elif op == "eq":
                    valid = required == expected
                elif op == "in":
                    valid = required in expected
                elif op == "truthy":
                    valid = bool(required)
                elif op == "falsy":
                    valid = not bool(required)
                else:
                    raise ValueError(f"Unsupported require_if_active operator: {op}")
                if not valid:
                    violations.append(str(rule["id"]))
            elif kind == "mutually_exclusive_true":
                if all(candidate.get(str(name)) is True for name in rule["parameters"]):
                    violations.append(str(rule["id"]))
            elif kind == "forbidden_pair":
                left, right = rule["left"], rule["right"]
                left_matches = candidate.get(str(left["parameter"])) == left.get(
                    "value"
                )
                right_value = candidate.get(str(right["parameter"]))
                right_matches = right_value not in right.get("not_in", [])
                if left_matches and right_matches:
                    violations.append(str(rule["id"]))
            else:
                raise ValueError(f"Unsupported compatibility constraint kind: {kind}")
        return violations

    @property
    def combination_constraints(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.policy.get("combination_constraints", []))
