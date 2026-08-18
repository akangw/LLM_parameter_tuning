from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml


def _read_structured(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def _metrics(trial: dict[str, Any]) -> dict[str, Any]:
    metrics = trial.get("metrics", {})
    if isinstance(metrics, dict) and isinstance(metrics.get("metrics"), dict):
        metrics = metrics["metrics"]
    return metrics if isinstance(metrics, dict) else {}


def _status(trial: dict[str, Any], metrics: dict[str, Any]) -> str:
    explicit = str(trial.get("status", trial.get("outcome", ""))).lower()
    if explicit in {"success", "succeeded", "ok", "accepted"}:
        return "success"
    if explicit in {"failure", "failed", "error", "rejected"}:
        return "failure"
    parse_status = trial.get("parse_status")
    if parse_status is None and isinstance(trial.get("metrics"), dict):
        parse_status = trial["metrics"].get("parse_status")
    failed = metrics.get("failed_requests", 0)
    if parse_status not in {None, "ok"} or (isinstance(failed, (int, float)) and failed > 0):
        return "failure"
    return "success" if metrics else "failure"


def load_trials(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = _read_structured(path)
    metadata: dict[str, Any] = {}
    if isinstance(root, list):
        raw_trials = root
        metadata["input_format"] = "legacy_history_input_list"
    elif isinstance(root, dict) and isinstance(root.get("trials"), list):
        raw_trials = root["trials"]
        metadata = {
            key: value
            for key, value in root.items()
            if key != "trials"
        }
        metadata["input_format"] = "normalized_session_history"
    else:
        raise ValueError(
            "History must be a legacy history_input.json list or an object with trials"
        )
    trials: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_trials):
        if not isinstance(raw, dict) or not isinstance(raw.get("params"), dict):
            continue
        metrics = _metrics(raw)
        failure_decision = raw.get("failure_decision", {})
        failure = raw.get("failure", {})
        failure_classification = raw.get("failure_classification")
        attributed_parameters = [
            str(name)
            for name in raw.get("attributed_parameters", [])
        ] if isinstance(raw.get("attributed_parameters", []), list) else []
        if isinstance(failure_decision, dict):
            if failure_decision.get("classification") is not None:
                failure_classification = failure_decision.get("classification")
            changes = failure_decision.get("changes", [])
            if isinstance(changes, list) and changes:
                attributed_parameters = [
                    str(change["parameter"])
                    for change in changes
                    if isinstance(change, dict) and change.get("parameter")
                ]
        if failure_classification is None and isinstance(failure, dict):
            failure_classification = failure.get("classification")
        trials.append(
            {
                "trial_id": str(
                    raw.get("trial_id", raw.get("round", f"trial_{index:04d}"))
                ),
                "params": raw["params"],
                "metrics": metrics,
                "status": _status(raw, metrics),
                "failure_classification": (
                    str(failure_classification)
                    if failure_classification is not None
                    else None
                ),
                "attributed_parameters": attributed_parameters,
                "objective_gain_percent": raw.get("objective_gain_percent"),
            }
        )
    return trials, metadata


def _percent_change(current: Any, reference: Any, higher_is_better: bool) -> float | None:
    if not isinstance(current, (int, float)) or not isinstance(reference, (int, float)):
        return None
    if not math.isfinite(float(current)) or not math.isfinite(float(reference)):
        return None
    if reference == 0:
        return None
    change = (float(current) - float(reference)) / abs(float(reference)) * 100.0
    return change if higher_is_better else -change


def analyze_history(
    history_path: Path,
    *,
    baseline_params: dict[str, Any],
    candidate_values: dict[str, list[Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    trials, metadata = load_trials(history_path)
    primary_metric = str(policy.get("primary_metric", "total_token_throughput"))
    higher_is_better = bool(policy.get("higher_is_better", True))
    latency_metrics = list(policy.get("latency_guardrail_metrics", []))
    maximum_latency_regression = float(
        policy.get("maximum_latency_regression_percent", 20.0)
    )
    observations: dict[str, list[dict[str, Any]]] = {
        name: [] for name in candidate_values
    }
    previous_params = dict(baseline_params)
    previous_success_metrics: dict[str, Any] | None = None
    attributed_trials = 0
    ignored_failure_trials = 0
    attributable_failures = set(
        policy.get(
            "attributable_failure_classifications",
            [
                "parameter_invalid",
                "parameter_oom",
                "parameter_runtime",
                "parameter_regression",
            ],
        )
    )
    hard_failure_classifications = set(
        policy.get(
            "hard_value_failure_classifications",
            ["parameter_invalid", "parameter_oom"],
        )
    )
    conditional_exclusions: list[dict[str, Any]] = []
    conditional_exclusion_keys: set[str] = set()

    for trial in trials:
        params = trial["params"]
        changed = [
            name
            for name in candidate_values
            if name in params and params.get(name) != previous_params.get(name)
        ]
        metrics = trial["metrics"]
        success = trial["status"] == "success"
        failure_is_attributable = (
            trial["status"] == "failure"
            and trial.get("failure_classification") in attributable_failures
        )
        if trial["status"] == "failure" and not failure_is_attributable:
            ignored_failure_trials += 1
            # Do not advance the reference configuration: a same-parameter
            # retry must remain attributable to the original parameter change.
            continue
        if failure_is_attributable and trial.get("attributed_parameters"):
            changed = [
                name
                for name in trial["attributed_parameters"]
                if name in candidate_values
            ]
        if (
            failure_is_attributable
            and trial.get("failure_classification") in hard_failure_classifications
            and changed
        ):
            conditions = {
                str(name): value
                for name, value in params.items()
                if name in candidate_values
            }
            exclusion_key = json.dumps(
                conditions, ensure_ascii=False, sort_keys=True
            )
            if conditions and exclusion_key not in conditional_exclusion_keys:
                conditional_exclusion_keys.add(exclusion_key)
                conditional_exclusions.append(
                    {
                        "trial_id": trial["trial_id"],
                        "failure_classification": trial["failure_classification"],
                        "conditions": conditions,
                        "attributed_parameters": list(changed),
                    }
                )
        gain = trial.get("objective_gain_percent")
        if not isinstance(gain, (int, float)):
            gain = (
                _percent_change(
                    metrics.get(primary_metric),
                    previous_success_metrics.get(primary_metric),
                    higher_is_better,
                )
                if previous_success_metrics
                else None
            )
        latency_regressions = [
            -change
            for metric in latency_metrics
            if (
                change := (
                    _percent_change(
                        metrics.get(metric),
                        previous_success_metrics.get(metric),
                        False,
                    )
                    if previous_success_metrics
                    else None
                )
            )
            is not None
            and change < 0
        ]
        guardrail_violation = bool(
            latency_regressions
            and max(latency_regressions) > maximum_latency_regression
        )
        if changed:
            attributed_trials += 1
            attribution_weight = 1.0 / len(changed)
            for name in changed:
                observations[name].append(
                    {
                        "trial_id": trial["trial_id"],
                        "value": params.get(name),
                        "success": success,
                        "failure_classification": trial.get(
                            "failure_classification"
                        ),
                        "gain_percent": gain,
                        "guardrail_violation": guardrail_violation,
                        "attribution_weight": attribution_weight,
                        "changed_parameter_count": len(changed),
                    }
                )
        if success:
            # Failed configurations never become the comparison reference.
            # The next trial remains attributable against the last successful
            # configuration, including a same-parameter retry.
            previous_params.update(params)
            previous_success_metrics = metrics

    minimum_evidence = max(1, int(policy.get("minimum_evidence_trials", 3)))
    weights = policy.get("weights", {})
    result: dict[str, Any] = {}
    for name, values in candidate_values.items():
        items = observations[name]
        candidate_keys = {
            json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values
        }
        parameter_conditional_failures = [
            copy
            for copy in conditional_exclusions
            if name in copy.get("attributed_parameters", [])
        ]
        # A hard failure is evidence about the complete configuration that was
        # launched, not proof that each individual value is globally invalid.
        # Keep the value available for a different companion combination and
        # retain the failed observation in its evidence score.
        scoring_items = list(items)
        weighted_samples = sum(
            item["attribution_weight"] for item in scoring_items
        )
        failure_weight = sum(
            item["attribution_weight"]
            for item in scoring_items
            if not item["success"]
        )
        guardrail_weight = sum(
            item["attribution_weight"]
            for item in scoring_items
            if item["guardrail_violation"]
        )
        gains = [
            float(item["gain_percent"])
            for item in scoring_items
            if item["success"] and isinstance(item["gain_percent"], (int, float))
        ]
        mean_gain = statistics.fmean(gains) if gains else None
        gain_stddev = statistics.pstdev(gains) if len(gains) >= 2 else 0.0
        unique_tested = {
            json.dumps(item["value"], ensure_ascii=False, sort_keys=True)
            for item in scoring_items
            if json.dumps(item["value"], ensure_ascii=False, sort_keys=True)
            in candidate_keys
        }
        coverage = min(1.0, len(unique_tested) / max(1, len(values)))
        confidence = min(1.0, weighted_samples / minimum_evidence)
        failure_rate = failure_weight / weighted_samples if weighted_samples else 0.0
        guardrail_rate = (
            guardrail_weight / weighted_samples if weighted_samples else 0.0
        )
        adjustment = 0.0
        reasons: list[str] = []
        if attributed_trials and not items:
            adjustment += float(weights.get("unexplored_bonus", 8.0))
            reasons.append("unexplored_parameter_bonus")
        elif items:
            coverage_bonus = float(weights.get("coverage_deficit_bonus", 6.0)) * (
                1.0 - coverage
            )
            adjustment += coverage_bonus
            if coverage_bonus:
                reasons.append("incomplete_value_coverage_bonus")
            if mean_gain is not None:
                bounded_gain = max(-20.0, min(20.0, mean_gain))
                gain_score = (
                    bounded_gain
                    * float(weights.get("gain_per_percent", 0.8))
                    * confidence
                )
                adjustment += gain_score
                reasons.append(
                    "positive_measured_gain"
                    if gain_score > 0
                    else "non_positive_measured_gain"
                )
            failure_score = -float(weights.get("failure_penalty", 18.0)) * failure_rate
            guardrail_score = -float(
                weights.get("guardrail_penalty", 12.0)
            ) * guardrail_rate
            instability_score = -min(20.0, gain_stddev) * float(
                weights.get("instability_per_percent", 0.3)
            ) * confidence
            adjustment += failure_score + guardrail_score + instability_score
            if failure_score:
                reasons.append("observed_failures")
            if guardrail_score:
                reasons.append("latency_guardrail_regressions")
            if instability_score:
                reasons.append("unstable_measured_gain")
        if parameter_conditional_failures:
            reasons.append("conditional_hard_failure_recorded")
        result[name] = {
            "trial_count": len(items),
            "scoring_trial_count": len(scoring_items),
            "weighted_evidence": round(weighted_samples, 4),
            "success_count": sum(item["success"] for item in scoring_items),
            "failure_rate": round(failure_rate, 4),
            "guardrail_violation_rate": round(guardrail_rate, 4),
            "mean_gain_percent": (
                round(mean_gain, 4) if mean_gain is not None else None
            ),
            "gain_stddev_percent": round(gain_stddev, 4),
            "tested_values": len(unique_tested),
            "candidate_values": len(values),
            "coverage_ratio": round(coverage, 4),
            "confidence": round(confidence, 4),
            "score_adjustment": round(adjustment, 4),
            # Retained as empty compatibility fields for archived consumers.
            "quarantined_values": [],
            "already_excluded_failed_values": [],
            "conditional_failures": parameter_conditional_failures,
            "reasons": reasons,
            "observations": items,
        }
    return {
        "source": str(history_path.resolve()),
        "metadata": metadata,
        "trial_count": len(trials),
        "attributed_trial_count": attributed_trials,
        "ignored_non_parameter_failure_trials": ignored_failure_trials,
        "primary_metric": primary_metric,
        "conditional_exclusions": conditional_exclusions,
        "parameters": result,
    }
