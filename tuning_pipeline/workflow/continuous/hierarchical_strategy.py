"""Pure planning and assessment helpers for the V3 search profile.

This module intentionally has no SSH, task-submission, or Controller imports.
The live Controller enforces V3's selection limits while these future screening
helpers remain isolated until a reviewed Screen-to-Full state machine exists.
"""
from __future__ import annotations

import math
from typing import Any


def screening_plan(profile: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable, explicit benchmark subset for a V3 screen."""
    screening = profile["screening"]
    cases = [dict(case) for case in screening["selected_cases"]]
    if not cases or any(int(case.get("concurrency", 0)) != 32 for case in cases):
        raise ValueError("V3 screening must contain one or more explicit C32 cases")
    return {
        "stage": "screening",
        "benchmark_mode": screening["benchmark_mode"],
        "formal_cases": cases,
        "warmup_cases": cases,
        "acceptance_note": (
            "Screening only ranks candidates. A candidate cannot be accepted as "
            "an improvement until it passes full aligned_l1 verification."
        ),
    }


def validate_exploration_change_count(
    profile: dict[str, Any],
    independent_parameters: list[str],
    *,
    exception_reason: str | None = None,
) -> None:
    """Enforce V3's multi-parameter exploration rule for isolated planning."""
    exploration = profile["exploration"]
    lower, upper = [
        int(value) for value in exploration["independent_parameters_per_round"]
    ]
    count = len(independent_parameters)
    if lower <= count <= upper:
        return
    if (
        count == 1
        and exploration.get("one_parameter_exception_required", False)
        and exception_reason
        and exception_reason.strip()
    ):
        return
    if count == 1 and exploration.get("one_parameter_exception_required", False):
        raise ValueError(
            "V3 exploration requires two to three independent parameters; "
            "a one-parameter probe needs an explicit exception reason"
        )
    raise ValueError(
        f"V3 exploration requires {lower} to {upper} independent parameters; "
        f"got {count}"
    )


def _case_key(case: dict[str, Any]) -> tuple[str, int]:
    return str(case["workload"]), int(case["concurrency"])


def _geomean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        return 0.0
    return math.prod(values) ** (1.0 / len(values))


def assess_screening(
    profile: dict[str, Any],
    anchor_cases: list[dict[str, Any]],
    candidate_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply V3's conservative screen-to-full promotion gate.

    ``aggregate_output_tps`` and optional ``failed_requests`` are the required
    inputs. Missing, non-positive, or erroneous cases fail closed.
    """
    plan = screening_plan(profile)
    screening = profile["screening"]
    required = {_case_key(case) for case in plan["formal_cases"]}
    anchor = {_case_key(case): case for case in anchor_cases}
    candidate = {_case_key(case): case for case in candidate_cases}
    violations: list[str] = []
    ratios: dict[str, float] = {}
    for key in sorted(required):
        label = f"{key[0]}@c{key[1]}"
        before = anchor.get(key)
        after = candidate.get(key)
        if before is None or after is None:
            violations.append(f"missing required screen case {label}")
            continue
        if screening.get("require_zero_errors", True) and (
            int(after.get("failed_requests", 0) or 0) > 0
            or bool(after.get("has_errors", False))
        ):
            violations.append(f"screen case has errors: {label}")
            continue
        anchor_tps = float(before.get("aggregate_output_tps", 0) or 0)
        candidate_tps = float(after.get("aggregate_output_tps", 0) or 0)
        ratio = candidate_tps / anchor_tps if anchor_tps > 0 else 0.0
        ratios[label] = ratio
        if ratio < float(screening["minimum_each_workload_tps_ratio_vs_anchor"]):
            violations.append(f"{label} TPS ratio={ratio:.4f} below workload floor")
    geomean_ratio = _geomean(list(ratios.values()))
    if len(ratios) != len(required):
        geomean_ratio = 0.0
    if geomean_ratio < float(screening["minimum_geomean_tps_ratio_vs_anchor"]):
        violations.append(
            f"C32 geometric-mean TPS ratio={geomean_ratio:.4f} below promotion floor"
        )
    return {
        "stage": "screening",
        "promote_to_full_verification": not violations,
        "screen_geomean_tps_ratio_vs_anchor": geomean_ratio,
        "workload_tps_ratios": ratios,
        "violations": violations,
        "next_stage": "full_verification" if not violations else "exploration",
    }


def next_stage_after_full_verification(full_assessment: dict[str, Any]) -> str:
    """A full L1 gate is the only route from V3 exploration to refinement."""
    return (
        "local_refinement"
        if full_assessment.get("eligible_as_improvement")
        else "exploration"
    )
