"""Fail-closed topology filtering and Agent-owned topology selection.

Topology changes require a new Session/Lease. This module deliberately does not
submit jobs or edit controller state; it produces the frozen first-layer plan
consumed before a serving-parameter Session is created.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def load_document(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict) or not isinstance(value.get("profiles"), dict):
        raise ValueError("topology profile document must contain profiles")
    return value


def build_plan(
    document: dict[str, Any],
    *,
    model_contract: str,
    available_nodes: int,
    npu_per_node: int,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for name, raw in document["profiles"].items():
        profile = dict(raw)
        blockers: list[str] = []
        if profile.get("status") != "integrated":
            blockers.append(f"status={profile.get('status', 'missing')}")
        validation = str(profile.get("validation", "missing"))
        if validation not in {
            "production_proven",
            "scenario_specific",
            "experimental_preflight_required",
        }:
            blockers.append(f"validation={profile.get('validation', 'missing')}")
        contracts = [str(item) for item in profile.get("model_contracts", [])]
        if model_contract not in contracts:
            blockers.append(f"model_contract={model_contract} is not validated")
        if int(profile.get("nodes", 0) or 0) > available_nodes:
            blockers.append("insufficient physical nodes")
        if int(profile.get("npu_per_node", 0) or 0) != npu_per_node:
            blockers.append("NPU count per node differs")
        blockers.extend(str(item) for item in profile.get("blockers", []))
        if blockers:
            eligibility = "blocked"
        elif validation == "experimental_preflight_required":
            preflight = profile.get("preflight", {})
            if not isinstance(preflight, dict) or preflight.get("required") is not True:
                blockers.append("experimental topology lacks a required preflight contract")
                eligibility = "blocked"
            else:
                eligibility = "experimental_eligible"
        else:
            eligibility = "production_eligible"
        candidates.append(
            {
                "profile": str(name),
                "label": str(profile.get("novice_label", name)),
                "dp": profile.get("data_parallel_size"),
                "tp": profile.get("tensor_parallel_size"),
                "nodes": profile.get("nodes"),
                "eligible": not blockers,
                "eligibility": eligibility,
                "requires_live_preflight": eligibility == "experimental_eligible",
                "blockers": blockers,
                "risk_notes": [str(item) for item in profile.get("risk_notes", [])],
                "validation": validation,
                "baseline_definition": profile.get("baseline_definition"),
            }
        )
    eligible = [item for item in candidates if item["eligible"]]
    production = [
        item for item in eligible if item["eligibility"] == "production_eligible"
    ]
    experimental = [
        item for item in eligible if item["eligibility"] == "experimental_eligible"
    ]
    default = str(document.get("default_profile", ""))
    selected = next((item for item in production if item["profile"] == default), None)
    if selected is None and len(eligible) == 1:
        selected = eligible[0]
    return {
        "schema_version": "vllmtkb-topology-plan/v1",
        "stage": "topology_outer_session",
        "decision_owner": "agent",
        "controller_role": "hard-filter-and-freeze",
        "requires_new_session_per_profile": True,
        "model_contract": model_contract,
        "resources": {
            "available_nodes": available_nodes,
            "npu_per_node": npu_per_node,
        },
        "candidates": candidates,
        "eligible_profiles": [item["profile"] for item in eligible],
        "production_eligible_profiles": [item["profile"] for item in production],
        "experimental_eligible_profiles": [item["profile"] for item in experimental],
        "recommended_profile": selected["profile"] if selected else None,
        "automatic_selection_reason": (
            "only one topology remains after hard filtering"
            if len(eligible) == 1
            else (
                "recommended_profile is the production incumbent; the Agent must "
                "schedule experimental preflight and compare topology-keyed Session metrics"
            )
        ),
    }


def select_measured_topology(
    plan: dict[str, Any],
    measurements: list[dict[str, Any]],
    *,
    agent_selected_profile: str | None = None,
) -> dict[str, Any]:
    """Validate an Agent selection; never replace it with a Controller argmax."""
    eligible = set(plan.get("eligible_profiles", []))
    valid = []
    for item in measurements:
        profile = str(item.get("profile", ""))
        score = item.get("output_token_throughput")
        if (
            profile in eligible
            and item.get("gate_passed") is True
            and isinstance(score, (int, float))
            and not isinstance(score, bool)
            and score > 0
        ):
            valid.append(dict(item))
    if not valid:
        return {
            "selected_profile": plan.get("recommended_profile"),
            "selection_basis": "safe_unmeasured_recommendation",
            "valid_measurements": [],
        }
    measured_profiles = {str(item["profile"]) for item in valid}
    if agent_selected_profile is None:
        if len(measured_profiles) != 1:
            return {
                "selected_profile": None,
                "selection_basis": "agent_selection_required",
                "valid_measurements": valid,
            }
        agent_selected_profile = next(iter(measured_profiles))
    if agent_selected_profile not in measured_profiles:
        raise ValueError(
            "Agent-selected topology must have a gate-passed comparable measurement: "
            f"selected={agent_selected_profile!r}, measured={sorted(measured_profiles)}"
        )
    selected = next(
        item for item in reversed(valid) if item["profile"] == agent_selected_profile
    )
    return {
        "selected_profile": selected["profile"],
        "selection_basis": "agent_selected_controller_validated",
        "valid_measurements": valid,
        "selected_measurement": selected,
    }


def validate_topology_baseline(
    profile_name: str,
    profile: dict[str, Any],
    baseline_document: dict[str, Any],
) -> dict[str, Any]:
    """Prove that a topology-specific baseline targets the frozen rank geometry."""
    invariants = baseline_document.get("deployment_invariants", {})
    topology = invariants.get("topology", {}) if isinstance(invariants, dict) else {}
    expected = {
        "pods": profile.get("nodes"),
        "npu_per_pod": profile.get("npu_per_node"),
        "data_parallel_size": profile.get("data_parallel_size"),
        "data_parallel_size_local": profile.get("data_parallel_size_local"),
        "tensor_parallel_size": profile.get("tensor_parallel_size"),
    }
    mismatches = {
        field: {"expected": value, "actual": topology.get(field)}
        for field, value in expected.items()
        if topology.get(field) != value
    }
    target = baseline_document.get("target_identity", {})
    declared_profile = target.get("topology_profile") if isinstance(target, dict) else None
    if declared_profile and declared_profile != profile_name:
        mismatches["topology_profile"] = {
            "expected": profile_name,
            "actual": declared_profile,
        }
    reference = baseline_document.get("reference_parameters")
    if not isinstance(reference, dict) or not reference:
        mismatches["reference_parameters"] = {
            "expected": "non-empty mapping",
            "actual": type(reference).__name__,
        }
    if mismatches:
        raise ValueError(
            f"Topology baseline {profile_name!r} does not match its frozen geometry: "
            f"{mismatches}"
        )
    return {"profile": profile_name, "passed": True, "geometry": expected}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a safe DP/TP topology plan")
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path(__file__).with_name("topology_profiles.yaml"),
    )
    parser.add_argument("--model-contract", default="glm-5.2-w8a8")
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--npu-per-node", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = build_plan(
        load_document(args.profiles),
        model_contract=args.model_contract,
        available_nodes=args.nodes,
        npu_per_node=args.npu_per_node,
    )
    rendered = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if plan["recommended_profile"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
