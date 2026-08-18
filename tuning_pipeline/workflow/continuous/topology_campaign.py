#!/usr/bin/env python3
"""Outer Agent-owned DP/TP Campaign over topology-frozen inner Sessions.

The Campaign owns scheduling and budget accounting, never candidate parameters.
Each topology receives an isolated Controller runtime root, state, history,
Active/Reserve selection and best anchor. Only comparable Fast-C32 metrics cross
the boundary. No task is submitted by ``--check-only``.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

if str(Path(__file__).resolve().parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from .agent_provider import (
        resolve_agent_profile,
        run_structured_agent,
        validate_agent_credentials,
    )
    from .continuous_tuning import (
        KB_ROOT,
        load_config,
        resolve_initial_baseline_definition,
    )
    from .runtime_profile import (
        apply_topology_baseline_binding,
        resolve_runtime_profile,
        validate_runtime_selections,
    )
    from .topology_advisor import (
        build_plan,
        load_document,
        validate_topology_baseline,
    )
    from .topology_profile import resolve_topology_profile
except ImportError:  # Direct script execution.
    from agent_provider import (
        resolve_agent_profile,
        run_structured_agent,
        validate_agent_credentials,
    )
    from continuous_tuning import (
        KB_ROOT,
        load_config,
        resolve_initial_baseline_definition,
    )
    from runtime_profile import (
        apply_topology_baseline_binding,
        resolve_runtime_profile,
        validate_runtime_selections,
    )
    from topology_advisor import (
        build_plan,
        load_document,
        validate_topology_baseline,
    )
    from topology_profile import resolve_topology_profile


HERE = Path(__file__).resolve().parent
CONTROLLER = HERE / "continuous_tuning.py"
DECISION_SCHEMA = HERE / "topology_campaign_decision.schema.json"
TERMINAL_INNER = {"completed_by_agent", "tuning_complete"}
PAUSED_INNER = {"budget_paused", "topology_feasibility_passed"}
FAILED_FEASIBILITY = {"topology_feasibility_failed"}
ACTIVE_ARM_STOP: Path | None = None
CAMPAIGN_STOP_REQUESTED = False


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.next-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(KB_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _campaign_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("topology_campaign", {})
    if not isinstance(settings, dict):
        raise ValueError("topology_campaign must be a mapping")
    return settings


def campaign_enabled(config_path: Path) -> bool:
    return _campaign_settings(load_config(config_path)).get("enabled") is True


def resolve_campaign_plan(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = load_config(config_path)
    runtime_config, identity = resolve_runtime_profile(raw, KB_ROOT)
    runtime_model = identity.get("model_contract", {})
    contract = f"{runtime_model.get('variant')}-{runtime_model.get('weight_format')}"
    topology_setting = runtime_config.get("topology", {})
    profiles_value = (
        topology_setting.get("profiles_file")
        if isinstance(topology_setting, dict)
        else None
    ) or "workflow/continuous/topology_profiles.yaml"
    profiles_path = Path(str(profiles_value))
    if not profiles_path.is_absolute():
        profiles_path = KB_ROOT / profiles_path
    document = load_document(profiles_path)
    defaults = document.get("selection", {})
    plan = build_plan(
        document,
        model_contract=contract,
        available_nodes=int(defaults.get("available_nodes", 2)),
        npu_per_node=int(defaults.get("npu_per_node", 16)),
    )
    requested = [
        str(item)
        for item in _campaign_settings(runtime_config).get(
            "candidate_profiles", plan["eligible_profiles"]
        )
    ]
    if not requested:
        raise ValueError("topology_campaign.candidate_profiles must not be empty")
    known = {str(item["profile"]) for item in plan["candidates"]}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f"Unknown topology Campaign profiles: {unknown}")
    plan["requested_profiles"] = requested
    plan["campaign_eligible_profiles"] = [
        name for name in requested if name in plan["eligible_profiles"]
    ]
    if not plan["campaign_eligible_profiles"]:
        raise ValueError("No requested topology survives Controller hard filters")
    return runtime_config, plan


def validate_arm_configuration(
    base_config: dict[str, Any], profile_name: str
) -> dict[str, Any]:
    """Resolve the same frozen topology/baseline pair the inner Controller uses."""
    configured = copy.deepcopy(base_config)
    configured.setdefault("topology", {})["profile"] = profile_name
    configured = apply_topology_baseline_binding(configured)
    configured = resolve_initial_baseline_definition(configured, KB_ROOT)
    validate_runtime_selections(configured)
    configured, profile = resolve_topology_profile(configured, KB_ROOT)
    definition_value = configured.get("initial_baseline", {}).get("definition")
    if not definition_value:
        raise ValueError(f"Topology {profile_name!r} lacks an initial baseline")
    definition_path = Path(str(definition_value))
    if not definition_path.is_absolute():
        definition_path = KB_ROOT / definition_path
    baseline = _load_yaml(definition_path)
    validation = validate_topology_baseline(profile_name, profile, baseline)
    return {
        "profile": profile_name,
        "topology": {
            name: profile.get(name)
            for name in (
                "executor",
                "nodes",
                "npu_per_node",
                "data_parallel_size",
                "data_parallel_size_local",
                "tensor_parallel_size",
                "worker_replicas",
                "worker_data_parallel_start_rank",
                "validation",
            )
        },
        "baseline": {
            "label": configured["initial_baseline"]["label"],
            "definition": _portable(definition_path),
            "baseline_id": baseline.get("baseline_id"),
        },
        "preflight": validation,
    }


def initialize_state(
    config_path: Path, campaign_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config, plan = resolve_campaign_plan(config_path)
    settings = _campaign_settings(config)
    candidates = {str(item["profile"]): item for item in plan["candidates"]}
    arms: dict[str, Any] = {}
    for profile in plan["requested_profiles"]:
        item = candidates[profile]
        arm: dict[str, Any] = {
            "profile": profile,
            "eligibility": item["eligibility"],
            "requires_live_preflight": item["requires_live_preflight"],
            "runtime_root": str((campaign_root / "sessions" / profile).resolve()),
            "status": "pending" if item["eligible"] else "blocked",
            "blockers": item["blockers"],
            "measurements": [],
        }
        if item["eligible"]:
            arm["static_preflight"] = validate_arm_configuration(config, profile)
        arms[profile] = arm
    state = {
        "schema_version": "vllmtkb-topology-campaign/v1",
        "campaign_id": "topology_campaign_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "status": "initialized",
        "stage": "feasibility",
        "config": str(config_path.resolve()),
        "campaign_root": str(campaign_root.resolve()),
        "plan": plan,
        "settings": {
            "screen_measurements_per_topology": int(
                settings.get("screen_measurements_per_topology", 3)
            ),
            "competitive_slice_rounds": int(
                settings.get("competitive_slice_rounds", 4)
            ),
            "minimum_challenger_rounds_per_slice": int(
                settings.get("minimum_challenger_rounds_per_slice", 1)
            ),
            "final_challenger_verification_rounds": int(
                settings.get("final_challenger_verification_rounds", 2)
            ),
            "maximum_competitive_cycles": int(
                settings.get("maximum_competitive_cycles", 8)
            ),
            "maximum_candidate_index_per_topology": int(
                settings.get("maximum_candidate_index_per_topology", 24)
            ),
        },
        "arms": arms,
        "competitive_cycles": 0,
        "final_verification": None,
        "decisions": [],
        "created_at": _now(),
        "updated_at": _now(),
    }
    validate_campaign_settings(state["settings"])
    return config, state


def validate_campaign_settings(settings: dict[str, Any]) -> None:
    limits = {
        "screen_measurements_per_topology": (2, 8),
        "competitive_slice_rounds": (2, 8),
        "minimum_challenger_rounds_per_slice": (1, 4),
        "final_challenger_verification_rounds": (1, 4),
        "maximum_competitive_cycles": (1, 32),
        "maximum_candidate_index_per_topology": (4, 128),
    }
    for name, (minimum, maximum) in limits.items():
        value = int(settings[name])
        if not minimum <= value <= maximum:
            raise ValueError(f"topology_campaign.{name} must be {minimum}..{maximum}")
    if settings["minimum_challenger_rounds_per_slice"] >= settings[
        "competitive_slice_rounds"
    ]:
        raise ValueError("competitive slice must retain more incumbent than challenger budget")


def collect_arm_measurements(arm: dict[str, Any]) -> dict[str, Any]:
    runtime_root = Path(str(arm["runtime_root"]))
    state_path = runtime_root / "state.json"
    inner_state = _load_json(state_path) if state_path.is_file() else {}
    measurements: list[dict[str, Any]] = []
    session_value = inner_state.get("session_dir")
    session_dir = Path(str(session_value)) if session_value else None
    if session_dir and session_dir.is_dir():
        for path in sorted(session_dir.glob("round_*/05_results/metrics.json")):
            try:
                document = _load_json(path)
                metrics = document.get("metrics", {})
                if not isinstance(metrics, dict):
                    continue
                round_match = re.match(r"round_(\d+)_", path.parents[1].name)
                throughput = metrics.get("output_token_throughput")
                successful = metrics.get("successful_requests", 0)
                failed = metrics.get("failed_requests", 0)
                gate = (
                    isinstance(throughput, (int, float))
                    and not isinstance(throughput, bool)
                    and throughput > 0
                    and isinstance(successful, (int, float))
                    and successful > 0
                    and failed == 0
                )
                measurements.append(
                    {
                        "round": path.parents[1].name,
                        "round_index": int(round_match.group(1)) if round_match else None,
                        "gate_passed": gate,
                        "output_token_throughput": throughput,
                        "ttft_p50_ms": metrics.get("ttft_p50_ms", metrics.get("mean_ttft")),
                        "ttft_p90_ms": metrics.get("ttft_p90_ms"),
                        "tpot_p50_ms": metrics.get("tpot_p50_ms", metrics.get("mean_tpot")),
                        "tpot_p90_ms": metrics.get("tpot_p90_ms"),
                        "benchmark_wall_time_seconds": metrics.get(
                            "benchmark_wall_time_seconds"
                        ),
                    }
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
    valid = [item for item in measurements if item["gate_passed"]]
    best = (
        max(valid, key=lambda item: float(item["output_token_throughput"]))
        if valid
        else None
    )
    arm["measurements"] = measurements
    arm["measurement_count"] = len(measurements)
    arm["valid_measurement_count"] = len(valid)
    arm["best_measurement"] = best
    arm["inner_state"] = {
        key: inner_state.get(key)
        for key in (
            "session_id",
            "session_dir",
            "status",
            "candidate_index",
            "round_index",
            "topology_profile",
        )
    }
    if inner_state.get("status") in FAILED_FEASIBILITY:
        arm["status"] = "infeasible"
    elif inner_state.get("status") in TERMINAL_INNER:
        arm["status"] = "inner_complete"
    elif inner_state.get("status") in PAUSED_INNER:
        arm["status"] = "ready"
    elif inner_state:
        arm["status"] = str(inner_state.get("status", "unknown"))
    return arm


def refresh_state(state: dict[str, Any]) -> dict[str, Any]:
    for arm in state["arms"].values():
        if arm["status"] != "blocked":
            collect_arm_measurements(arm)
    state["updated_at"] = _now()
    return state


def _eligible_live_profiles(state: dict[str, Any]) -> list[str]:
    return [
        name
        for name in state["plan"]["campaign_eligible_profiles"]
        if state["arms"][name]["status"] not in {"blocked", "infeasible"}
    ]


def validate_legacy_runtime_is_idle(campaign_root: Path) -> None:
    """Never let a new Campaign silently orphan the prior single-Session chain."""
    legacy_state = campaign_root.parent / "state.json"
    if not legacy_state.is_file():
        return
    state = _load_json(legacy_state)
    status = str(state.get("status", ""))
    terminal = {"completed_by_agent", "tuning_complete", "dry_run_complete"}
    if status not in terminal or isinstance(state.get("pending_submission"), dict):
        raise RuntimeError(
            "Legacy single-Session runtime is not terminal; archive or finish it "
            f"before starting topology Campaign: status={status!r}"
        )


def build_agent_evidence(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "campaign_id": state["campaign_id"],
        "stage": state["stage"],
        "policy": {
            "topology_is_frozen_inside_each_session": True,
            "history_and_best_anchor_are_topology_keyed": True,
            "controller_does_not_argmax_or_choose_topology": True,
            "experimental_topology_requires_first_round_preflight": True,
            "screening_is_equal_budget": True,
            "competitive_budget_is_agent_allocated_with_challenger_floor": True,
        },
        "settings": state["settings"],
        "eligible_profiles": _eligible_live_profiles(state),
        "arms": {
            name: {
                "eligibility": arm["eligibility"],
                "status": arm["status"],
                "topology": arm.get("static_preflight", {}).get("topology"),
                "baseline": arm.get("static_preflight", {}).get("baseline"),
                "measurement_count": arm.get("measurement_count", 0),
                "valid_measurement_count": arm.get("valid_measurement_count", 0),
                "best_measurement": arm.get("best_measurement"),
                "latest_measurement": (
                    arm.get("measurements", [])[-1]
                    if arm.get("measurements")
                    else None
                ),
                "inner_status": arm.get("inner_state", {}).get("status"),
                "candidate_index": arm.get("inner_state", {}).get("candidate_index"),
            }
            for name, arm in state["arms"].items()
        },
        "competitive_cycles": state["competitive_cycles"],
        "final_verification": state["final_verification"],
        "prior_decisions": state["decisions"][-4:],
    }


def _agent_prompt(state: dict[str, Any]) -> str:
    stage = state["stage"]
    rules = {
        "feasibility": (
            "Allocate exactly one baseline round to every pending eligible topology. "
            "Your allocation array order decides execution order."
        ),
        "equal_budget_screening": (
            "Bring every live topology to the same configured screening measurement "
            "count. Allocate each exact deficit; your array order decides execution order."
        ),
        "competitive": (
            "Choose the incumbent and challenger from measured topology Sessions. "
            "For allocate_budget, allocate exactly the competitive slice total and "
            "give at least the configured floor to a non-incumbent challenger when two "
            "live topologies remain. You may request final verification when evidence "
            "is mature, but may not complete before it runs."
        ),
        "post_final_verification": (
            "Final challenger evidence now exists. Complete with the Agent-selected "
            "winner, allocate another bounded competitive slice if uncertainty remains, "
            "or pause with a concrete external blocker."
        ),
    }[stage]
    return f"""You are the topology-level optimization Agent for vLLM-Ascend.
DP/TP topology has very large impact, so make the decision yourself from the full
evidence. The Controller only enforces physical feasibility, frozen Session identity,
comparable Benchmark gates, equal-screening/fairness budgets, and hard caps. It will
not substitute a throughput argmax for your judgment.

Stage rule: {rules}

Prefer fast convergence, but accept high-risk, technically plausible exploration.
Do not allocate obviously impossible, failed-preflight, repeated, or evidence-free
directions. TTFT and TPOT are evidence and guardrails; output-token throughput is the
primary objective. Return one schema-valid decision.

Evidence:
{json.dumps(build_agent_evidence(state), ensure_ascii=False, indent=2)}
"""


def run_campaign_agent(
    config: dict[str, Any], state: dict[str, Any], campaign_root: Path
) -> dict[str, Any]:
    agent = resolve_agent_profile(
        config.get("agent", {}), legacy_command=str(config.get("codex_command", "auto"))
    )
    validate_agent_credentials(agent)
    decision_dir = campaign_root / "decisions"
    decision_dir.mkdir(parents=True, exist_ok=True)
    sequence = len(state["decisions"]) + 1
    output = decision_dir / f"decision_{sequence:03d}.json"
    prompt_path = decision_dir / f"prompt_{sequence:03d}.txt"
    prompt = _agent_prompt(state)
    prompt_path.write_text(prompt, encoding="utf-8")
    failures: list[str] = []
    retries = int(agent.get("settings", {}).get("max_protocol_retries", 2)) + 1
    for attempt in range(1, retries + 1):
        retry_prompt = prompt
        if failures:
            retry_prompt += (
                "\nThe prior provider output failed. Return a fresh complete decision. "
                "Failure: " + failures[-1][-2000:]
            )
        result = run_structured_agent(
            agent,
            prompt=retry_prompt,
            schema_path=DECISION_SCHEMA,
            output_path=output,
            cwd=KB_ROOT,
            allowed_dir=campaign_root,
            timeout=1800,
        )
        if result.returncode == 0 and output.is_file():
            try:
                decision = _load_json(output)
                validate_agent_decision(state, decision)
                return decision
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                failures.append(
                    f"attempt {attempt}/{retries}: Controller decision validation: {exc}"
                )
                continue
        failures.append(
            f"attempt {attempt}/{retries}: returncode={result.returncode}; "
            f"stderr={result.stderr[-4000:]}"
        )
    raise RuntimeError("Topology Campaign Agent protocol failed: " + " | ".join(failures))


def validate_agent_decision(state: dict[str, Any], decision: dict[str, Any]) -> None:
    live = _eligible_live_profiles(state)
    live_set = set(live)
    action = decision.get("action")
    incumbent = decision.get("incumbent_profile")
    challenger = decision.get("challenger_profile")
    allocations = decision.get("allocations", [])
    if incumbent is not None and incumbent not in live_set:
        raise ValueError("Agent incumbent is not a live topology")
    if challenger is not None and challenger not in live_set:
        raise ValueError("Agent challenger is not a live topology")
    if incumbent is not None and challenger == incumbent:
        raise ValueError("Agent incumbent and challenger must differ")
    names = [str(item.get("profile", "")) for item in allocations]
    if len(names) != len(set(names)) or any(name not in live_set for name in names):
        raise ValueError("Agent allocations must be unique live topology profiles")
    completed_allocations = [
        name for name in names if state["arms"][name]["status"] == "inner_complete"
    ]
    if completed_allocations:
        raise ValueError(
            f"Agent cannot allocate completed inner Sessions: {completed_allocations}"
        )
    stage = state["stage"]
    counts = {
        name: int(state["arms"][name].get("measurement_count", 0)) for name in live
    }
    allocated = {
        str(item["profile"]): int(item["additional_candidate_rounds"])
        for item in allocations
    }
    if action == "pause_for_human":
        if allocations:
            raise ValueError("pause_for_human must not allocate NPU rounds")
        return
    if stage == "feasibility":
        pending = {name for name in live if counts[name] == 0}
        if action != "allocate_budget" or set(allocated) != pending or any(
            value != 1 for value in allocated.values()
        ):
            raise ValueError("Feasibility must allocate exactly one round per pending arm")
        return
    if stage == "equal_budget_screening":
        target = int(state["settings"]["screen_measurements_per_topology"])
        deficits = {
            name: target - counts[name]
            for name in live
            if counts[name] < target
            and state["arms"][name]["status"] != "inner_complete"
        }
        if action != "allocate_budget" or allocated != deficits:
            raise ValueError(f"Equal screening allocations must equal deficits={deficits}")
        return
    measured = {name for name in live if counts[name] > 0}
    if incumbent not in measured:
        raise ValueError("Competitive decisions require a measured incumbent")
    if action == "complete_campaign":
        if stage != "post_final_verification" and len(live) > 1:
            raise ValueError("Campaign cannot complete before final challenger verification")
        if allocations:
            raise ValueError("complete_campaign must not allocate rounds")
        return
    if action == "start_final_verification":
        if len(live) > 1 and challenger not in measured:
            raise ValueError("Final challenger must have comparable measured evidence")
        expected = int(state["settings"]["final_challenger_verification_rounds"])
        if len(live) > 1 and allocated != {challenger: expected}:
            raise ValueError("Final verification must allocate only the configured challenger budget")
        if len(live) == 1 and allocations:
            raise ValueError("A single live topology has no challenger to verify")
        return
    if action != "allocate_budget":
        raise ValueError(f"Unsupported Campaign action: {action!r}")
    total = sum(allocated.values())
    expected_total = int(state["settings"]["competitive_slice_rounds"])
    if total != expected_total:
        raise ValueError(f"Competitive allocation total must equal {expected_total}")
    if len(live) > 1:
        floor = int(state["settings"]["minimum_challenger_rounds_per_slice"])
        non_incumbent = sum(value for name, value in allocated.items() if name != incumbent)
        if non_incumbent < floor:
            raise ValueError(f"Competitive challenger allocation must be at least {floor}")


def _arm_action(arm: dict[str, Any]) -> str:
    state_path = Path(str(arm["runtime_root"])) / "state.json"
    return "--resume" if state_path.is_file() else "--start"


def advance_arm(
    config_path: Path,
    state: dict[str, Any],
    profile: str,
    additional_rounds: int,
) -> None:
    global ACTIVE_ARM_STOP
    arm = state["arms"][profile]
    for other_name, other in state["arms"].items():
        if other_name == profile or other.get("status") == "blocked":
            continue
        other_state_path = Path(str(other["runtime_root"])) / "state.json"
        if not other_state_path.is_file():
            continue
        other_state = _load_json(other_state_path)
        other_status = str(other_state.get("status", ""))
        safely_idle = other_status in TERMINAL_INNER | PAUSED_INNER | FAILED_FEASIBILITY
        if not safely_idle and (
            other_state.get("active_task_id") or other_state.get("active_run_id")
        ):
            raise RuntimeError(
                f"Topology {other_name} still owns an active task/run; refusing "
                f"overlapping allocation to {profile}"
            )
    collect_arm_measurements(arm)
    if arm["status"] in {"blocked", "infeasible", "inner_complete"}:
        raise RuntimeError(f"Cannot allocate topology {profile}: status={arm['status']}")
    current_count = int(arm.get("measurement_count", 0))
    maximum_index = int(state["settings"]["maximum_candidate_index_per_topology"])
    inner = arm.get("inner_state", {})
    current_candidate_index = inner.get("candidate_index")
    current_round_index = inner.get("round_index")
    current_round_is_measured = any(
        item.get("round_index") == current_round_index
        for item in arm.get("measurements", [])
    )
    target_index = (
        int(current_candidate_index)
        + additional_rounds
        - (0 if current_round_is_measured else 1)
        if current_candidate_index is not None
        else additional_rounds - 1
    )
    if target_index > maximum_index:
        raise RuntimeError(
            f"Topology {profile} would exceed candidate-index cap {maximum_index}"
        )
    runtime_root = Path(str(arm["runtime_root"]))
    runtime_root.mkdir(parents=True, exist_ok=True)
    action = _arm_action(arm)
    command = [
        sys.executable,
        str(CONTROLLER),
        "--config",
        str(config_path),
        "--runtime-root",
        str(runtime_root),
        action,
        "--pause-after-candidate-index",
        str(target_index),
    ]
    if action == "--start":
        command.extend(["--topology-profile", profile])
    if current_count == 0 and arm["requires_live_preflight"]:
        command.append("--topology-feasibility-only")
    ACTIVE_ARM_STOP = runtime_root / "STOP_REQUESTED"
    completed = subprocess.run(command, cwd=str(KB_ROOT), check=False)
    ACTIVE_ARM_STOP = None
    collect_arm_measurements(arm)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Inner Controller failed for {profile} with exit code {completed.returncode}"
        )
    if int(arm.get("measurement_count", 0)) < current_count + additional_rounds:
        if arm["status"] not in {"infeasible", "inner_complete"}:
            raise RuntimeError(
                f"Topology {profile} did not consume its allocated measurement budget"
            )


def apply_decision(
    config_path: Path,
    state: dict[str, Any],
    decision: dict[str, Any],
    *,
    state_path: Path | None = None,
) -> None:
    pending = state.get("pending_decision")
    if isinstance(pending, dict):
        decision = copy.deepcopy(pending["decision"])
        next_allocation = int(pending.get("next_allocation_index", 0))
    else:
        record = {"at": _now(), "stage": state["stage"], **copy.deepcopy(decision)}
        state["decisions"].append(record)
        next_allocation = 0
        state["pending_decision"] = {
            "decision": copy.deepcopy(decision),
            "next_allocation_index": 0,
            "created_at": _now(),
        }
        if state_path is not None:
            _atomic_json(state_path, state)
    action = decision["action"]
    if action == "pause_for_human":
        state["status"] = "paused_for_human"
        state["pause_summary"] = decision["summary"]
        state.pop("pending_decision", None)
        return
    for index, allocation in enumerate(decision["allocations"]):
        if index < next_allocation:
            continue
        if CAMPAIGN_STOP_REQUESTED:
            state["status"] = "stop_requested"
            return
        advance_arm(
            config_path,
            state,
            str(allocation["profile"]),
            int(allocation["additional_candidate_rounds"]),
        )
        state["pending_decision"]["next_allocation_index"] = index + 1
        if state_path is not None:
            _atomic_json(state_path, state)
    refresh_state(state)
    if action == "complete_campaign":
        state["status"] = "completed_by_agent"
        state["selected_profile"] = decision["incumbent_profile"]
        state["completion_summary"] = decision["summary"]
    elif action == "start_final_verification":
        state["final_verification"] = {
            "incumbent_profile": decision["incumbent_profile"],
            "challenger_profile": decision["challenger_profile"],
            "completed_at": _now(),
        }
        state["stage"] = "post_final_verification"
    elif state["stage"] == "feasibility":
        state["stage"] = "equal_budget_screening"
    elif state["stage"] == "equal_budget_screening":
        state["stage"] = "competitive"
    elif state["stage"] in {"competitive", "post_final_verification"}:
        state["competitive_cycles"] = int(state["competitive_cycles"]) + 1
        if state["stage"] == "post_final_verification":
            state["final_verification"] = None
            state["stage"] = "competitive"
    state.pop("pending_decision", None)


def _signal_stop(_signum: int, _frame: object) -> None:
    global CAMPAIGN_STOP_REQUESTED
    CAMPAIGN_STOP_REQUESTED = True
    if ACTIVE_ARM_STOP is not None:
        ACTIVE_ARM_STOP.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_ARM_STOP.touch(exist_ok=True)


def run_campaign(config_path: Path, campaign_root: Path, mode: str) -> dict[str, Any]:
    validate_legacy_runtime_is_idle(campaign_root)
    state_path = campaign_root / "campaign_state.json"
    if mode == "new" and state_path.exists():
        raise RuntimeError("A topology Campaign state already exists; use resume or auto")
    if mode == "resume" and not state_path.exists():
        raise RuntimeError("No topology Campaign state exists to resume")
    if state_path.exists():
        state = _load_json(state_path)
        config = load_config(config_path)
        config, _ = resolve_runtime_profile(config, KB_ROOT)
    else:
        config, state = initialize_state(config_path, campaign_root)
        _atomic_json(state_path, state)
    if state["status"] == "completed_by_agent":
        return state
    signal.signal(signal.SIGTERM, _signal_stop)
    signal.signal(signal.SIGINT, _signal_stop)
    state["status"] = "running"
    while not CAMPAIGN_STOP_REQUESTED:
        refresh_state(state)
        live = _eligible_live_profiles(state)
        if not live:
            state["status"] = "paused_for_human"
            state["pause_summary"] = "Every topology is blocked or failed feasibility."
            break
        if state["stage"] == "feasibility" and all(
            int(state["arms"][name].get("measurement_count", 0)) > 0 for name in live
        ):
            state["stage"] = "equal_budget_screening"
        if state["stage"] == "equal_budget_screening":
            target = int(state["settings"]["screen_measurements_per_topology"])
            if all(
                int(state["arms"][name].get("measurement_count", 0)) >= target
                or state["arms"][name]["status"] == "inner_complete"
                for name in live
            ):
                state["stage"] = "competitive"
        if (
            state["stage"] == "competitive"
            and int(state["competitive_cycles"])
            >= int(state["settings"]["maximum_competitive_cycles"])
        ):
            # The Agent still chooses the challenger/winner. The prompt exposes
            # the hard cap and validation accepts only final verification/pause.
            state["budget_cap_reached"] = True
        _atomic_json(state_path, state)
        pending = state.get("pending_decision")
        decision = (
            copy.deepcopy(pending["decision"])
            if isinstance(pending, dict)
            else run_campaign_agent(config, state, campaign_root)
        )
        if state.get("budget_cap_reached") and decision["action"] == "allocate_budget":
            raise ValueError(
                "Competitive cycle cap reached; Agent must request final verification or pause"
            )
        apply_decision(config_path, state, decision, state_path=state_path)
        _atomic_json(state_path, state)
        if state["status"] in {"completed_by_agent", "paused_for_human"}:
            break
    if CAMPAIGN_STOP_REQUESTED:
        state["status"] = "stopped_after_active_slice"
        state["updated_at"] = _now()
        _atomic_json(state_path, state)
    return state


def check_only(config_path: Path, campaign_root: Path) -> dict[str, Any]:
    _, state = initialize_state(config_path, campaign_root)
    return {
        "status": "ok",
        "mutates_remote_or_submits_tasks": False,
        "plan": state["plan"],
        "arms": {
            name: {
                "status": arm["status"],
                "eligibility": arm["eligibility"],
                "static_preflight": arm.get("static_preflight"),
                "blockers": arm["blockers"],
            }
            for name, arm in state["arms"].items()
        },
        "settings": state["settings"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the topology-first tuning Campaign")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--is-enabled", action="store_true")
    group.add_argument("--check-only", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--mode", choices=("auto", "new", "resume"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    campaign_root = args.campaign_root.expanduser().resolve()
    if args.is_enabled:
        return 0 if campaign_enabled(config_path) else 1
    if args.check_only:
        print(
            json.dumps(
                check_only(config_path, campaign_root), ensure_ascii=False, indent=2
            )
        )
        return 0
    state = run_campaign(config_path, campaign_root, args.mode)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if state.get("status") == "completed_by_agent" else 78


if __name__ == "__main__":
    raise SystemExit(main())
