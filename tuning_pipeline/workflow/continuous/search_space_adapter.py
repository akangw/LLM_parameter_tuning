from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflow.search_space_compiler.compiler import SearchSpaceCompiler


def _latest_history(archive_root: Path) -> Path | None:
    candidates = list(
        archive_root.glob("glm52_continuous_*/round_*/06_agent_analysis/history_input.json")
    )
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _latest_previous_selection(archive_root: Path) -> Path | None:
    candidates = list(
        archive_root.glob(
            "glm52_continuous_*/00_search_space/search_space.compiled.yaml"
        )
    )
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def resolve_search_limits(
    raw_config: dict[str, Any],
    *,
    project_root: Path,
    archive_root: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Resolve a frozen per-Session configuration without mutating the input."""
    config = copy.deepcopy(raw_config)
    if config.get("search_limits_resolved"):
        return config, None
    mode = str(config.get("search_limits_mode", "manual"))
    if mode == "manual":
        config["manual_search_limits"] = copy.deepcopy(config["search_limits"])
        config["search_limits_resolved"] = True
        config["resolved_search_space"] = {
            "mode": "manual",
            "source": "config.search_limits",
        }
        return config, None
    if mode != "automated":
        raise ValueError(f"Unsupported search_limits_mode={mode!r}")

    settings = config.get("automated_search_limits", {})
    scenario_path = project_root / settings.get(
        "scenario", "workflow/search_space_compiler/scenario.glm52-a3-aligned-l1.yaml"
    )
    registry_path = project_root / settings.get(
        "registry", "workflow/search_space_compiler/registry.yaml"
    )
    policy_path = project_root / settings.get(
        "policy", "workflow/search_space_compiler/policy.yaml"
    )
    knowledge_dir = project_root / settings.get(
        "knowledge_dir", "tag_params/output/params"
    )
    history_path = None
    history_mode = str(settings.get("history_source", "latest_completed_session"))
    if history_mode == "latest_completed_session":
        history_path = _latest_history(archive_root)
    elif history_mode == "none":
        history_path = None
    elif history_mode == "explicit":
        explicit = settings.get("history_path")
        if not explicit:
            raise ValueError("history_path is required for history_source=explicit")
        history_path = project_root / str(explicit)
    else:
        raise ValueError(f"Unsupported history_source={history_mode!r}")

    previous_selection = _latest_previous_selection(archive_root)
    compiler = SearchSpaceCompiler(
        knowledge_dir=knowledge_dir,
        scenario_path=scenario_path,
        registry_path=registry_path,
        policy_path=policy_path,
        history_path=history_path,
        previous_selection_path=previous_selection,
    )
    result = compiler.compile()
    approved = {
        str(name) for name in settings.get("approved_planned_parameters", [])
    }
    unapproved = [
        item["canonical_name"]
        for item in result["active_parameters"]
        if item["integration_status"] != "existing"
        and item["canonical_name"] not in approved
    ]
    if unapproved:
        raise ValueError(
            "Automated search selected planned parameters without explicit "
            f"integration approval: {sorted(unapproved)}"
        )

    manual_limits = copy.deepcopy(config["search_limits"])
    effective_limits = copy.deepcopy(result["active_search_limits"])
    baseline = copy.deepcopy(config["baseline"])
    compiler_baseline = compiler.scenario.get("baseline", {})
    for name, values in effective_limits.items():
        if name not in baseline:
            preferred = compiler_baseline.get(name)
            baseline[name] = preferred if preferred in values else values[0]
    # Runtime-contract parameters and rotated-out legacy dimensions remain
    # present without consuming an active tuning slot. A derived parameter
    # keeps its manual values when one of its drivers is active, so the Agent
    # can maintain coupled invariants (for example, clearing an explicit
    # cudagraph capture list when changing its maximum).
    derived_runtime_parameters: list[str] = []
    derived_rules = config.get("change_policy", {}).get("derived_parameters", {})
    active_names = set(result["active_search_limits"])
    for name, values in manual_limits.items():
        if name not in effective_limits:
            if name not in baseline:
                raise ValueError(f"Manual parameter {name} has no baseline")
            rule = derived_rules.get(name, {})
            drivers = rule.get("drivers", []) if isinstance(rule, dict) else []
            if any(driver in active_names for driver in drivers):
                effective_limits[name] = [baseline[name]] + [
                    value for value in values if value != baseline[name]
                ]
                derived_runtime_parameters.append(name)
            else:
                effective_limits[name] = [baseline[name]]

    if set(baseline) != set(effective_limits):
        missing = sorted(set(effective_limits) - set(baseline))
        extra = sorted(set(baseline) - set(effective_limits))
        raise ValueError(
            f"Resolved automated baseline mismatch; missing={missing}, extra={extra}"
        )
    # A source-default B0 may resolve to a value outside the curated Agent
    # proposal grid (for example max_num_seqs=256 while the useful tuning grid
    # starts at 8..64). Preserve that measured starting point as the first
    # whitelisted value without pretending the compiler proposed it.
    source_default_anchors: list[str] = []
    for name, value in baseline.items():
        if value not in effective_limits[name]:
            effective_limits[name] = [value, *effective_limits[name]]
            source_default_anchors.append(name)

    result["integration"]["connected_to_mainflow"] = True
    result["integration"]["approval_source"] = (
        "config.automated_search_limits.approved_planned_parameters"
    )
    result["integration"]["approved_planned_parameters"] = sorted(approved)
    result["integration"]["effective_candidate_parameters"] = list(effective_limits)
    config["manual_search_limits"] = manual_limits
    config["search_limits"] = effective_limits
    config["baseline"] = baseline
    config["search_limits_resolved"] = True
    config["resolved_search_space"] = {
        "mode": "automated",
        "history": str(history_path) if history_path else None,
        "previous_selection": (
            str(previous_selection) if previous_selection else None
        ),
        "active_tunable_parameters": list(result["active_search_limits"]),
        "derived_runtime_parameters": derived_runtime_parameters,
        "fixed_runtime_parameters": [
            name
            for name in effective_limits
            if name not in result["active_search_limits"]
            and name not in derived_runtime_parameters
        ],
        "source_default_anchor_parameters": source_default_anchors,
        "rotation_swaps": result["rotation_audit"]["swaps"],
    }
    return config, result


def write_session_search_space(
    session_dir: Path,
    *,
    result: dict[str, Any] | None,
    config: dict[str, Any],
) -> None:
    output = session_dir / "00_search_space"
    output.mkdir(parents=True, exist_ok=False)
    (output / "manual_search_limits.yaml").write_text(
        yaml.safe_dump(
            {"search_limits": config["manual_search_limits"]},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    if result is None:
        (output / "selection.yaml").write_text(
            yaml.safe_dump(
                {
                    "mode": "manual",
                    "effective_search_limits": config["search_limits"],
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return
    (output / "search_space.compiled.yaml").write_text(
        yaml.safe_dump(result, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (output / "rotation_report.yaml").write_text(
        yaml.safe_dump(
            {
                "rotation_audit": result["rotation_audit"],
                "history_analysis": result["history_analysis"],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
