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
from workflow.registry_builder.pipeline import AutomaticRegistryPipeline


def _selected_benchmark_identity(
    config: dict[str, Any], project_root: Path
) -> dict[str, Any]:
    settings = config.get("benchmark", {})
    if not isinstance(settings, dict):
        raise ValueError("benchmark configuration must be an object")
    profile_name = settings.get("profile")
    if profile_name:
        frozen = settings.get("resolved_profile")
        if isinstance(frozen, dict):
            profile = frozen
        else:
            profiles_path = _project_path(
                project_root,
                str(
                    settings.get(
                        "profiles_file", "workflow/continuous/benchmark_profiles.yaml"
                    )
                ),
            )
            document = yaml.safe_load(profiles_path.read_text(encoding="utf-8"))
            profile = document.get("profiles", {}).get(str(profile_name))
            if not isinstance(profile, dict):
                raise ValueError(f"Unknown Benchmark profile {profile_name!r}")
        mode = str(profile["mode"])
        definition_key = str(profile["definition_key"])
    else:
        mode = str(settings.get("mode", ""))
        definition_key = mode
    definition = settings.get(definition_key)
    if not mode or not isinstance(definition, dict):
        raise ValueError(
            f"Benchmark definition is missing for mode={mode!r}, key={definition_key!r}"
        )
    return {"mode": mode, "definition": copy.deepcopy(definition)}


def _manifest_matches_scenario(
    manifest: dict[str, Any], scenario_image: dict[str, Any]
) -> bool:
    return (
        manifest.get("target_image", {}).get("digest") == scenario_image.get("digest")
        and manifest.get("versions", {}).get("vllm", {}).get("commit")
        == scenario_image.get("vllm_commit")
        and manifest.get("versions", {}).get("vllm_ascend", {}).get("commit")
        == scenario_image.get("vllm_ascend_commit")
    )


def _latest_history(
    archive_root: Path,
    *,
    benchmark_identity: dict[str, Any],
    scenario_image: dict[str, Any],
    project_root: Path,
) -> Path | None:
    candidates = sorted(
        archive_root.glob("*/round_*/06_agent_analysis/history_input.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        session_dir = path.parents[2]
        config_path = session_dir / "session_config.yaml"
        manifest_path = session_dir / "image_version_manifest.yaml"
        try:
            frozen_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(frozen_config, dict) or not isinstance(manifest, dict):
                continue
            frozen_benchmark = _selected_benchmark_identity(
                frozen_config, project_root
            )
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
            continue
        if frozen_benchmark != benchmark_identity:
            continue
        if not _manifest_matches_scenario(manifest, scenario_image):
            continue
        return path
    return None


def _latest_previous_selection(
    archive_root: Path,
    profile_name: str,
    *,
    benchmark_identity: dict[str, Any],
    scenario_image: dict[str, Any],
    project_root: Path,
) -> Path | None:
    candidates = sorted(
        archive_root.glob("*/00_search_space/search_space.compiled.yaml"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        session_dir = path.parents[1]
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            frozen_config = yaml.safe_load(
                (session_dir / "session_config.yaml").read_text(encoding="utf-8")
            )
            manifest = yaml.safe_load(
                (session_dir / "image_version_manifest.yaml").read_text(
                    encoding="utf-8"
                )
            )
            if not isinstance(frozen_config, dict) or not isinstance(manifest, dict):
                continue
            frozen_benchmark = _selected_benchmark_identity(
                frozen_config, project_root
            )
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
            continue
        if (
            isinstance(value, dict)
            and value.get("integration", {}).get("search_space_profile") == profile_name
            and frozen_benchmark == benchmark_identity
            and _manifest_matches_scenario(manifest, scenario_image)
        ):
            return path
    return None


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _resolve_profile(
    config: dict[str, Any], project_root: Path
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    settings = config.get("search_space")
    if not isinstance(settings, dict):
        return None
    frozen = settings.get("resolved_profile")
    if isinstance(frozen, dict):
        name = str(settings["profile"])
        profile = copy.deepcopy(frozen)
    else:
        profiles_path = _project_path(
            project_root,
            str(settings.get("profiles_file", "workflow/search_space_profiles.yaml")),
        )
        document = yaml.safe_load(profiles_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(
                f"Search-space profiles must be an object: {profiles_path}"
            )
        name = str(settings.get("profile") or document.get("default_profile"))
        profiles = document.get("profiles", {})
        if name not in profiles:
            raise ValueError(
                f"Unknown search-space profile {name!r}; available={sorted(profiles)}"
            )
        profile = copy.deepcopy(profiles[name])
    if profile.get("status") != "integrated":
        raise ValueError(f"Search-space profile {name!r} is not integrated")
    mode = str(profile.get("mode", ""))
    if mode not in {"curated_registry", "automatic_registry"}:
        raise ValueError(f"Unsupported search-space profile mode={mode!r}")
    config.setdefault("search_space", {})["profile"] = name
    config["search_space"]["resolved_profile"] = copy.deepcopy(profile)
    return name, profile, settings


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
    profile_resolution = _resolve_profile(config, project_root)
    legacy_mode = str(config.get("search_limits_mode", "manual"))
    if profile_resolution is None and legacy_mode == "manual":
        config["manual_search_limits"] = copy.deepcopy(config["search_limits"])
        config["search_limits_resolved"] = True
        config["resolved_search_space"] = {
            "mode": "manual",
            "source": "config.search_limits",
        }
        return config, None
    if profile_resolution is None:
        if legacy_mode != "automated":
            raise ValueError(f"Unsupported search_limits_mode={legacy_mode!r}")
        profile_name = "legacy_curated_registry"
        profile = {
            "mode": "curated_registry",
            **dict(config.get("automated_search_limits", {})),
        }
        settings = config.get("automated_search_limits", {})
    else:
        profile_name, profile, settings = profile_resolution
    mode = str(profile["mode"])
    scenario_path = _project_path(
        project_root,
        str(
            profile.get(
                "scenario",
                "workflow/search_space_compiler/scenario.glm52-a3-aligned-l1.yaml",
            )
        ),
    )
    scenario_document = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    if not isinstance(scenario_document, dict):
        raise ValueError(f"Scenario must be an object: {scenario_path}")
    policy_path = _project_path(
        project_root,
        str(profile.get("policy", "workflow/search_space_compiler/policy.yaml")),
    )
    knowledge_dir = _project_path(
        project_root, str(profile.get("knowledge_dir", "tag_params/output/params"))
    )
    history_path = None
    history_mode = str(settings.get("history_source", "latest_completed_session"))
    if history_mode == "latest_completed_session":
        history_path = _latest_history(
            archive_root,
            benchmark_identity=_selected_benchmark_identity(config, project_root),
            scenario_image=dict(scenario_document.get("image", {})),
            project_root=project_root,
        )
    elif history_mode == "none":
        history_path = None
    elif history_mode == "explicit":
        explicit = settings.get("history_path")
        if not explicit:
            raise ValueError("history_path is required for history_source=explicit")
        history_path = _project_path(project_root, str(explicit))
    else:
        raise ValueError(f"Unsupported history_source={history_mode!r}")

    benchmark_identity = _selected_benchmark_identity(config, project_root)
    scenario_image = dict(scenario_document.get("image", {}))
    previous_selection = _latest_previous_selection(
        archive_root,
        profile_name,
        benchmark_identity=benchmark_identity,
        scenario_image=scenario_image,
        project_root=project_root,
    )
    automatic_registry: dict[str, Any] | None = None
    if mode == "curated_registry":
        registry_path = _project_path(
            project_root,
            str(
                profile.get("registry", "workflow/search_space_compiler/registry.yaml")
            ),
        )
        compiler = SearchSpaceCompiler(
            knowledge_dir=knowledge_dir,
            scenario_path=scenario_path,
            registry_path=registry_path,
            policy_path=policy_path,
            history_path=history_path,
            previous_selection_path=previous_selection,
            activation_override=dict(profile.get("activation", {})),
        )
        result = compiler.compile()
        compiler_scenario = compiler.scenario
    else:
        compatibility_policy_path = _project_path(
            project_root,
            str(
                profile.get(
                    "compatibility_policy",
                    "workflow/registry_builder/compatibility_policy.yaml",
                )
            ),
        )
        source_root = _project_path(
            project_root,
            str(profile.get("source_root", "../portrait_pipeline/sources")),
        )
        automatic_pipeline = AutomaticRegistryPipeline(
            knowledge_dir=knowledge_dir,
            scenario_path=scenario_path,
            policy_path=policy_path,
            compatibility_policy_path=compatibility_policy_path,
            source_root=source_root,
            activation_override=dict(profile.get("activation", {})),
        )
        automatic_registry, result = automatic_pipeline.compile()
        compiler_scenario = automatic_pipeline.scenario
        result["automatic_registry_snapshot"] = automatic_registry
        result["integration"]["compatibility_policy"] = str(compatibility_policy_path)
        result["integration"]["source_identity"] = automatic_pipeline.source_identity
    approved = {str(name) for name in settings.get("approved_planned_parameters", [])}
    unapproved = (
        []
        if mode == "automatic_registry"
        else [
            item["canonical_name"]
            for item in result["active_parameters"]
            if item["integration_status"] != "existing"
            and item["canonical_name"] not in approved
        ]
    )
    if unapproved:
        raise ValueError(
            "Automated search selected planned parameters without explicit "
            f"integration approval: {sorted(unapproved)}"
        )

    manual_limits = copy.deepcopy(config["search_limits"])
    effective_limits = copy.deepcopy(result["active_search_limits"])
    baseline = copy.deepcopy(config["baseline"])
    compiler_baseline = compiler_scenario.get("baseline", {})
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
    change_policy = config.setdefault("change_policy", {})
    derived_rules = change_policy.setdefault("derived_parameters", {})
    active_names = set(result["active_search_limits"])
    profile_derived_rules = profile.get("derived_parameters", {})
    if not isinstance(profile_derived_rules, dict):
        raise ValueError("Search-space profile derived_parameters must be an object")
    unknown_derived = sorted(set(profile_derived_rules) - set(manual_limits))
    if unknown_derived:
        raise ValueError(
            "Search-space profile derived parameters are missing from the runtime "
            f"schema: {unknown_derived}"
        )
    for name, rule in profile_derived_rules.items():
        if not isinstance(rule, dict) or not isinstance(rule.get("drivers"), list):
            raise ValueError(
                f"Search-space derived rule for {name!r} must define a drivers list"
            )
        unknown_drivers = sorted(set(map(str, rule["drivers"])) - active_names)
        if unknown_drivers:
            raise ValueError(
                f"Search-space derived rule for {name!r} has non-active drivers: "
                f"{unknown_drivers}"
            )
        derived_rules[str(name)] = copy.deepcopy(rule)
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

    fixed_runtime_parameters = [
        name
        for name in effective_limits
        if name not in result["active_search_limits"]
        and name not in derived_runtime_parameters
    ]

    result["integration"]["connected_to_mainflow"] = True
    result["integration"]["note"] = (
        "This Session-frozen result is connected to workflow/continuous via "
        "the generic injection contract and is revalidated before submission."
    )
    result["integration"]["search_space_profile"] = profile_name
    result["integration"]["profile_mode"] = mode
    result["integration"]["approval_source"] = (
        "deterministic_compatibility_and_runtime_validation"
        if mode == "automatic_registry"
        else "config.search_space.approved_planned_parameters"
    )
    result["integration"]["approved_planned_parameters"] = sorted(approved)
    result["integration"]["effective_candidate_parameters"] = list(effective_limits)
    result["integration"]["effective_search_limits"] = copy.deepcopy(
        effective_limits
    )
    result["integration"]["classified_search_limits"] = copy.deepcopy(
        result.get(
            "classified_search_limits",
            {
                "active": result["active_search_limits"],
                "reserve": {
                    str(item["canonical_name"]): item["values"]
                    for item in result.get("reserve_candidates", [])
                },
            },
        )
    )
    result["integration"]["derived_runtime_parameters"] = list(
        derived_runtime_parameters
    )
    result["integration"]["fixed_runtime_parameters"] = list(
        fixed_runtime_parameters
    )
    config["manual_search_limits"] = manual_limits
    config["search_limits"] = effective_limits
    config["baseline"] = baseline
    config["search_limits_resolved"] = True
    if automatic_registry is not None:
        parameters = {
            str(item["canonical_name"]): item
            for item in automatic_registry["parameters"]
        }
        active_names = list(result["active_search_limits"])
        validation_active = copy.deepcopy(result["active_parameters"])
        for item in validation_active:
            item["values"] = copy.deepcopy(
                effective_limits[str(item["canonical_name"])]
            )
        config["automatic_registry_validation"] = {
            "scenario": copy.deepcopy(compiler_scenario),
            "compatibility_policy": copy.deepcopy(
                automatic_pipeline.compatibility.policy
            ),
            "compiled": {
                "active_parameters": validation_active,
            },
            "active_injections": {
                name: copy.deepcopy(parameters[name]["injection"])
                for name in active_names
            },
        }
    config["resolved_search_space"] = {
        "mode": mode,
        "profile": profile_name,
        "history": str(history_path) if history_path else None,
        "previous_selection": (str(previous_selection) if previous_selection else None),
        "active_tunable_parameters": list(result["active_search_limits"]),
        "reserve_tunable_parameters": list(result.get("reserve_search_limits", {})),
        "derived_runtime_parameters": derived_runtime_parameters,
        "fixed_runtime_parameters": fixed_runtime_parameters,
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
    profile = config.get("search_space", {})
    if isinstance(profile, dict) and profile.get("resolved_profile"):
        (output / "search_space_profile.yaml").write_text(
            yaml.safe_dump(
                {
                    "profile": profile.get("profile"),
                    "definition": profile.get("resolved_profile"),
                },
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
    compiled = copy.deepcopy(result)
    automatic_registry = compiled.pop("automatic_registry_snapshot", None)
    if isinstance(automatic_registry, dict):
        (output / "registry.generated.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": automatic_registry["schema_version"],
                    "mode": automatic_registry["mode"],
                    "parameters": automatic_registry["parameters"],
                    "combination_constraints": automatic_registry[
                        "combination_constraints"
                    ],
                },
                allow_unicode=True,
                sort_keys=False,
                width=120,
            ),
            encoding="utf-8",
        )
        (output / "registry.audit.yaml").write_text(
            yaml.safe_dump(
                {
                    "provenance": automatic_registry["provenance"],
                    "audit": automatic_registry["audit"],
                    "compatibility_audit": automatic_registry["compatibility_audit"],
                    "recall_outcomes": automatic_registry["recall_outcomes"],
                },
                allow_unicode=True,
                sort_keys=False,
                width=120,
            ),
            encoding="utf-8",
        )
    (output / "classified_search_limits.yaml").write_text(
        yaml.safe_dump(
            compiled.get(
                "classified_search_limits",
                {
                    "active": compiled["active_search_limits"],
                    "reserve": compiled.get("reserve_search_limits", {}),
                },
            ),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (output / "search_space.compiled.yaml").write_text(
        yaml.safe_dump(compiled, allow_unicode=True, sort_keys=False),
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
