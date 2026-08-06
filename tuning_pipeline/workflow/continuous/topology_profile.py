"""Resolve and validate frozen execution-topology profiles."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


LEGACY_PROFILE = {
    "status": "integrated",
    "executor": "ktp_two_role",
    "nodes": 2,
    "npu_per_node": 16,
    "data_parallel_size": 2,
    "data_parallel_size_local": 1,
    "tensor_parallel_size": 16,
    "data_parallel_rpc_port": 12980,
    "worker_replicas": 1,
    "worker_data_parallel_start_rank": 1,
}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Topology profile document must be a mapping: {path}")
    return value


def resolve_topology_profile(
    config: dict[str, Any], project_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a copied config and its fail-closed, frozen topology profile."""
    resolved_config = copy.deepcopy(config)
    setting = resolved_config.get("topology")
    if not isinstance(setting, dict) or not setting:
        profile_name = "legacy_a3_dp2_tp16"
        profile = copy.deepcopy(LEGACY_PROFILE)
        profiles_file = None
    elif isinstance(setting.get("resolved_profile"), dict):
        profile_name = str(setting.get("profile", "frozen_topology"))
        profile = copy.deepcopy(setting["resolved_profile"])
        profiles_file = setting.get("profiles_file")
    else:
        profiles_file = str(
            setting.get(
                "profiles_file", "workflow/continuous/topology_profiles.yaml"
            )
        )
        path = Path(profiles_file)
        if not path.is_absolute():
            path = project_root / path
        document = _load_yaml(path)
        profile_name = str(setting.get("profile") or document.get("default_profile"))
        profiles = document.get("profiles", {})
        if not isinstance(profiles, dict) or profile_name not in profiles:
            raise ValueError(
                f"Unknown topology profile {profile_name!r}; "
                f"available={sorted(profiles) if isinstance(profiles, dict) else []}"
            )
        profile = copy.deepcopy(profiles[profile_name])

    if profile.get("status") != "integrated":
        raise ValueError(f"Topology profile {profile_name!r} is not integrated")
    integer_fields = (
        "nodes",
        "npu_per_node",
        "data_parallel_size",
        "data_parallel_size_local",
        "tensor_parallel_size",
        "data_parallel_rpc_port",
        "worker_replicas",
        "worker_data_parallel_start_rank",
    )
    for field in integer_fields:
        try:
            profile[field] = int(profile[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Topology profile {profile_name!r} has invalid {field}"
            ) from exc
        if profile[field] < 1:
            raise ValueError(
                f"Topology profile {profile_name!r} requires positive {field}"
            )

    if profile["data_parallel_size"] * profile["tensor_parallel_size"] != (
        profile["nodes"] * profile["npu_per_node"]
    ):
        raise ValueError(
            f"Topology profile {profile_name!r} has inconsistent DP/TP/NPU capacity"
        )

    executor_name = str(profile.get("executor", ""))
    executor_file = Path(
        str(
            profile.get(
                "executor_profiles_file",
                "workflow/continuous/executor_profiles.yaml",
            )
        )
    )
    if not executor_file.is_absolute():
        executor_file = project_root / executor_file
    if not executor_file.is_file() and "executor_profiles_file" not in profile:
        executor_file = Path(__file__).resolve().parent / "executor_profiles.yaml"
    executor_document = _load_yaml(executor_file)
    executors = executor_document.get("executors", {})
    if not isinstance(executors, dict) or executor_name not in executors:
        raise ValueError(
            f"Unknown executor profile {executor_name!r}; "
            f"available={sorted(executors) if isinstance(executors, dict) else []}"
        )
    executor = copy.deepcopy(executors[executor_name])
    if executor.get("status") != "integrated":
        raise ValueError(f"Executor profile {executor_name!r} is not integrated")
    if executor.get("platform") != "ascend":
        raise ValueError(f"Executor profile {executor_name!r} is not an Ascend executor")
    constraints = executor.get("constraints", {})
    if not isinstance(constraints, dict):
        raise ValueError(f"Executor profile {executor_name!r} constraints must be a mapping")
    mismatches = {
        field: {"actual": profile.get(field), "allowed": allowed}
        for field, allowed in constraints.items()
        if not isinstance(allowed, list) or profile.get(field) not in allowed
    }
    if mismatches:
        raise ValueError(
            f"Topology profile {profile_name!r} is incompatible with "
            f"executor {executor_name!r}: {mismatches}"
        )
    profile["resolved_executor"] = {
        "name": executor_name,
        "remote_contract": executor.get("remote_contract"),
        "constraints": constraints,
    }

    resolved_config["topology"] = {
        "profile": profile_name,
        "profiles_file": profiles_file,
        "resolved_profile": copy.deepcopy(profile),
    }
    return resolved_config, profile
