"""Resolve the model-loading transport independently from serving parameters."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Model-loading profile document must be a mapping: {path}")
    return value


def resolve_model_loading_profile(
    config: dict[str, Any], project_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze a legacy DTFS contract or an explicitly selected loading Profile."""
    resolved = copy.deepcopy(config)
    setting = resolved.get("model_loading", {})
    if not isinstance(setting, dict):
        raise ValueError("model_loading must be a mapping")

    frozen = setting.get("resolved_profile")
    if isinstance(frozen, dict):
        profile = copy.deepcopy(frozen)
        name = str(setting.get("profile", profile.get("name", "frozen")))
    elif setting.get("profile"):
        profiles_path = Path(
            str(
                setting.get(
                    "profiles_file", "workflow/continuous/model_loading_profiles.yaml"
                )
            )
        )
        if not profiles_path.is_absolute():
            profiles_path = project_root / profiles_path
        document = _yaml(profiles_path)
        name = str(setting["profile"])
        profiles = document.get("profiles", {})
        if not isinstance(profiles, dict) or not isinstance(profiles.get(name), dict):
            raise ValueError(f"Unknown model-loading profile {name!r}: {profiles_path}")
        profile = copy.deepcopy(profiles[name])
    else:
        # Preserve all Sessions created before loading Profiles existed.
        name = "legacy_dtfs_page_cache_v1"
        profile = {
            "status": "integrated",
            "backend": "dtfs_page_cache",
            "load_format": "auto",
            "require_transfer_hit": False,
            "safetensors_prefetch_mode": setting.get(
                "safetensors_prefetch_mode", "node_blocking"
            ),
            "safetensors_load_strategy": setting.get(
                "safetensors_load_strategy", "prefetch"
            ),
            "safetensors_prefetch_num_threads": setting.get(
                "safetensors_prefetch_num_threads", 8
            ),
            "safetensors_prefetch_block_size": setting.get(
                "safetensors_prefetch_block_size", 16 * 1024 * 1024
            ),
        }

    if profile.get("status") != "integrated":
        raise ValueError(f"Model-loading profile {name!r} is not integrated")
    backend = str(profile.get("backend", ""))
    load_format = str(profile.get("load_format", "auto"))
    require_hit = bool(profile.get("require_transfer_hit", False))
    if backend not in {"dtfs_page_cache", "rfork"}:
        raise ValueError(f"Unsupported model-loading backend={backend!r}")
    if backend == "rfork" and load_format != "rfork":
        raise ValueError("RFork model-loading profiles must use load_format=rfork")
    if require_hit and backend != "rfork":
        raise ValueError("require_transfer_hit is valid only for RFork")

    scheduler_url = str(profile.get("rfork_scheduler_url", "")).strip()
    scheduler_env = str(profile.get("rfork_scheduler_url_env", "")).strip()
    if backend == "rfork" and not scheduler_url and scheduler_env:
        scheduler_url = os.environ.get(scheduler_env, "").strip()
    if backend == "rfork":
        for field in ("model_url", "model_deploy_strategy_name"):
            if not str(profile.get(field, "")).strip():
                raise ValueError(f"RFork model-loading profile requires {field}")
        if not scheduler_url:
            raise ValueError(
                "RFork scheduler URL is missing; set "
                f"{scheduler_env or 'model_loading.rfork_scheduler_url'}"
            )
        profile["rfork_scheduler_url"] = scheduler_url

    resolved["model_loading"] = {
        "profile": name,
        "profiles_file": setting.get("profiles_file"),
        "resolved_profile": copy.deepcopy(profile),
    }
    return resolved, profile
