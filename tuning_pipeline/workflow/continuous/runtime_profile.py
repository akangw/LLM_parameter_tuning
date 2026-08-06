"""Resolve an immutable Ascend model/image/topology runtime adapter."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Runtime adapter document must be a mapping: {path}")
    return value


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _artifact_identities(profile: dict[str, Any], project_root: Path) -> dict[str, Any]:
    artifacts = profile.get("artifacts", {})
    if not artifacts:
        return {}
    if not isinstance(artifacts, dict):
        raise ValueError("Runtime profile artifacts must be a mapping")
    identities: dict[str, Any] = {}
    for name, value in artifacts.items():
        path = Path(str(value))
        if not path.is_absolute():
            path = project_root / path
        if not path.is_file():
            raise FileNotFoundError(f"Runtime profile artifact does not exist: {path}")
        data = path.read_bytes()
        identities[str(name)] = {
            "path": _portable_path(path, project_root),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }
    if "scenario" in artifacts and "image_manifest" in artifacts:
        scenario = _yaml(
            project_root / str(artifacts["scenario"])
            if not Path(str(artifacts["scenario"])).is_absolute()
            else Path(str(artifacts["scenario"]))
        )
        manifest = _yaml(
            project_root / str(artifacts["image_manifest"])
            if not Path(str(artifacts["image_manifest"])).is_absolute()
            else Path(str(artifacts["image_manifest"]))
        )
        scenario_image = scenario.get("image", {})
        target = manifest.get("target_image", {})
        versions = manifest.get("versions", {})
        expected = {
            "digest": target.get("digest"),
            "vllm_commit": versions.get("vllm", {}).get("commit"),
            "vllm_ascend_commit": versions.get("vllm_ascend", {}).get("commit"),
        }
        if not isinstance(scenario_image, dict) or any(
            scenario_image.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(
                "Runtime adapter scenario image digest/commits differ from image manifest"
            )
    return identities


def validate_runtime_selections(config: dict[str, Any]) -> None:
    """Reject a Profile that is integrated globally but invalid for this runtime."""
    runtime = config.get("runtime", {})
    profile = runtime.get("resolved_profile", {}) if isinstance(runtime, dict) else {}
    compatibility = profile.get("compatibility", {}) if isinstance(profile, dict) else {}
    allowed = (
        compatibility.get("allowed_profiles", {})
        if isinstance(compatibility, dict)
        else {}
    )
    if not isinstance(allowed, dict) or not allowed:
        # Legacy frozen Sessions predate this contract and retain their exact
        # already-frozen selections instead of inheriting a new allowlist.
        return
    selected = {
        "search_space": config.get("search_space", {}).get("profile"),
        "strategy": config.get("strategy", {}).get("profile"),
        "benchmark": config.get("benchmark", {}).get("profile"),
        "agent_provider": config.get("agent", {}).get("provider", "codex"),
    }
    runtime_name = str(runtime.get("profile", "unknown"))
    for kind, value in selected.items():
        choices = allowed.get(kind)
        if not isinstance(choices, list) or not choices:
            raise ValueError(
                f"Runtime profile {runtime_name!r} lacks an allowlist for {kind}"
            )
        if value not in choices:
            raise ValueError(
                f"{kind} profile {value!r} is incompatible with runtime "
                f"{runtime_name!r}; allowed={choices}"
            )


def resolve_runtime_profile(
    config: dict[str, Any], project_root: Path, *, apply_bindings: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply and freeze a selected runtime adapter before other profiles resolve."""
    source = copy.deepcopy(config)
    setting = source.get("runtime")
    frozen_identity: dict[str, Any] | None = None
    if not isinstance(setting, dict) or not setting:
        # Sessions and autonomous configs created before runtime adapters existed
        # retain their exact frozen selections instead of inheriting a new default.
        topology = source.get("topology", {})
        profile = {
            "status": "integrated",
            "platform": "ascend",
            "model_contract": {
                "family": "glm",
                "variant": "legacy-frozen-glm-5.2",
                "weight_format": "legacy-frozen",
                "supports_mtp": bool(source.get("mtp_draft_model")),
            },
            "config": {},
            "compatibility": {
                "executor": (
                    topology.get("resolved_profile", {}).get("executor", "ktp_two_role")
                    if isinstance(topology, dict)
                    else "ktp_two_role"
                ),
                "requires_new_session": True,
            },
        }
        profile_name = "legacy_implicit_ascend"
        resolved = source
        origin = "legacy_implicit"
    elif isinstance(setting.get("resolved_profile"), dict):
        profile = copy.deepcopy(setting["resolved_profile"])
        profile_name = str(setting.get("profile", profile.get("name", "frozen")))
        resolved = source
        origin = setting.get("origin", "frozen_session")
        if isinstance(setting.get("identity"), dict):
            frozen_identity = copy.deepcopy(setting["identity"])
    else:
        setting = setting if isinstance(setting, dict) else {}
        adapter_file = setting.get("adapter_file")
        if adapter_file:
            path = Path(str(adapter_file))
            if not path.is_absolute():
                path = project_root / path
            document = _yaml(path)
            profile = copy.deepcopy(document.get("adapter", document))
            profile_name = str(profile.get("name") or path.stem)
            origin = _portable_path(path, project_root)
        else:
            profiles_file = Path(
                str(
                    setting.get(
                        "profiles_file", "workflow/continuous/runtime_profiles.yaml"
                    )
                )
            )
            if not profiles_file.is_absolute():
                profiles_file = project_root / profiles_file
            document = _yaml(profiles_file)
            profile_name = str(
                setting.get("profile") or document.get("default_profile")
            )
            profiles = document.get("profiles", {})
            if not isinstance(profiles, dict) or profile_name not in profiles:
                raise ValueError(
                    f"Unknown runtime profile {profile_name!r}; "
                    f"available={sorted(profiles) if isinstance(profiles, dict) else []}"
                )
            profile = copy.deepcopy(profiles[profile_name])
            origin = _portable_path(profiles_file, project_root)

        if profile.get("status") != "integrated":
            blockers = profile.get("readiness", {}).get("blockers", [])
            raise ValueError(
                f"Runtime profile {profile_name!r} is not integrated; blockers={blockers}"
            )
        if profile.get("platform") != "ascend":
            raise ValueError("Only platform=ascend runtime adapters are supported")
        model = profile.get("model_contract")
        if not isinstance(model, dict):
            raise ValueError(f"Runtime profile {profile_name!r} lacks model_contract")
        missing_model = [
            field
            for field in ("family", "variant", "weight_format")
            if not str(model.get(field, "")).strip()
        ]
        if missing_model:
            raise ValueError(
                f"Runtime profile {profile_name!r} lacks model fields: {missing_model}"
            )
        if adapter_file:
            attestations = profile.get("readiness", {}).get("attestations", {})
            missing_attestations = [
                name
                for name in (
                    "executor_validated",
                    "b0_validated",
                    "benchmark_validated",
                    "search_space_validated",
                )
                if attestations.get(name) is not True
            ]
            if missing_attestations:
                raise ValueError(
                    f"Runtime adapter {profile_name!r} lacks validation attestations: "
                    f"{missing_attestations}"
                )
        bindings = profile.get("config", {})
        if not isinstance(bindings, dict):
            raise ValueError(f"Runtime profile {profile_name!r} config must be a mapping")
        # The adapter is the source of truth for compatibility-critical profile
        # selections. Host paths and credentials remain operator overlays.
        resolved = _merge(source, bindings) if apply_bindings else source

    if frozen_identity is not None:
        identity = frozen_identity
    else:
        canonical = json.dumps(profile, ensure_ascii=False, sort_keys=True).encode("utf-8")
        identity = {
            "profile": profile_name,
            "origin": origin,
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "platform": profile.get("platform"),
            "model_contract": copy.deepcopy(profile.get("model_contract", {})),
            "compatibility": copy.deepcopy(profile.get("compatibility", {})),
            "artifacts": _artifact_identities(profile, project_root),
        }
    resolved["runtime"] = {
        "profile": profile_name,
        "origin": origin,
        "resolved_profile": copy.deepcopy(profile),
        "identity": identity,
    }
    return resolved, identity
