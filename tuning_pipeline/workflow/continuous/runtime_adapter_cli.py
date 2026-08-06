#!/usr/bin/env python3
"""Scaffold and validate immutable Ascend runtime adapter packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

try:
    from .topology_profile import resolve_topology_profile
except ImportError:
    from topology_profile import resolve_topology_profile

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DEFAULT_SEARCH_PROFILES = PROJECT_ROOT / "workflow" / "search_space_profiles.yaml"


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _portable(path: Path, root: Path = PROJECT_ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def validate_adapter_document(
    document: dict[str, Any], *, project_root: Path = PROJECT_ROOT,
    require_integrated: bool = True,
) -> list[str]:
    adapter = document.get("adapter", document)
    if not isinstance(adapter, dict):
        return ["adapter must be a mapping"]
    errors: list[str] = []
    for field in ("name", "status", "platform", "model_contract", "config"):
        if not adapter.get(field):
            errors.append(f"missing adapter.{field}")
    if adapter.get("platform") != "ascend":
        errors.append("platform must be ascend")
    if require_integrated and adapter.get("status") != "integrated":
        errors.append(f"adapter status {adapter.get('status')!r} is not runnable")
        readiness = adapter.get("readiness", {})
        if isinstance(readiness, dict):
            errors.extend(str(item) for item in readiness.get("blockers", []))
    model = adapter.get("model_contract", {})
    if isinstance(model, dict):
        for field in ("family", "variant", "weight_format"):
            if not str(model.get(field, "")).strip():
                errors.append(f"missing model_contract.{field}")
    else:
        errors.append("model_contract must be a mapping")
    config = adapter.get("config", {})
    if not isinstance(config, dict):
        return errors + ["config must be a mapping"]

    try:
        _, topology = resolve_topology_profile(config, project_root)
        expected_executor = adapter.get("compatibility", {}).get("executor")
        if expected_executor and topology.get("executor") != expected_executor:
            errors.append("compatibility.executor differs from topology executor")
    except Exception as exc:
        errors.append(f"topology: {exc}")

    image = config.get("image_identity", {})
    if not isinstance(image, dict):
        errors.append("config.image_identity must be a mapping")
        image = {}
    manifest_path = _path(str(image.get("manifest", "")), project_root)
    activation_path = _path(str(image.get("activation", "")), project_root)
    manifest: dict[str, Any] = {}
    if not manifest_path.is_file():
        errors.append(f"missing image manifest: {manifest_path}")
    else:
        manifest = _yaml(manifest_path)
    if not activation_path.is_file():
        errors.append(f"missing image activation: {activation_path}")
    elif manifest:
        try:
            try:
                from .continuous_tuning import validate_activation_approval
            except ImportError:
                from continuous_tuning import validate_activation_approval
            validate_activation_approval(manifest, approval_path=activation_path)
        except Exception as exc:
            errors.append(f"image approval: {exc}")

    search = config.get("search_space", {})
    profile = search.get("resolved_profile", {}) if isinstance(search, dict) else {}
    scenario_value = profile.get("scenario") if isinstance(profile, dict) else None
    if not scenario_value:
        errors.append("runtime adapter must freeze a search-space scenario")
    else:
        scenario_path = _path(str(scenario_value), project_root)
        if not scenario_path.is_file():
            errors.append(f"missing scenario: {scenario_path}")
        elif manifest:
            scenario = _yaml(scenario_path)
            scenario_image = scenario.get("image", {})
            target = manifest.get("target_image", {})
            versions = manifest.get("versions", {})
            expected = {
                "digest": target.get("digest"),
                "vllm_commit": versions.get("vllm", {}).get("commit"),
                "vllm_ascend_commit": versions.get("vllm_ascend", {}).get("commit"),
            }
            if not isinstance(scenario_image, dict):
                errors.append("scenario.image must be a mapping")
            elif any(scenario_image.get(key) != value for key, value in expected.items()):
                errors.append("scenario image digest/commits differ from image manifest")

    baseline = config.get("initial_baseline", {}).get("definition")
    if not baseline or not _path(str(baseline), project_root).is_file():
        errors.append(f"missing B0 definition: {baseline!r}")
    readiness = adapter.get("readiness", {})
    attestations = readiness.get("attestations", {}) if isinstance(readiness, dict) else {}
    required_attestations = (
        "executor_validated",
        "b0_validated",
        "benchmark_validated",
        "search_space_validated",
    )
    if adapter.get("status") == "integrated":
        for field in required_attestations:
            if attestations.get(field) is not True:
                errors.append(f"integrated adapter requires readiness.attestations.{field}=true")
    return errors


def scaffold(args: argparse.Namespace) -> dict[str, Any]:
    search_document = _yaml(DEFAULT_SEARCH_PROFILES)
    profiles = search_document.get("profiles", {})
    if args.search_space_profile not in profiles:
        raise ValueError(f"Unknown search-space profile {args.search_space_profile!r}")
    search_profile = dict(profiles[args.search_space_profile])
    search_profile["scenario"] = _portable(args.scenario)
    topology = {
        "status": "integrated",
        "executor": args.executor,
        "nodes": args.nodes,
        "npu_per_node": args.npu_per_node,
        "data_parallel_size": args.data_parallel_size,
        "data_parallel_size_local": args.data_parallel_size_local,
        "tensor_parallel_size": args.tensor_parallel_size,
        "data_parallel_rpc_port": args.data_parallel_rpc_port,
        "worker_replicas": args.worker_replicas,
        "worker_data_parallel_start_rank": args.worker_data_parallel_start_rank,
    }
    attestations = {
        "executor_validated": args.executor_validated,
        "b0_validated": args.b0_validated,
        "benchmark_validated": args.benchmark_validated,
        "search_space_validated": args.search_space_validated,
    }
    adapter = {
        "schema_version": 1,
        "adapter": {
            "name": args.name,
            "status": "integrated" if args.integrated else "planned",
            "platform": "ascend",
            "session_prefix": args.name.replace("-", "_") + "_continuous",
            "model_contract": {
                "family": args.model_family,
                "variant": args.model_variant,
                "weight_format": args.weight_format,
                "supports_mtp": args.supports_mtp,
            },
            "artifacts": {
                "scenario": _portable(args.scenario),
                "baseline": _portable(args.baseline),
                "image_manifest": _portable(args.image_manifest),
                "image_activation": _portable(args.activation),
                "executor_profiles": "workflow/continuous/executor_profiles.yaml",
            },
            "config": {
                "topology": {
                    "profile": args.topology_name or f"{args.name}_topology",
                    "resolved_profile": topology,
                },
                "image_identity": {
                    "manifest": _portable(args.image_manifest),
                    "activation": _portable(args.activation),
                },
                "search_space": {
                    "profile": f"{args.name}_{args.search_space_profile}",
                    "resolved_profile": search_profile,
                    "history_source": "none",
                },
                "strategy": {"profile": args.strategy_profile},
                "benchmark": {"profile": args.benchmark_profile},
                "initial_baseline": {
                    "label": "b0_deployable",
                    "launch_profile": "official_source_defaults_deployable",
                    "definition": _portable(args.baseline),
                },
            },
            "compatibility": {
                "executor": args.executor,
                "requires_new_session": True,
            },
            "readiness": {
                "attestations": attestations,
                "blockers": [],
            },
        },
    }
    errors = validate_adapter_document(adapter, require_integrated=args.integrated)
    if not args.integrated:
        errors.extend(
            f"pending attestation: {name}"
            for name, complete in attestations.items()
            if not complete
        )
    adapter["adapter"]["readiness"]["blockers"] = errors
    if args.integrated and errors:
        raise ValueError("Cannot create integrated adapter:\n- " + "\n- ".join(errors))
    return adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("scaffold")
    create.add_argument("--name", required=True)
    create.add_argument("--model-family", required=True)
    create.add_argument("--model-variant", required=True)
    create.add_argument("--weight-format", required=True)
    create.add_argument("--supports-mtp", action="store_true")
    create.add_argument("--image-manifest", type=Path, required=True)
    create.add_argument("--activation", type=Path, required=True)
    create.add_argument("--scenario", type=Path, required=True)
    create.add_argument("--baseline", type=Path, required=True)
    create.add_argument("--topology-name")
    create.add_argument("--nodes", type=int, required=True)
    create.add_argument("--npu-per-node", type=int, required=True)
    create.add_argument("--data-parallel-size", type=int, required=True)
    create.add_argument("--data-parallel-size-local", type=int, default=1)
    create.add_argument("--tensor-parallel-size", type=int, required=True)
    create.add_argument("--data-parallel-rpc-port", type=int, default=12980)
    create.add_argument("--worker-replicas", type=int, default=1)
    create.add_argument("--worker-data-parallel-start-rank", type=int, default=1)
    create.add_argument("--executor", default="ktp_two_role")
    create.add_argument("--search-space-profile", default="automatic_registry_v1")
    create.add_argument("--strategy-profile", default="best_anchor_coverage_v2")
    create.add_argument("--benchmark-profile", default="vllm_bench_public_v1")
    create.add_argument("--executor-validated", action="store_true")
    create.add_argument("--b0-validated", action="store_true")
    create.add_argument("--benchmark-validated", action="store_true")
    create.add_argument("--search-space-validated", action="store_true")
    create.add_argument("--integrated", action="store_true")
    create.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("adapter", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "scaffold":
        document = scaffold(args)
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite adapter: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(args.output)
        return 0
    document = _yaml(args.adapter)
    errors = validate_adapter_document(document)
    print(json.dumps({"adapter": str(args.adapter), "valid": not errors, "errors": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
