#!/usr/bin/env python3
"""Discover and validate the repository's user-facing scenario packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
VALID_STATUSES = {"integrated", "planned", "retired"}


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def discover() -> dict[str, tuple[Path, dict[str, Any]]]:
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(HERE.glob("*/scenario.yaml")):
        document = load_yaml(path)
        scenario = document.get("scenario", document)
        if not isinstance(scenario, dict):
            raise ValueError(f"scenario must be a mapping: {path}")
        scenario_id = str(scenario.get("id", "")).strip()
        if not scenario_id:
            raise ValueError(f"scenario.id is missing: {path}")
        if scenario_id in result:
            raise ValueError(f"Duplicate scenario id: {scenario_id}")
        result[scenario_id] = (path.parent, scenario)
    return result


def repo_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def validate_one(package: Path, scenario: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    scenario_id = str(scenario.get("id", ""))
    if package.name != scenario_id:
        errors.append(f"package directory must match scenario.id ({scenario_id})")
    if scenario.get("status") not in VALID_STATUSES:
        errors.append(f"status must be one of {sorted(VALID_STATUSES)}")
    for field in (
        "summary",
        "model",
        "topology",
        "runtime",
        "benchmark",
        "search_space",
        "agent",
        "required_inputs",
        "artifacts",
        "outputs",
    ):
        if not scenario.get(field):
            errors.append(f"missing scenario.{field}")

    entry = scenario.get("entry", {})
    if not isinstance(entry, dict):
        errors.append("scenario.entry must be a mapping")
    else:
        for field in ("default_config", "operator_template", "runtime_root"):
            if not entry.get(field):
                errors.append(f"missing scenario.entry.{field}")
        for field in ("default_config", "operator_template"):
            if entry.get(field) and not repo_path(entry[field]).is_file():
                errors.append(f"missing {field}: {entry[field]}")

    runtime = scenario.get("runtime", {})
    if isinstance(runtime, dict):
        selector = runtime.get("selector")
        if selector == "external_adapter":
            adapter_path = repo_path(runtime.get("adapter_file", ""))
            if not adapter_path.is_file():
                errors.append(f"missing runtime adapter: {runtime.get('adapter_file')}")
            else:
                adapter = load_yaml(adapter_path).get("adapter", {})
                if adapter.get("status") != scenario.get("status"):
                    errors.append("scenario status differs from external adapter status")
        elif selector == "registry_profile":
            registry_value = scenario.get("artifacts", {}).get("runtime_registry")
            registry = load_yaml(repo_path(registry_value)) if registry_value else {}
            if runtime.get("profile") not in registry.get("profiles", {}):
                errors.append(f"unknown runtime registry profile: {runtime.get('profile')}")
        else:
            errors.append("runtime.selector must be registry_profile or external_adapter")

    artifacts = scenario.get("artifacts", {})
    if isinstance(artifacts, dict):
        for name, value in artifacts.items():
            if not repo_path(value).is_file():
                errors.append(f"missing artifact {name}: {value}")
    else:
        errors.append("scenario.artifacts must be a mapping")
    for section, artifact_name, profile_key in (
        ("benchmark", "benchmark_registry", "default_profile"),
        ("search_space", "search_space_registry", "default_profile"),
        ("agent", "strategy_registry", "strategy"),
    ):
        selection = scenario.get(section, {})
        registry_value = artifacts.get(artifact_name) if isinstance(artifacts, dict) else None
        if not isinstance(selection, dict) or not registry_value:
            continue
        registry = load_yaml(repo_path(registry_value))
        profiles = registry.get("profiles", registry.get("strategies", {}))
        if selection.get(profile_key) not in profiles:
            errors.append(f"unknown {section} profile: {selection.get(profile_key)}")
    if isinstance(entry, dict) and isinstance(scenario.get("outputs"), dict):
        if entry.get("runtime_root") != scenario["outputs"].get("local"):
            errors.append("entry.runtime_root and outputs.local must match")
    return errors


def resolved_record(
    scenario_id: str, package: Path, scenario: dict[str, Any]
) -> dict[str, Any]:
    entry = dict(scenario["entry"])
    local_config = package / "operator.local.yaml"
    selected_config = local_config if local_config.is_file() else repo_path(entry["default_config"])
    return {
        **scenario,
        "package_dir": str(package.resolve()),
        "selected_config": str(selected_config.resolve()),
        "operator_local_config": str(local_config.resolve()),
        "operator_template": str(repo_path(entry["operator_template"]).resolve()),
        "runtime_root": str(repo_path(entry["runtime_root"]).resolve()),
        "using_operator_local_config": local_config.is_file(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    show = commands.add_parser("show")
    show.add_argument("name")
    validate = commands.add_parser("validate")
    validate.add_argument("name", nargs="?")
    resolve = commands.add_parser("resolve")
    resolve.add_argument("name")
    args = parser.parse_args()
    scenarios = discover()

    if args.command == "list":
        print("NAME                                      STATUS      MODEL / TOPOLOGY                         BENCHMARK")
        for name, (_, scenario) in scenarios.items():
            model = scenario["model"]
            topology = scenario["topology"]
            identity = (
                f"{model['weight_format']} / {topology['nodes']}x{topology['npu_per_node']} "
                f"NPU / DP{topology['data_parallel_size']}/TP{topology['tensor_parallel_size']}"
            )
            print(
                f"{name:<41} {scenario['status']:<11} {identity:<40} "
                f"{scenario['benchmark']['default_profile']}"
            )
        return 0

    if args.command == "validate" and args.name is None:
        report = {
            name: validate_one(item_package, item)
            for name, (item_package, item) in scenarios.items()
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if any(report.values()) else 0

    if args.name not in scenarios:
        raise SystemExit(
            f"Unknown scenario {args.name!r}; available={sorted(scenarios)}"
        )
    package, scenario = scenarios[args.name]
    if args.command == "show":
        print(yaml.safe_dump({"scenario": scenario}, allow_unicode=True, sort_keys=False))
        return 0
    if args.command == "resolve":
        errors = validate_one(package, scenario)
        if errors:
            print(json.dumps({"valid": False, "errors": errors}, indent=2))
            return 1
        print(json.dumps(resolved_record(args.name, package, scenario), ensure_ascii=False))
        return 0

    report = {args.name: validate_one(package, scenario)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if any(report.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
