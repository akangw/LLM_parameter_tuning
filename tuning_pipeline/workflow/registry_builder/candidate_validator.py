from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from workflow.search_space_compiler.compiler import (
    validate_candidate as validate_machine,
)

from .compatibility import CompatibilityValidator, DEFAULT_POLICY_PATH
from .pipeline import render_generic_injection


def read_object(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def validate_trial_candidate(
    *,
    candidate: dict[str, Any],
    compiled: dict[str, Any],
    scenario: dict[str, Any],
    compatibility: CompatibilityValidator,
    context_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active = {
        str(item["canonical_name"]): item
        for item in compiled.get("active_parameters", [])
    }
    violations: list[dict[str, Any]] = []
    rendered: dict[str, Any] = {}
    for name, value in candidate.items():
        parameter = active.get(str(name))
        if parameter is None:
            violations.append(
                {"id": "parameter_not_active", "parameter": str(name), "value": value}
            )
            continue
        value_key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        allowed_keys = {
            json.dumps(item, ensure_ascii=False, sort_keys=True)
            for item in parameter.get("values", [])
        }
        if value_key not in allowed_keys:
            violations.append(
                {
                    "id": "value_not_in_compiled_domain",
                    "parameter": str(name),
                    "value": value,
                    "allowed_values": parameter.get("values", []),
                }
            )
            continue
        try:
            rendered[str(name)] = render_generic_injection(
                parameter["injection"], value
            )
        except (KeyError, TypeError, ValueError) as exc:
            violations.append(
                {
                    "id": "injection_render_failed",
                    "parameter": str(name),
                    "detail": str(exc),
                }
            )

    merged = {
        **scenario.get("baseline", {}),
        **(context_candidate or {}),
        **candidate,
    }
    violations.extend(
        {"id": identifier, "source": "machine_constraint"}
        for identifier in validate_machine(merged, scenario)
    )
    violations.extend(
        {"id": identifier, "source": "compatibility_constraint"}
        for identifier in compatibility.validate_combination(merged)
    )
    return {
        "schema_version": 1,
        "valid": not violations,
        "candidate": candidate,
        "violations": violations,
        "rendered_injection": rendered,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministically validate one Agent candidate before submission."
    )
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument(
        "--compatibility-policy", type=Path, default=DEFAULT_POLICY_PATH
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenario = read_object(args.scenario)
    candidate_document = read_object(args.candidate)
    candidate = candidate_document.get("params", candidate_document)
    if not isinstance(candidate, dict):
        raise ValueError("Candidate must be an object or contain a params object")
    report = validate_trial_candidate(
        candidate=candidate,
        compiled=read_object(args.compiled),
        scenario=scenario,
        compatibility=CompatibilityValidator(
            scenario=scenario, policy_path=args.compatibility_policy
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
