from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from workflow.search_space_compiler.history import load_trials


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
CONTINUOUS_DIR = PROJECT_ROOT / "workflow" / "continuous"
DEFAULT_RULES = HERE / "default_rules.yaml"
PROPOSAL_FAILURES = {
    "parameter_invalid",
    "parameter_oom",
    "parameter_runtime",
    "parameter_regression",
}


def _read(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _dotted_get(value: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _render(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _resolve(
    operand: Any,
    candidate: dict[str, Any],
    scenario: dict[str, Any],
) -> Any:
    if isinstance(operand, dict) and "add" in operand:
        values = [_resolve(item, candidate, scenario) for item in operand["add"]]
        return sum(values) if all(isinstance(value, (int, float)) for value in values) else None
    if isinstance(operand, str):
        if operand in candidate:
            return candidate[operand]
        return _dotted_get(scenario, operand)
    return operand


def _condition(
    condition: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    name = str(condition["parameter"])
    if name not in candidate:
        return False
    actual = candidate[name]
    expected = condition.get("value")
    op = condition.get("op", "eq")
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op == "gt":
        return isinstance(actual, (int, float)) and actual > expected
    if op == "gte":
        return isinstance(actual, (int, float)) and actual >= expected
    if op == "lt":
        return isinstance(actual, (int, float)) and actual < expected
    if op == "lte":
        return isinstance(actual, (int, float)) and actual <= expected
    if op == "in":
        return actual in condition.get("values", [])
    raise ValueError(f"Unsupported condition operator: {op}")


def _rule_violated(
    rule: dict[str, Any],
    candidate: dict[str, Any],
    scenario: dict[str, Any],
) -> bool:
    kind = str(rule["kind"])
    if kind == "lte_or_disabled":
        left = candidate.get(str(rule["left"]))
        right = candidate.get(str(rule["right"]))
        return (
            left is not None
            and right is not None
            and left != rule.get("disabled_value")
            and left > right
        )
    if kind == "product_lte":
        factors = [_resolve(item, candidate, scenario) for item in rule["factors"]]
        right = _resolve(rule["right"], candidate, scenario)
        return (
            right is not None
            and all(isinstance(value, (int, float)) for value in factors)
            and math.prod(factors) > right
        )
    if kind in {"divides", "divides_or_disabled"}:
        if (
            kind == "divides_or_disabled"
            and candidate.get(str(rule["disabled_parameter"]))
            == rule.get("disabled_value")
        ):
            return False
        divisor = _resolve(rule["divisor"], candidate, scenario)
        dividend = _dotted_get(scenario, str(rule["dividend_scenario"]))
        if divisor is None:
            return False
        return (
            not isinstance(divisor, int)
            or divisor <= 0
            or not isinstance(dividend, int)
            or dividend % divisor != 0
        )
    if kind == "implies":
        return _condition(rule["if"], candidate) and not _condition(
            rule["then"], candidate
        )
    if kind == "conditional_comparison":
        if not _condition(rule["if"], candidate):
            return False
        left = _resolve(rule["left"], candidate, scenario)
        right = _resolve(rule["right"], candidate, scenario)
        operator = str(rule["op"])
        if left is None or right is None:
            return True
        comparisons = {
            "gt": lambda: left > right,
            "gte": lambda: left >= right,
            "lt": lambda: left < right,
            "lte": lambda: left <= right,
            "eq": lambda: left == right,
            "ne": lambda: left != right,
        }
        if operator not in comparisons:
            raise ValueError(
                f"Unsupported conditional comparison operator: {operator}"
            )
        try:
            satisfied = comparisons[operator]()
        except TypeError:
            return True
        return not satisfied
    if kind == "list_max_lte":
        values = candidate.get(str(rule["values"]))
        right = candidate.get(str(rule["right"]))
        return (
            isinstance(values, list)
            and bool(values)
            and isinstance(right, (int, float))
            and max(values) > right
        )
    if kind == "all_multiples":
        values = candidate.get(str(rule["values"]))
        factor = _resolve(rule["factor"], candidate, scenario)
        return (
            isinstance(values, list)
            and isinstance(factor, int)
            and factor > 0
            and any(not isinstance(value, int) or value % factor for value in values)
        )
    if kind == "forbidden_combination":
        values = rule.get("values", {})
        return isinstance(values, dict) and all(
            candidate.get(name) == expected for name, expected in values.items()
        )
    raise ValueError(f"Unsupported runtime rule kind: {kind}")


def _scope_from_scenario(scenario: dict[str, Any] | None) -> dict[str, Any]:
    if not scenario:
        return {}
    scope: dict[str, Any] = {}
    mappings = {
        "scenario_id": "scenario_id",
        "image.digest": "image.digest",
        "model.architecture": "model.architecture",
        "topology.tensor_parallel_size": "topology.tensor_parallel_size",
        "topology.data_parallel_size": "topology.data_parallel_size",
    }
    for output_name, input_path in mappings.items():
        value = _dotted_get(scenario, input_path)
        if value is not None:
            scope[output_name] = value
    return scope


def _scope_matches(scope: dict[str, Any], scenario: dict[str, Any]) -> bool:
    return all(_dotted_get(scenario, path) == expected for path, expected in scope.items())


def _limit_mapping(value: Any) -> dict[str, list[Any]] | None:
    if not isinstance(value, dict):
        return None
    for field in ("active_search_limits", "search_limits"):
        if isinstance(value.get(field), dict):
            return value[field]
    if all(isinstance(item, list) for item in value.values()):
        return value
    return None


class RuntimeRuleStore:
    """Independent deterministic rule store with conservative history feedback."""

    def __init__(
        self,
        path: Path,
        *,
        allow_continuous_session: bool = False,
    ) -> None:
        self.path = path.resolve()
        if _inside(self.path, CONTINUOUS_DIR) and not allow_continuous_session:
            raise ValueError("Runtime rule sidecar refuses workflow/continuous paths")
        value = _read(self.path)
        if not isinstance(value, dict):
            raise ValueError("Rule store root must be an object")
        for field in ("rules", "quarantines", "proposals", "processed_history"):
            if not isinstance(value.get(field), list):
                raise ValueError(f"Rule store field {field!r} must be a list")
        self.data = value

    @classmethod
    def initialize(
        cls,
        path: Path,
        *,
        defaults_path: Path = DEFAULT_RULES,
        allow_continuous_session: bool = False,
    ) -> "RuntimeRuleStore":
        path = path.resolve()
        if _inside(path, CONTINUOUS_DIR) and not allow_continuous_session:
            raise ValueError("Runtime rule sidecar refuses workflow/continuous paths")
        if path.exists():
            raise FileExistsError(f"Rule store already exists: {path}")
        value = _read(defaults_path.resolve())
        if not isinstance(value, dict):
            raise ValueError("Default rules must be an object")
        value = copy.deepcopy(value)
        value["created_at"] = dt.datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        value["safety_policy"] = {
            "auto_activate": "static_source_proven_rules_only",
            "history_failures": "exact_candidate_proposal_only",
            "generalized_rules": "proposal_only",
            "remote_execution": "none",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return cls(
            path,
            allow_continuous_session=allow_continuous_session,
        )

    def save(self) -> None:
        self.data["updated_at"] = dt.datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        self.path.write_text(
            yaml.safe_dump(self.data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def evaluate(
        self,
        candidate: dict[str, Any],
        *,
        scenario: dict[str, Any] | None = None,
        search_limits: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        scenario = scenario or {}
        violations: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        limits = _limit_mapping(search_limits)
        if limits is not None:
            for name, value in candidate.items():
                if name not in limits:
                    violations.append(
                        {
                            "id": f"search_limits_missing:{name}",
                            "source": "session_search_limits",
                            "detail": f"{name} is absent from Search Limits",
                        }
                    )
                elif value not in limits[name]:
                    violations.append(
                        {
                            "id": f"search_limits_value:{name}",
                            "source": "session_search_limits",
                            "detail": f"{name}={value!r} is outside the frozen values",
                        }
                    )

        for rule in self.data["rules"]:
            if not isinstance(rule, dict) or rule.get("status") not in {
                "active",
                "shadow",
            }:
                continue
            if _rule_violated(rule, candidate, scenario):
                item = {
                    "id": rule.get("id"),
                    "source": rule.get("source"),
                    "kind": rule.get("kind"),
                }
                (violations if rule["status"] == "active" else warnings).append(item)

        for quarantine in self.data["quarantines"]:
            if not isinstance(quarantine, dict) or quarantine.get("status") != "active":
                continue
            if not _scope_matches(quarantine.get("scope", {}), scenario):
                continue
            name = str(quarantine["parameter"])
            if name in candidate and candidate[name] == quarantine.get("value"):
                violations.append(
                    {
                        "id": quarantine.get("id"),
                        "source": "history_quarantine",
                        "detail": quarantine.get("reason"),
                    }
                )

        for proposal in self.data["proposals"]:
            if not isinstance(proposal, dict) or proposal.get("status") not in {
                "proposed",
                "shadow",
                "active",
            }:
                continue
            if not _scope_matches(proposal.get("scope", {}), scenario):
                continue
            expression = proposal.get("expression")
            if isinstance(expression, dict) and _rule_violated(
                expression, candidate, scenario
            ):
                item = {
                    "id": proposal.get("id"),
                    "source": "history_rule_proposal",
                    "status": proposal.get("status"),
                }
                (
                    violations
                    if proposal.get("status") == "active"
                    else warnings
                ).append(item)

        return {
            "schema_version": 1,
            "evaluated_at": dt.datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "allowed": not violations,
            "violations": violations,
            "warnings": warnings,
            "store": str(self.path),
        }

    def ingest_history(
        self,
        history_path: Path,
        *,
        scenario: dict[str, Any] | None = None,
        proposal_threshold: int = 2,
    ) -> dict[str, Any]:
        history_path = history_path.resolve()
        source_sha = _sha256(history_path)
        if source_sha in self.data["processed_history"]:
            return {
                "status": "already_processed",
                "source_sha256": source_sha,
                "quarantines_added": 0,
                "proposals_added": 0,
            }

        trials, metadata = load_trials(history_path)
        session_id = str(metadata.get("session_id", history_path.stem))
        scope = _scope_from_scenario(scenario)
        quarantines_added = 0
        proposals_added = 0
        combination_evidence: dict[str, dict[str, Any]] = {}

        for trial in trials:
            classification = trial.get("failure_classification")
            if trial.get("status") != "failure" or classification not in PROPOSAL_FAILURES:
                continue
            attributed = sorted(set(trial.get("attributed_parameters", [])))
            if not attributed:
                continue
            evidence_id = f"{session_id}:{trial['trial_id']}"
            attributed_values = {
                name: trial["params"].get(name)
                for name in attributed
                if name in trial["params"]
            }
            # A single attributed value is not automatically a universal fact:
            # OOM and most runtime-invalid outcomes depend on the complete
            # capacity/graph/communication combination.  Preserve the exact
            # failed candidate as a non-blocking proposal; only checked-in,
            # source-proven rules are active without explicit review.
            if attributed_values:
                exact_values = dict(trial.get("params", {}))
                key = _render({"scope": scope, "values": exact_values})
                record = combination_evidence.setdefault(
                    key,
                    {
                        "values": exact_values,
                        "attributed_parameters": set(),
                        "classifications": set(),
                        "evidence": set(),
                    },
                )
                record["attributed_parameters"].update(attributed)
                record["classifications"].add(str(classification))
                record["evidence"].add(evidence_id)

        for key, record in combination_evidence.items():
            existing = next(
                (
                    item
                    for item in self.data["proposals"]
                    if item.get("fingerprint") == hashlib.sha256(key.encode()).hexdigest()
                ),
                None,
            )
            if existing is None:
                existing = {
                    "id": f"proposal:forbidden_combination:{hashlib.sha256(key.encode()).hexdigest()[:12]}",
                    "fingerprint": hashlib.sha256(key.encode()).hexdigest(),
                    "status": "proposed",
                    "scope": scope,
                    "expression": {
                        "kind": "forbidden_combination",
                        "values": record["values"],
                    },
                    "evidence": [],
                    "classifications": [],
                    "activation": "human_or_evidence_review_required",
                }
                self.data["proposals"].append(existing)
                proposals_added += 1
            existing["evidence"] = sorted(
                set(existing["evidence"]) | record["evidence"]
            )
            existing["classifications"] = sorted(
                set(existing["classifications"]) | record["classifications"]
            )
            existing["attributed_parameters"] = sorted(
                set(existing.get("attributed_parameters", []))
                | record["attributed_parameters"]
            )
            existing["evidence_threshold_met"] = (
                len(existing["evidence"]) >= max(2, proposal_threshold)
            )

        self.data["processed_history"].append(source_sha)
        self.save()
        return {
            "status": "processed",
            "source": str(history_path),
            "source_sha256": source_sha,
            "trial_count": len(trials),
            "quarantines_added": quarantines_added,
            "proposals_added": proposals_added,
            "generalized_rules_activated": 0,
        }

    def transition_proposal(self, proposal_id: str, status: str) -> dict[str, Any]:
        allowed = {"proposed", "shadow", "active", "rejected"}
        if status not in allowed:
            raise ValueError(f"Unsupported proposal status {status!r}")
        proposal = next(
            (
                item
                for item in self.data["proposals"]
                if item.get("id") == proposal_id
            ),
            None,
        )
        if proposal is None:
            raise KeyError(f"Proposal not found: {proposal_id}")
        before = proposal.get("status")
        proposal["status"] = status
        proposal.setdefault("transitions", []).append(
            {
                "from": before,
                "to": status,
                "at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "actor": "explicit_cli_action",
            }
        )
        self.save()
        return {"id": proposal_id, "before": before, "after": status}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("init")
    initialize.add_argument("--store", type=Path, required=True)
    initialize.add_argument("--defaults", type=Path, default=DEFAULT_RULES)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--store", type=Path, required=True)
    evaluate.add_argument("--candidate", type=Path, required=True)
    evaluate.add_argument("--scenario", type=Path)
    evaluate.add_argument("--search-limits", type=Path)
    evaluate.add_argument("--output", type=Path)

    ingest = subparsers.add_parser("ingest-history")
    ingest.add_argument("--store", type=Path, required=True)
    ingest.add_argument("--history", type=Path, required=True)
    ingest.add_argument("--scenario", type=Path)
    ingest.add_argument("--proposal-threshold", type=int, default=2)

    transition = subparsers.add_parser("transition")
    transition.add_argument("--store", type=Path, required=True)
    transition.add_argument("--proposal-id", required=True)
    transition.add_argument(
        "--status",
        required=True,
        choices=["proposed", "shadow", "active", "rejected"],
    )
    args = parser.parse_args()

    if args.command == "init":
        print(RuntimeRuleStore.initialize(args.store, defaults_path=args.defaults).path)
        return 0

    store = RuntimeRuleStore(args.store)
    if args.command == "evaluate":
        result = store.evaluate(
            _read(args.candidate),
            scenario=_read(args.scenario) if args.scenario else None,
            search_limits=_read(args.search_limits) if args.search_limits else None,
        )
        rendered = yaml.safe_dump(result, allow_unicode=True, sort_keys=False)
        if args.output:
            output = args.output.resolve()
            if _inside(output, CONTINUOUS_DIR):
                raise ValueError("Runtime rule sidecar refuses workflow/continuous paths")
            if output.exists():
                raise FileExistsError(f"Output already exists: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
            print(output)
        else:
            print(rendered)
        return 0 if result["allowed"] else 2
    if args.command == "ingest-history":
        result = store.ingest_history(
            args.history,
            scenario=_read(args.scenario) if args.scenario else None,
            proposal_threshold=args.proposal_threshold,
        )
        print(yaml.safe_dump(result, allow_unicode=True, sort_keys=False))
        return 0
    result = store.transition_proposal(args.proposal_id, args.status)
    print(yaml.safe_dump(result, allow_unicode=True, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
