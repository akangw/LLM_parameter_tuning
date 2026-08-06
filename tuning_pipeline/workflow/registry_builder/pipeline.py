from __future__ import annotations

import copy
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from workflow.search_space_compiler.compiler import (
    SearchSpaceCompiler,
    write_outputs as write_search_limits,
)

from .builder import (
    CONTINUOUS_DIR,
    PROJECT_ROOT,
    RegistryBuilder,
    file_sha256,
    stable_unique,
)
from .compatibility import CompatibilityValidator, DEFAULT_POLICY_PATH


DEFAULT_SOURCE_ROOT = PROJECT_ROOT.parent / "portrait_pipeline" / "sources"


def snake_case(value: str) -> str:
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    return re.sub(r"[^a-z0-9_]+", "_", value).strip("_")


def semantic_name(record: dict[str, Any]) -> str:
    canonical = str(record["canonical_name"])
    if record.get("source_type") == "nested":
        return snake_case(canonical.split(".")[-1])
    return canonical


def semantic_groups(
    records: list[dict[str, Any]], preferred_names: set[str] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Merge only source surfaces with an explicit portrait relationship.

    Generic leaf names such as ``backend`` or ``method`` are not sufficient
    evidence by themselves because unrelated configuration objects reuse them.
    """
    preferred_names = preferred_names or set()
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(semantic_name(record), []).append(record)
    result: dict[str, list[dict[str, Any]]] = {}
    for leaf, bucket in buckets.items():
        if len(bucket) == 1:
            item = bucket[0]
            if item.get("source_type") == "nested" and leaf not in preferred_names:
                key = "__".join(
                    snake_case(part) for part in str(item["canonical_name"]).split(".")
                )
            else:
                key = leaf
            result[key] = bucket
            continue
        knowledge_owner = {
            str(name): index
            for index, item in enumerate(bucket)
            for name in item.get("knowledge_names", [])
        }
        parents = list(range(len(bucket)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        for index, item in enumerate(bucket):
            for relation in item.get("related_parameters", []):
                if isinstance(relation, dict):
                    target = knowledge_owner.get(str(relation.get("name", "")))
                    if target is not None:
                        union(index, target)
        components: dict[int, list[dict[str, Any]]] = {}
        for index, item in enumerate(bucket):
            components.setdefault(find(index), []).append(item)
        for component in components.values():
            if len(component) > 1:
                key = leaf
            else:
                canonical = str(component[0]["canonical_name"])
                key = "__".join(snake_case(part) for part in canonical.split("."))
            suffix = 2
            unique_key = key
            while unique_key in result:
                unique_key = f"{key}_{suffix}"
                suffix += 1
            result[unique_key] = component
    return result


def normalize_json_path(path: str) -> list[str]:
    parts = path.split(".")
    first = parts[0]
    object_aliases = {
        "CompilationConfig": "compilation_config",
        "SpeculativeConfig": "speculative_config",
        "SchedulerConfig": "scheduler_config",
        "EPLBConfig": "eplb_config",
        "KVTransferConfig": "kv_transfer_config",
        "OffloadConfig": "offload_config",
        "AttentionConfig": "attention_config",
        "OnlineQuantizationConfigArgs": "quantization_config",
        "EngineArgs": "engine_args",
    }
    parts[0] = object_aliases.get(first, snake_case(first))
    return [snake_case(part) for part in parts]


def detect_cli_false_flag(
    record: dict[str, Any], flag: str, source_root: Path = DEFAULT_SOURCE_ROOT
) -> str | None:
    negative = "--no-" + flag.removeprefix("--")
    for location in record.get("source_files", []):
        path = _source_path(str(location), str(record.get("scope", "")), source_root)
        if path is not None and path.is_file():
            try:
                if negative in path.read_text(encoding="utf-8", errors="replace"):
                    return negative
            except OSError:
                continue
    return None


def normalize_injection(
    record: dict[str, Any], source_root: Path = DEFAULT_SOURCE_ROOT
) -> dict[str, Any] | None:
    injection = record.get("injection")
    if not isinstance(injection, dict):
        return None
    kind = injection.get("kind")
    if kind in {"cli_value", "cli_bool_flag"} and injection.get("flag"):
        normalized = {"kind": kind, "flag": str(injection["flag"])}
        if record.get("value_type", "").startswith("list["):
            normalized["kind"] = "cli_list"
        elif kind == "cli_bool_flag":
            false_flag = detect_cli_false_flag(
                record, str(injection["flag"]), source_root
            )
            if false_flag:
                normalized["false_flag"] = false_flag
        return normalized
    if kind in {"env_value", "env_bool"} and injection.get("name"):
        return {
            "kind": "env_bool" if kind == "env_bool" else "env_value",
            "name": str(injection["name"]),
        }
    if kind == "nested_field" and injection.get("path"):
        return {
            "kind": "json_path",
            "path": normalize_json_path(str(injection["path"])),
        }
    return None


def render_generic_injection(injection: dict[str, Any], value: Any) -> dict[str, Any]:
    """Validate and render one value using the isolated generic contract."""
    kind = injection.get("kind")
    if kind == "cli_value":
        if isinstance(value, (list, dict)):
            raise ValueError("cli_value requires a scalar")
        return {"cli_args": [] if value is None else [injection["flag"], str(value)]}
    if kind == "cli_list":
        if value is None:
            return {"cli_args": []}
        if not isinstance(value, list):
            raise ValueError("cli_list requires a list or null")
        return {"cli_args": [injection["flag"], *[str(item) for item in value]]}
    if kind == "cli_bool_flag":
        if value is None:
            return {"cli_args": []}
        if not isinstance(value, bool):
            raise ValueError("cli_bool_flag requires bool or null")
        if value:
            return {"cli_args": [injection["flag"]]}
        false_flag = injection.get("false_flag")
        return {"cli_args": [false_flag] if false_flag else []}
    if kind in {"env_value", "env_bool"}:
        if kind == "env_bool" and value is not None and not isinstance(value, bool):
            raise ValueError("env_bool requires bool or null")
        rendered = (
            None
            if value is None
            else ("1" if value is True else "0" if value is False else str(value))
        )
        return {"environment": {injection["name"]: rendered}}
    if kind == "json_path":
        path = injection.get("path")
        if (
            not isinstance(path, list)
            or not path
            or not all(isinstance(part, str) for part in path)
        ):
            raise ValueError("json_path requires a non-empty string path")
        return {"json_patch": {"path": path, "value": value}}
    raise ValueError(f"Unsupported generic injection kind: {kind}")


def _nested_set(target: dict[str, Any], path: list[str], value: Any) -> None:
    if not path:
        raise ValueError("JSON injection path must contain a field below its root")
    current = target
    for part in path[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"JSON injection path collides at {part!r}")
        current = child
    current[path[-1]] = value


def compile_generic_runtime_payload(
    candidate: dict[str, Any], injections: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Render generated parameters into one safe, auditable remote payload."""
    cli_args: list[str] = []
    environment: dict[str, str | None] = {}
    json_configs: dict[str, dict[str, Any]] = {}
    for name in sorted(injections):
        if name not in candidate:
            raise ValueError(f"Missing generated runtime parameter: {name}")
        value = candidate[name]
        injection = injections[name]
        if injection.get("kind") == "json_path":
            path = injection.get("path")
            if (
                not isinstance(path, list)
                or not path
                or not all(isinstance(part, str) and part for part in path)
            ):
                raise ValueError(f"Invalid JSON injection path for {name}")
            # The compatibility layer defines None as an omission action, not
            # a literal JSON null to pass into vLLM.
            if value is not None:
                _nested_set(json_configs.setdefault(path[0], {}), path[1:], value)
            continue
        rendered = render_generic_injection(injection, value)
        cli_args.extend(str(item) for item in rendered.get("cli_args", []))
        environment.update(
            {
                str(key): None if item is None else str(item)
                for key, item in rendered.get("environment", {}).items()
            }
        )
    return {
        "schema_version": 1,
        "cli_args": cli_args,
        "environment": environment,
        "json_configs": json_configs,
    }


def _source_path(location: str, scope: str, source_root: Path) -> Path | None:
    relative = location.rsplit(":", 1)[0]
    if relative.startswith("vllm_ascend/"):
        return source_root / "vllm-ascend" / relative
    if relative.startswith("vllm/"):
        return source_root / "vllm" / relative
    if relative.startswith("docs/"):
        checkout = "vllm-ascend" if scope == "vllm-ascend" else "vllm"
        return source_root / checkout / relative
    return None


def verify_source_capability(
    record: dict[str, Any], scenario: dict[str, Any], source_root: Path
) -> dict[str, Any]:
    exact = bool(scenario.get("image", {}).get("parameter_portrait_exact_match"))
    injection = record.get("injection") or {}
    primary_token = (
        injection.get("flag")
        or injection.get("name")
        or str(injection.get("path", "")).split(".")[-1]
    )
    tokens = stable_unique(
        token
        for token in (
            primary_token,
            str(record.get("canonical_name", "")).split(".")[-1],
        )
        if token
    )
    checked: list[str] = []
    matches: list[str] = []
    for location in record.get("source_files", []):
        path = _source_path(str(location), str(record.get("scope", "")), source_root)
        if path is None or not path.is_file():
            continue
        checked.append(str(path))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(str(token) in text for token in tokens):
            matches.append(str(path))
    verified = bool(exact and matches)
    return {
        "status": "verified_from_exact_pinned_source" if verified else "unverified",
        "scenario_exact_match": exact,
        "tokens": tokens,
        "checked_files": checked,
        "matching_files": matches,
    }


def verify_source_identity(
    source_root: Path, scenario: dict[str, Any]
) -> dict[str, Any]:
    expected = {
        "vllm": str(scenario.get("image", {}).get("vllm_commit", "")),
        "vllm-ascend": str(scenario.get("image", {}).get("vllm_ascend_commit", "")),
    }
    actual: dict[str, str] = {}
    for checkout, expected_commit in expected.items():
        path = source_root / checkout
        if not path.is_dir() or not expected_commit:
            raise ValueError(
                f"Pinned source identity is incomplete for {checkout}: "
                f"path={path}, expected_commit={expected_commit!r}"
            )
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode:
            raise ValueError(
                completed.stderr.strip()
                or f"Unable to resolve pinned source HEAD: {path}"
            )
        actual[checkout] = completed.stdout.strip()
        if actual[checkout] != expected_commit:
            raise ValueError(
                f"Pinned source mismatch for {checkout}: "
                f"actual={actual[checkout]}, expected={expected_commit}"
            )
    return {
        "source_root": str(source_root),
        "expected": expected,
        "actual": actual,
    }


def _risk_rank(value: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(value, 1)


class AutomaticRegistryPipeline:
    """Build an executable registry and compile Search Limits.

    The builder never reads the curated registry. It can run standalone for
    isolated review or be selected as a frozen Controller Session profile.
    """

    def __init__(
        self,
        *,
        knowledge_dir: Path,
        scenario_path: Path,
        policy_path: Path,
        compatibility_policy_path: Path | None = None,
        source_root: Path | None = None,
        activation_override: dict[str, Any] | None = None,
    ) -> None:
        self.knowledge_dir = knowledge_dir.resolve()
        self.scenario_path = scenario_path.resolve()
        self.policy_path = policy_path.resolve()
        self.compatibility_policy_path = (
            compatibility_policy_path or DEFAULT_POLICY_PATH
        ).resolve()
        self.source_root = (source_root or DEFAULT_SOURCE_ROOT).resolve()
        self.activation_override = copy.deepcopy(activation_override or {})
        self.scenario = yaml.safe_load(self.scenario_path.read_text(encoding="utf-8"))
        self.source_identity = verify_source_identity(self.source_root, self.scenario)
        self.compatibility = CompatibilityValidator(
            scenario=self.scenario,
            policy_path=self.compatibility_policy_path,
        )

    def build_registry(self) -> dict[str, Any]:
        proposal = RegistryBuilder(
            knowledge_dir=self.knowledge_dir,
            scenario_path=self.scenario_path,
            policy_path=self.policy_path,
        ).build()
        sections = (
            proposal["generated_candidates"]
            + proposal["review_queue"]
            + proposal["unsupported"]
        )
        eligible_records = [
            item
            for item in sections
            if "deprecated_parameter" not in item.get("review_reasons", [])
            and item.get("value_type") not in {"unknown", "dict"}
            and len(item.get("candidate_values", [])) >= 2
            and normalize_injection(item, self.source_root) is not None
        ]

        preferred_names = set(self.scenario.get("baseline", {}))
        groups = semantic_groups(eligible_records, preferred_names)
        knowledge_to_key: dict[str, str] = {}
        for key, items in groups.items():
            for item in items:
                for name in item.get("knowledge_names", []):
                    knowledge_to_key[str(name)] = key

        deprecated_aliases: dict[str, list[str]] = {}
        for item in sections:
            if "deprecated_parameter" not in item.get("review_reasons", []):
                continue
            for relation in item.get("related_parameters", []):
                if not isinstance(relation, dict):
                    continue
                text = str(relation.get("relation", "")).lower()
                target = str(relation.get("name", ""))
                if not any(
                    word in text
                    for word in ("preferred", "replacement", "replaces", "overrides")
                ):
                    continue
                key = knowledge_to_key.get(target)
                if key:
                    deprecated_aliases.setdefault(key, []).extend(
                        item.get("knowledge_names", [])
                    )

        parameters: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        compatibility_audit: list[dict[str, Any]] = []
        accepted_primary_names: set[str] = set()
        accepted_alias_names: set[str] = set()
        injection_rank = {"cli": 0, "nested": 1, "env": 2}
        for key, records in sorted(groups.items()):
            verified: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for record in records:
                evidence = verify_source_capability(
                    record, self.scenario, self.source_root
                )
                if evidence["status"].startswith("verified"):
                    verified.append((record, evidence))
            if not verified:
                rejected.append(
                    {
                        "canonical_name": key,
                        "knowledge_names": stable_unique(
                            name
                            for item in records
                            for name in item.get("knowledge_names", [])
                        ),
                        "reason": "source_or_image_capability_not_verified",
                    }
                )
                continue
            preferred, capability = sorted(
                verified,
                key=lambda pair: injection_rank.get(str(pair[0].get("source_type")), 9),
            )[0]
            injection = normalize_injection(preferred, self.source_root)
            values = stable_unique(
                value for item in records for value in item.get("candidate_values", [])
            )
            if len(values) < 2 or injection is None:
                rejected.append(
                    {
                        "canonical_name": key,
                        "reason": "insufficient_values_or_injection",
                    }
                )
                continue
            risk = max(
                (str(item.get("risk", "medium")) for item in records),
                key=_risk_rank,
            )
            aliases = stable_unique(
                [
                    *(
                        name
                        for item in records
                        for name in item.get("knowledge_names", [])
                    ),
                    *deprecated_aliases.get(key, []),
                ]
            )
            raw_parameter = {
                "canonical_name": key,
                "knowledge_names": aliases,
                "candidate_values": values,
                "risk": risk,
                "integration_status": "generated",
                "injection": injection,
                "generation": {
                    "semantic_group_members": [
                        item["canonical_name"] for item in records
                    ],
                    "preferred_source_type": preferred.get("source_type"),
                    "capability": capability,
                    "controller_contract": "generic_injection_schema_v1",
                    "value_sources": [
                        source
                        for item in records
                        for source in item.get("value_sources", [])
                    ],
                    "constraints_evidence": stable_unique(
                        constraint
                        for item in records
                        for constraint in item.get("constraints_evidence", [])
                    ),
                },
            }
            decision = self.compatibility.validate_parameter(raw_parameter)
            compatibility_audit.append(decision["audit"])
            if not decision["accepted"]:
                rejected.append(decision["audit"])
                continue
            compatible = decision["parameter"]
            try:
                rendered_examples = [
                    render_generic_injection(compatible["injection"], value)
                    for value in compatible["candidate_values"]
                ]
            except (KeyError, TypeError, ValueError) as exc:
                rejection = {
                    "canonical_name": key,
                    "reason": "generic_controller_injection_validation_failed",
                    "detail": str(exc),
                }
                rejected.append(rejection)
                compatibility_audit.append(
                    {**rejection, "status": "excluded_fail_closed"}
                )
                continue
            compatible["generation"]["rendered_value_examples"] = rendered_examples
            accepted_primary_names.update(
                str(name)
                for item in records
                for name in item.get("knowledge_names", [])
            )
            accepted_alias_names.update(
                str(name) for name in deprecated_aliases.get(key, [])
            )
            parameters.append(compatible)
        recall_outcomes: list[dict[str, Any]] = []
        for preliminary_status, items in (
            ("generated_candidate", proposal["generated_candidates"]),
            ("review_required", proposal["review_queue"]),
            ("unsupported", proposal["unsupported"]),
        ):
            for item in items:
                names = [str(name) for name in item.get("knowledge_names", [])]
                if any(name in accepted_primary_names for name in names):
                    final_status = "accepted_to_automatic_registry"
                elif any(name in accepted_alias_names for name in names):
                    final_status = "merged_as_deprecated_alias"
                else:
                    final_status = "excluded_fail_closed"
                recall_outcomes.append(
                    {
                        "knowledge_names": names,
                        "canonical_name": item.get("canonical_name"),
                        "preliminary_status": preliminary_status,
                        "final_status": final_status,
                        "reasons": item.get("review_reasons", []),
                    }
                )
        return {
            "schema_version": 1,
            "mode": "automatic_registry_v1",
            "parameters": parameters,
            "provenance": {
                **proposal["inputs"],
                "source_identity": self.source_identity,
                "compatibility_policy": str(self.compatibility_policy_path),
                "compatibility_policy_sha256": file_sha256(
                    self.compatibility_policy_path
                ),
            },
            "recall_outcomes": recall_outcomes,
            "combination_constraints": self.compatibility.combination_constraints,
            "compatibility_audit": compatibility_audit,
            "audit": {
                "tag_recalled_parameters": proposal["summary"][
                    "tag_recalled_parameters"
                ],
                "proposal_generated": proposal["summary"]["generated_candidates"],
                "proposal_review": proposal["summary"]["review_required_candidates"],
                "proposal_unsupported": proposal["summary"]["unsupported_candidates"],
                "semantic_groups": len(groups),
                "pre_compatibility_semantic_groups": len(groups),
                "compatible_registry_parameters": len(parameters),
                "rejected_groups": rejected,
                "existing_registry_dependency": False,
                "connected_to_mainflow": False,
            },
        }

    def compile(self) -> tuple[dict[str, Any], dict[str, Any]]:
        registry = self.build_registry()
        with tempfile.TemporaryDirectory(prefix="vllmtkb-auto-registry-") as temporary:
            registry_path = Path(temporary) / "registry.generated.yaml"
            registry_path.write_text(
                yaml.safe_dump(
                    {"schema_version": 1, "parameters": registry["parameters"]},
                    allow_unicode=True,
                    sort_keys=False,
                    width=120,
                ),
                encoding="utf-8",
            )
            result = SearchSpaceCompiler(
                knowledge_dir=self.knowledge_dir,
                scenario_path=self.scenario_path,
                registry_path=registry_path,
                policy_path=self.policy_path,
                activation_override=self.activation_override,
            ).compile()
        result["integration"]["connected_to_mainflow"] = False
        result["integration"]["registry_source"] = "automatic_registry_generated"
        return registry, result


def write_full_outputs(
    registry: dict[str, Any], search_result: dict[str, Any], output_dir: Path
) -> list[Path]:
    output = output_dir.resolve()
    if output == CONTINUOUS_DIR.resolve() or CONTINUOUS_DIR.resolve() in output.parents:
        raise ValueError(
            "Automatic registry pipeline cannot write under workflow/continuous"
        )
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)
    registry_path = output / "registry.generated.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "mode": registry["mode"],
                "parameters": registry["parameters"],
                "combination_constraints": registry["combination_constraints"],
            },
            allow_unicode=True,
            sort_keys=False,
            width=120,
        ),
        encoding="utf-8",
    )
    audit_path = output / "registry.audit.json"
    audit_path.write_text(
        json.dumps(
            {
                **registry["audit"],
                "provenance": registry["provenance"],
                "recall_outcomes": registry["recall_outcomes"],
                "compatibility_audit": registry["compatibility_audit"],
                "combination_constraints": registry["combination_constraints"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    search_files = write_search_limits(search_result, output / "search_limits")
    constraints_path = output / "compatibility_constraints.yaml"
    constraints_path.write_text(
        yaml.safe_dump(
            {"constraints": registry["combination_constraints"]},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    files = [registry_path, audit_path, constraints_path, *search_files]
    manifest_path = output / "pipeline_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "automatic_registry_to_search_limits_isolated",
                "connected_to_mainflow": False,
                "files": {
                    str(path.relative_to(output)).replace("\\", "/"): file_sha256(path)
                    for path in files
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    files.append(manifest_path)
    return files
