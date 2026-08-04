#!/usr/bin/env python3
"""Continuous GLM-5.2 tuning controller.

The controller runs on Windows beside the local knowledge base. The server is
only used to execute one parameterized experiment at a time. Remote experiment
artifacts are never deleted by this program.
"""

from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import yaml

if str(Path(__file__).resolve().parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from .search_space_adapter import resolve_search_limits, write_session_search_space
    from .agent_provider import (
        resolve_agent_profile,
        run_structured_agent,
        validate_agent_credentials,
    )
except ImportError:  # Direct script execution.
    from search_space_adapter import resolve_search_limits, write_session_search_space
    from agent_provider import (
        resolve_agent_profile,
        run_structured_agent,
        validate_agent_credentials,
    )
from workflow.search_space_compiler.compiler import (
    validate_candidate as validate_search_space_candidate,
)
from workflow.sidecars.portrait_retriever import PortraitRetriever
from workflow.sidecars.runtime_rule_store import RuntimeRuleStore

HERE = Path(__file__).resolve().parent
KB_ROOT = HERE.parent.parent
ARCHIVE_ROOT = HERE / "experiments"
STATE_FILE = HERE / "state.json"
STOP_FILE = HERE / "STOP_REQUESTED"
LOG_FILE = Path(
    os.environ.get(
        "VLLMTKB_CONTROLLER_LOG",
        HERE / "logs" / "controller" / "controller.log",
    )
)
LOCK_FILE = HERE / "controller.lock"
IMAGE_MANIFEST_FILE = HERE / "remote" / "image_version_manifest.yaml"
REMOTE_ARTIFACTS = (
    "candidate.env",
    "task.yaml",
    "effective_config.yaml",
    "vllm_common_command.txt",
    "models_response.json",
    "run_status.json",
    "server_run_manifest.yaml",
    "startup_timeline.jsonl",
    "master.log",
    "worker.log",
    "warmup.log",
    "formal.log",
    "benchmark_runner.log",
    "benchmark_watchdog.log",
    "benchmark_case_retry_plan.json",
    "BENCHMARK_RUNTIME_RETRY_STATE",
    "BENCHMARK_RUNTIME_RETRY_RECOVERED",
    "BENCHMARK_CASE_RETRY_RECOVERED",
    "BENCHMARK_METRICS_RETRY_STATE",
    "BENCHMARK_METRICS_RETRY_RECOVERED",
    "BENCHMARK_WATCHDOG_STARTED",
    "BENCHMARK_WATCHDOG_TIMEOUT",
    "SERVICE_READY",
    "BENCHMARK_STARTED",
    "BENCHMARK_DONE",
    "BENCHMARK_FAILED",
    "metrics.json",
    "MASTER_DONE",
)
REMOTE_SCRIPT_NAMES = (
    "WORKSPACE_MANIFEST.md",
    "ARTIFACT_LAYOUT.md",
    "common_runtime_loop.sh",
    "run_master_loop.sh",
    "run_worker_loop.sh",
    "submit_candidate.sh",
    "experiment_loop.yaml",
    "lease_loop.yaml",
    "image_version_manifest.yaml",
    "extract_metrics.py",
    "aligned_l1_metrics.py",
    "run_aligned_l1.sh",
    "run_servebench_attempt.sh",
    "benchmark_watchdog.sh",
    "validate_aligned_l1_inputs.py",
)


class RepeatedCandidateRejection(RuntimeError):
    """Raised after the Agent repeatedly fails deterministic candidate validation."""
ALL_PARAM_TO_ENV = {
    "max_num_seqs": "MAX_NUM_SEQS",
    "max_model_len": "MAX_MODEL_LEN",
    "max_num_batched_tokens": "MAX_NUM_BATCHED_TOKENS",
    "gpu_memory_utilization": "GPU_MEMORY_UTILIZATION",
    "enable_prefix_caching": "ENABLE_PREFIX_CACHING",
    "async_scheduling": "ASYNC_SCHEDULING",
    "enable_expert_parallel": "ENABLE_EXPERT_PARALLEL",
    "compilation_mode": "COMPILATION_MODE",
    "num_speculative_tokens": "NUM_SPECULATIVE_TOKENS",
    "long_prefill_token_threshold": "LONG_PREFILL_TOKEN_THRESHOLD",
    "enable_chunked_prefill": "ENABLE_CHUNKED_PREFILL",
    "max_cudagraph_capture_size": "MAX_CUDAGRAPH_CAPTURE_SIZE",
    "enable_eplb": "ENABLE_EPLB",
    "eplb_num_redundant_experts": "EPLB_NUM_REDUNDANT_EXPERTS",
    "compilation_enable_sp": "COMPILATION_ENABLE_SP",
    "cudagraph_capture_sizes": "CUDAGRAPH_CAPTURE_SIZES_JSON",
    "decode_context_parallel_size": "DECODE_CONTEXT_PARALLEL_SIZE",
    "flashcomm1": "ADDITIONAL_CONFIG_ENABLE_FLASHCOMM1",
    "mlapo": "ADDITIONAL_CONFIG_ENABLE_MLAPO",
    "fused_mc2": "ADDITIONAL_CONFIG_ENABLE_FUSED_MC2",
}


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def run_process(
    args: list[str],
    *,
    cwd: Path | None = None,
    stdin: str | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        input=stdin,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def resolve_codex_command(configured: str | None = None) -> str:
    """Resolve Codex without binding the project to one operator's home directory."""
    override = os.environ.get("VLLMTKB_CODEX_COMMAND", "").strip()
    requested = override or str(configured or "auto").strip()
    if requested.lower() == "auto":
        resolved = shutil.which("codex.cmd") or shutil.which("codex")
    else:
        candidate = Path(requested).expanduser()
        resolved = str(candidate) if candidate.is_file() else shutil.which(requested)
    if not resolved:
        raise RuntimeError(
            "Codex CLI was not found. Install/login Codex and ensure it is on PATH, "
            "or set VLLMTKB_CODEX_COMMAND to its executable path."
        )
    return resolved


class Controller:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        dry_run: bool = False,
        search_space_result: dict[str, Any] | None = None,
    ):
        self.config = copy.deepcopy(config)
        config = self.config
        self.dry_run = dry_run
        self.search_space_result = search_space_result
        if set(config["baseline"]) != set(config["search_limits"]):
            raise ValueError("baseline and search_limits parameter schemas differ")
        unknown_parameters = sorted(set(config["baseline"]) - set(ALL_PARAM_TO_ENV))
        if unknown_parameters:
            raise ValueError(
                f"No runtime injection contract for parameters: {unknown_parameters}"
            )
        self.param_to_env = {
            name: ALL_PARAM_TO_ENV[name] for name in config["baseline"]
        }
        self.remote_host = config["remote_host"]
        self.remote_transport = str(
            config.get("remote_transport", "native_ssh")
        )
        if self.remote_transport not in {"native_ssh", "paramiko"}:
            raise ValueError(
                f"Unsupported remote_transport={self.remote_transport!r}"
            )
        self._paramiko_client_cache: Any | None = None
        self.remote_project = config["remote_project"]
        self.remote_auto = f"{self.remote_project}/workflow/auto"
        self.poll_seconds = int(config["poll_seconds"])
        self.round_timeout_minutes = int(config.get("round_timeout_minutes", 750))
        self.partial_exit_grace_seconds = int(
            config.get("partial_exit_grace_seconds", 120)
        )
        self.execution_mode = config.get("execution_mode", "ktp")
        self.agent_config = resolve_agent_profile(
            config.get("agent", {}),
            legacy_command=str(config.get("codex_command", "auto")),
        )
        self.mtp_draft_model = str(config.get("mtp_draft_model", "")).strip()
        self.model_loading = dict(config.get("model_loading", {}))
        self.safetensors_load_strategy = str(
            self.model_loading.get("safetensors_load_strategy", "prefetch")
        ).strip()
        self.safetensors_prefetch_num_threads = int(
            self.model_loading.get("safetensors_prefetch_num_threads", 8)
        )
        self.safetensors_prefetch_block_size = int(
            self.model_loading.get("safetensors_prefetch_block_size", 16 * 1024 * 1024)
        )
        if self.safetensors_load_strategy not in {"prefetch", "eager", "lazy"}:
            raise ValueError(
                "model_loading.safetensors_load_strategy must be prefetch, eager, or lazy"
            )
        if not 1 <= self.safetensors_prefetch_num_threads <= 32:
            raise ValueError(
                "model_loading.safetensors_prefetch_num_threads must be between 1 and 32"
            )
        if not 1024 * 1024 <= self.safetensors_prefetch_block_size <= 1024**3:
            raise ValueError(
                "model_loading.safetensors_prefetch_block_size must be between 1 MiB and 1 GiB"
            )
        self.lab = config.get("lab", {})
        self.image_manifest = load_yaml(IMAGE_MANIFEST_FILE)
        target = self.image_manifest.get("target_image", {})
        versions = self.image_manifest.get("versions", {})
        self.image_identity = {
            "reference": f"{target.get('repository', '')}:{target.get('tag', '')}",
            "digest": target.get("digest"),
            "vllm_commit": versions.get("vllm", {}).get("commit"),
            "vllm_ascend_commit": versions.get("vllm_ascend", {}).get("commit"),
        }
        if not all(self.image_identity.values()):
            raise ValueError(
                f"Incomplete image identity in {IMAGE_MANIFEST_FILE}"
            )
        self.measurement_policy = config.get("measurement_policy", {})
        benchmark_settings = dict(config.get("benchmark", {}))
        if not benchmark_settings:
            benchmark_settings = {
                "profile": "legacy_random_32k1k",
                "legacy_random_32k1k": dict(config.get("fixed_scenario", {})),
            }
        elif not benchmark_settings.get("profile") and benchmark_settings.get("mode"):
            legacy_mode = str(benchmark_settings["mode"])
            benchmark_settings["profile"] = (
                "aligned_l1_v4" if legacy_mode == "aligned_l1" else legacy_mode
            )
        frozen_benchmark_profile = benchmark_settings.get("resolved_profile")
        if isinstance(frozen_benchmark_profile, dict):
            self.benchmark_profile_name = str(benchmark_settings["profile"])
            self.benchmark_profile = dict(frozen_benchmark_profile)
        else:
            benchmark_profiles_path = Path(
                benchmark_settings.get(
                    "profiles_file", "workflow/continuous/benchmark_profiles.yaml"
                )
            )
            if not benchmark_profiles_path.is_absolute():
                benchmark_profiles_path = KB_ROOT / benchmark_profiles_path
            benchmark_profiles_doc = load_yaml(benchmark_profiles_path)
            self.benchmark_profile_name = str(
                benchmark_settings.get("profile")
                or benchmark_profiles_doc.get("default_profile")
            )
            benchmark_profiles = benchmark_profiles_doc.get("profiles", {})
            if self.benchmark_profile_name not in benchmark_profiles:
                raise ValueError(
                    f"Unknown benchmark profile {self.benchmark_profile_name!r}; "
                    f"available={sorted(benchmark_profiles)}"
                )
            self.benchmark_profile = dict(
                benchmark_profiles[self.benchmark_profile_name]
            )
        if self.benchmark_profile.get("status") != "integrated":
            raise ValueError(
                f"Benchmark profile {self.benchmark_profile_name!r} is not integrated"
            )
        self.benchmark_mode = str(self.benchmark_profile.get("mode", ""))
        if self.benchmark_mode not in {"aligned_l1", "legacy_random_32k1k"}:
            raise ValueError(f"Unsupported benchmark.mode={self.benchmark_mode!r}")
        definition_key = str(
            self.benchmark_profile.get("definition_key", self.benchmark_mode)
        )
        definition = benchmark_settings.get(definition_key)
        if not isinstance(definition, dict):
            raise ValueError(
                f"benchmark.{definition_key} must configure profile "
                f"{self.benchmark_profile_name!r}"
            )
        # Normalize the selected definition so existing execution paths consume
        # only the frozen profile, not an accidental global mode toggle.
        self.benchmark = {
            "profile": self.benchmark_profile_name,
            "mode": self.benchmark_mode,
            self.benchmark_mode: dict(definition),
        }
        self.config.setdefault("benchmark", {})["profile"] = (
            self.benchmark_profile_name
        )
        self.config["benchmark"][definition_key] = dict(definition)
        self.config["benchmark"]["resolved_profile"] = dict(
            self.benchmark_profile
        )
        if self.benchmark_mode == "aligned_l1":
            aligned = self.benchmark.get("aligned_l1")
            if not isinstance(aligned, dict):
                raise ValueError("benchmark.aligned_l1 must be configured")
            required = (
                "suite",
                "repetitions",
                "served_model",
                "service_port",
                "tokenizer",
                "servebench_root",
                "servebench_workspace",
                "servebench_version",
                "servebench_commit",
                "docker_image",
                "guidellm_activation",
                "spec_root",
                "dataset_root",
                "suite_sha256",
                "schema_files_sha256",
                "tokenizer_files_sha256",
                "dataset_manifests_sha256",
                "primary_concurrency",
                "workloads",
            )
            missing = [field for field in required if not aligned.get(field)]
            if missing:
                raise ValueError(
                    f"benchmark.aligned_l1 is missing required fields: {missing}"
                )
            if int(aligned["repetitions"]) < 1:
                raise ValueError("aligned L1 repetitions must be positive")
            for field in (
                "case_retry_limit",
                "runtime_retry_limit",
                "metrics_retry_limit",
                "total_full_retry_limit",
            ):
                retry_limit = int(aligned.get(field, 2))
                if not 0 <= retry_limit <= 3:
                    raise ValueError(f"aligned L1 {field} must be between 0 and 3")
        self.change_policy = config.get("change_policy", {})
        strategy_settings = dict(config.get("strategy", {}))
        frozen_strategy_profile = strategy_settings.get("resolved_profile")
        if isinstance(frozen_strategy_profile, dict):
            self.strategy_profile_name = str(strategy_settings["profile"])
            self.strategy_profile = dict(frozen_strategy_profile)
        else:
            profiles_path = Path(
                strategy_settings.get(
                    "profiles_file", "workflow/continuous/strategy_profiles.yaml"
                )
            )
            if not profiles_path.is_absolute():
                profiles_path = KB_ROOT / profiles_path
            profiles_doc = load_yaml(profiles_path)
            self.strategy_profile_name = str(
                strategy_settings.get("profile") or profiles_doc.get("default_strategy")
            )
            strategies = profiles_doc.get("strategies", {})
            if self.strategy_profile_name not in strategies:
                raise ValueError(
                    f"Unknown strategy profile {self.strategy_profile_name!r}; "
                    f"available={sorted(strategies)}"
                )
            self.strategy_profile = dict(strategies[self.strategy_profile_name])
        if self.strategy_profile.get("status") != "integrated":
            raise ValueError(f"Strategy {self.strategy_profile_name!r} is not integrated")
        self.config.setdefault("strategy", {})["profile"] = self.strategy_profile_name
        self.config["strategy"]["resolved_profile"] = dict(self.strategy_profile)
        exploration_profile = dict(self.strategy_profile.get("exploration", {}))
        refinement_profile = dict(
            self.strategy_profile.get("local_refinement")
            or self.strategy_profile.get("refinement", {})
        )
        adaptive = self.change_policy.setdefault("adaptive", {})
        # A selected profile is an explicit strategy choice, so its phase
        # limits must remain active even in minimal/custom configurations.
        adaptive["enabled"] = True
        for phase, profile, fallback in (
            ("exploration", exploration_profile, [2, 3]),
            ("refinement", refinement_profile, [1, 2]),
        ):
            if not profile:
                continue
            preferred = profile.get("independent_parameters_per_round", fallback)
            adaptive.setdefault(phase, {}).update(
                preferred_parameters_per_round=preferred,
                minimum_parameters_per_round=int(
                    profile.get("minimum_independent_parameters", 1)
                ),
                max_parameters_per_round=max(preferred),
                max_grid_steps_per_parameter=profile.get(
                    "max_grid_steps_per_parameter", 2
                ),
                max_total_grid_steps=profile.get("max_total_grid_steps", 6),
            )
        self.max_parameters_per_round = int(
            self.change_policy.get("max_parameters_per_round", 3)
        )
        self.max_grid_steps_per_parameter = int(
            self.change_policy.get("max_grid_steps_per_parameter", 2)
        )
        self.max_total_grid_steps = int(
            self.change_policy.get("max_total_grid_steps", 4)
        )
        self.adaptive_change_policy = self.change_policy.get("adaptive", {})
        self.derived_parameter_rules = self.change_policy.get(
            "derived_parameters", {}
        )
        self.max_candidate_reselections = int(
            self.change_policy.get("max_candidate_reselections", 2)
        )
        if not 0 <= self.max_candidate_reselections <= 5:
            raise ValueError("max_candidate_reselections must be between 0 and 5")
        if not 1 <= self.max_parameters_per_round <= len(self.param_to_env):
            raise ValueError("max_parameters_per_round is outside the candidate schema")
        if self.max_grid_steps_per_parameter < 1 or self.max_total_grid_steps < 1:
            raise ValueError("change-policy grid step budgets must be positive")
        if self.execution_mode not in {"ktp", "ktp_lab"}:
            raise ValueError(f"Unsupported execution_mode={self.execution_mode!r}")
        if self.execution_mode == "ktp_lab" and not self.lab.get("lease_name"):
            raise ValueError("lab.lease_name is required in ktp_lab mode")
        self.sidecar_settings = config.get("sidecars", {})
        self.sidecars_enabled = bool(
            self.sidecar_settings.get("enabled", False)
        )
        self.portrait_retriever: PortraitRetriever | None = None
        self.runtime_rule_store: RuntimeRuleStore | None = None
        self.sidecar_scenario: dict[str, Any] = {}

    def _paramiko_client(self) -> Any:
        if self._paramiko_client_cache is not None:
            transport = self._paramiko_client_cache.get_transport()
            if transport is not None and transport.is_active():
                return self._paramiko_client_cache
        import paramiko

        ssh_config_path = Path.home() / ".ssh" / "config"
        resolved: dict[str, Any] = {}
        if ssh_config_path.is_file():
            parser = paramiko.SSHConfig()
            with ssh_config_path.open("r", encoding="utf-8") as handle:
                parser.parse(handle)
            resolved = parser.lookup(self.remote_host)
        hostname = str(resolved.get("hostname", self.remote_host))
        port = int(resolved.get("port", 22))
        username = resolved.get("user")
        identities = [
            str(Path(value).expanduser())
            for value in resolved.get("identityfile", [])
        ]
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect(
            hostname=hostname,
            port=port,
            username=username,
            key_filename=identities or None,
            allow_agent=True,
            look_for_keys=True,
            timeout=15,
            banner_timeout=15,
            auth_timeout=15,
        )
        self._paramiko_client_cache = client
        return client

    def _ssh_paramiko(self, command: str, *, timeout: int) -> str:
        marker = "__CONTINUOUS_REMOTE_RC__="
        wrapped = (
            "{\n"
            f"{command}\n"
            "} 2>&1\n"
            "__continuous_rc=$?\n"
            f"printf '\\n{marker}%s\\n' \"$__continuous_rc\"\n"
            "exit \"$__continuous_rc\""
        )
        client = self._paramiko_client()
        _, stdout, stderr = client.exec_command(wrapped, timeout=timeout)
        output = stdout.read().decode("utf-8", errors="replace").strip()
        errors = stderr.read().decode("utf-8", errors="replace").strip()
        match = re.search(rf"(?:^|\n){re.escape(marker)}(\d+)\s*$", output)
        if not match:
            raise RuntimeError(
                errors
                or output
                or f"SSH did not return a command status marker: {command}"
            )
        remote_rc = int(match.group(1))
        output = output[:match.start()].strip()
        if remote_rc != 0:
            details = "\n".join(part for part in (output, errors) if part)
            raise RuntimeError(
                details or f"Remote command failed ({remote_rc}): {command}"
            )
        return output

    def ssh(self, command: str, *, timeout: int = 120) -> str:
        if self.remote_transport == "paramiko":
            return self._ssh_paramiko(command, timeout=timeout)
        marker = "__CONTINUOUS_REMOTE_RC__="
        wrapped = (
            "{\n"
            f"{command}\n"
            "} 2>&1\n"
            "__continuous_rc=$?\n"
            f"printf '\\n{marker}%s\\n' \"$__continuous_rc\"\n"
            "exit \"$__continuous_rc\""
        )
        result = run_process(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                self.remote_host,
                wrapped,
            ],
            timeout=timeout,
        )
        # The SSH service can return a nonzero transport status after a valid
        # remote command. Trust the explicit remote return-code marker instead.
        match = re.search(rf"(?:^|\n){re.escape(marker)}(\d+)\s*$", result.stdout)
        if not match:
            raise RuntimeError(
                result.stderr.strip()
                or result.stdout.strip()
                or f"SSH did not return a command status marker: {command}"
            )
        remote_rc = int(match.group(1))
        output = result.stdout[:match.start()].strip()
        if remote_rc != 0:
            details = "\n".join(
                part for part in (output, result.stderr.strip()) if part
            )
            raise RuntimeError(details or f"Remote command failed ({remote_rc}): {command}")
        return output

    @staticmethod
    def _json_schema_type(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "string"

    @classmethod
    def _json_schema_for_values(cls, values: list[Any]) -> dict[str, Any]:
        types = {cls._json_schema_type(value) for value in values}
        schemas: list[dict[str, Any]] = []
        for value_type in sorted(types):
            if value_type == "array":
                members = [
                    member
                    for value in values
                    if isinstance(value, list)
                    for member in value
                ]
                member_types = {
                    cls._json_schema_type(member) for member in members
                } or {"string"}
                if "array" in member_types or "object" in member_types:
                    raise ValueError(
                        "Nested array/object candidate values are not supported "
                        "by the strict Agent schema"
                    )
                schemas.append(
                    {
                        "type": "array",
                        "items": {
                            "type": (
                                sorted(member_types)
                                if len(member_types) > 1
                                else next(iter(member_types))
                            )
                        },
                    }
                )
            elif value_type == "object":
                raise ValueError(
                    "Object-valued candidate parameters require an explicit "
                    "strict schema before activation"
                )
            else:
                schemas.append({"type": value_type})
        return schemas[0] if len(schemas) == 1 else {"anyOf": schemas}

    def write_decision_schemas(self, session_dir: Path) -> None:
        schema_dir = session_dir / "00_search_space"
        schema_dir.mkdir(parents=True, exist_ok=True)
        candidate_properties: dict[str, Any] = {}
        for name, values in self.config["search_limits"].items():
            candidate_properties[name] = self._json_schema_for_values(
                [self.config["baseline"][name], *values]
            )
        for filename in ("agent_decision.schema.json", "failure_decision.schema.json"):
            schema = json.loads((HERE / filename).read_text(encoding="utf-8"))
            candidate_schema = schema["properties"]["candidate"]
            candidate_schema["required"] = list(self.config["search_limits"])
            candidate_schema["properties"] = candidate_properties
            save_json(schema_dir / filename, schema)

    def _sidecar_project_path(self, key: str, default: str) -> Path:
        value = self.sidecar_settings.get(key, default)
        return (KB_ROOT / str(value)).resolve()

    @staticmethod
    def _compact_portrait_bundle(result: dict[str, Any]) -> dict[str, Any]:
        compact: list[dict[str, Any]] = []
        for group in result.get("changed_parameters", []):
            constraints: list[str] = []
            relations: list[dict[str, Any]] = []
            valid_choices: list[Any] = []
            advice: list[Any] = []
            aliases: list[str] = []
            sources: list[str] = []
            seen_constraints: set[str] = set()
            seen_relations: set[str] = set()
            seen_choices: set[str] = set()
            seen_advice: set[str] = set()
            for variant in group.get("variants", []):
                portrait = variant.get("portrait", {})
                if not isinstance(portrait, dict):
                    continue
                aliases.append(str(portrait.get("name", "")))
                sources.append(str(variant.get("source_file", "")))
                for constraint in portrait.get("constraints", []):
                    rendered = str(constraint)
                    if rendered not in seen_constraints:
                        seen_constraints.add(rendered)
                        constraints.append(rendered)
                for relation in portrait.get("related_parameters", []):
                    key = json.dumps(
                        relation, ensure_ascii=False, sort_keys=True
                    )
                    if key not in seen_relations:
                        seen_relations.add(key)
                        relations.append(relation)
                choice = portrait.get("valid_choices")
                key = json.dumps(choice, ensure_ascii=False, sort_keys=True)
                if choice is not None and key not in seen_choices:
                    seen_choices.add(key)
                    valid_choices.append(choice)
                tuning = portrait.get("tuning_advice")
                if isinstance(tuning, dict):
                    selected = {
                        name: tuning.get(name)
                        for name in ("summary", "suggested_values", "caveats")
                        if tuning.get(name) is not None
                    }
                    key = json.dumps(
                        selected, ensure_ascii=False, sort_keys=True
                    )
                    if selected and key not in seen_advice:
                        seen_advice.add(key)
                        advice.append(selected)
            compact.append(
                {
                    "canonical_name": group.get("canonical_name"),
                    "aliases": sorted(set(filter(None, aliases))),
                    "constraints": constraints,
                    "related_parameters": relations,
                    "valid_choices": valid_choices,
                    "tuning_advice": advice,
                    "portrait_sources": sorted(set(filter(None, sources))),
                }
            )
        return {
            "schema_version": 1,
            "generated_at": result.get("generated_at"),
            "parameters": compact,
            "unresolved_related_names": result.get("unresolved_names", []),
            "agent_contract": (
                "Read and cite these portrait constraints before proposing "
                "a parameter change."
            ),
        }

    def initialize_session_sidecars(self, session_dir: Path) -> None:
        if not self.sidecars_enabled:
            return
        scenario_path = self._sidecar_project_path(
            "scenario",
            "workflow/search_space_compiler/scenario.glm52-a3-aligned-l1.yaml",
        )
        self.sidecar_scenario = load_yaml(scenario_path)
        self.portrait_retriever = PortraitRetriever(
            knowledge_dir=self._sidecar_project_path(
                "knowledge_dir", "tag_params/output/params"
            ),
            registry_path=self._sidecar_project_path(
                "registry", "workflow/search_space_compiler/registry.yaml"
            ),
        )
        self.runtime_rule_store = RuntimeRuleStore.initialize(
            session_dir / "00_search_space" / "runtime_rules.yaml",
            defaults_path=self._sidecar_project_path(
                "default_rules", "workflow/sidecars/default_rules.yaml"
            ),
            allow_continuous_session=True,
        )
        active_names = list(
            self.config.get("resolved_search_space", {}).get(
                "active_tunable_parameters",
                [
                    name
                    for name, values in self.config["search_limits"].items()
                    if len(values) > 1
                ],
            )
        )
        portraits = self.portrait_retriever.retrieve(
            active_names,
            search_limits=self.config["search_limits"],
            scenario=self.sidecar_scenario,
            include_one_hop=bool(
                self.sidecar_settings.get("include_one_hop", True)
            ),
        )
        save_yaml(
            session_dir / "00_search_space" / "parameter_portraits.full.yaml",
            portraits,
        )
        save_yaml(
            session_dir / "00_search_space" / "parameter_portraits.agent.yaml",
            self._compact_portrait_bundle(portraits),
        )

    def load_session_sidecars(self, session_dir: Path) -> None:
        if not self.sidecars_enabled:
            return
        self.sidecar_scenario = load_yaml(
            self._sidecar_project_path(
                "scenario",
                "workflow/search_space_compiler/scenario.glm52-a3-aligned-l1.yaml",
            )
        )
        self.portrait_retriever = PortraitRetriever(
            knowledge_dir=self._sidecar_project_path(
                "knowledge_dir", "tag_params/output/params"
            ),
            registry_path=self._sidecar_project_path(
                "registry", "workflow/search_space_compiler/registry.yaml"
            ),
        )
        store_path = session_dir / "00_search_space" / "runtime_rules.yaml"
        if not store_path.is_file():
            raise RuntimeError(f"Frozen Session runtime rule store is missing: {store_path}")
        self.runtime_rule_store = RuntimeRuleStore(
            store_path,
            allow_continuous_session=True,
        )

    def runtime_rule_evaluation(
        self, candidate: dict[str, Any]
    ) -> dict[str, Any] | None:
        if not self.runtime_rule_store:
            return None
        return self.runtime_rule_store.evaluate(
            candidate,
            scenario=self.sidecar_scenario,
            search_limits=self.config["search_limits"],
        )

    def write_selected_portrait_evidence(
        self,
        round_dir: Path,
        changes: list[dict[str, Any]],
        *,
        prefix: str = "selected",
    ) -> None:
        if not self.portrait_retriever or not changes:
            return
        names = [
            str(item["parameter"]).removeprefix("--").replace("-", "_")
            for item in changes
        ]
        result = self.portrait_retriever.retrieve(
            names,
            search_limits=self.config["search_limits"],
            scenario=self.sidecar_scenario,
            include_one_hop=True,
        )
        unresolved_changed = [
            group["canonical_name"]
            for group in result["changed_parameters"]
            if group.get("variant_count", 0) == 0
        ]
        if unresolved_changed:
            raise ValueError(
                "No parameter portrait evidence for changed parameters: "
                f"{unresolved_changed}"
            )
        save_yaml(
            round_dir
            / "06_agent_analysis"
            / f"{prefix}_parameter_portraits.yaml",
            result,
        )

    def refresh_runtime_rules(self, session_dir: Path, round_dir: Path) -> None:
        if not self.runtime_rule_store:
            return
        history_path = (
            round_dir / "06_agent_analysis" / "runtime_rule_history.json"
        )
        save_json(history_path, self.attempted_history_summary(session_dir))
        audit = self.runtime_rule_store.ingest_history(
            history_path,
            scenario=self.sidecar_scenario,
        )
        save_yaml(
            round_dir / "06_agent_analysis" / "runtime_rule_update.yaml",
            audit,
        )

    @staticmethod
    def decision_schema_path(session_dir: Path, filename: str) -> Path:
        path = session_dir / "00_search_space" / filename
        if not path.is_file():
            raise RuntimeError(f"Frozen Session decision schema is missing: {path}")
        return path

    def scp_to(self, source: Path, destination: str) -> None:
        if self.remote_transport == "paramiko":
            sftp = self._paramiko_client().open_sftp()
            try:
                sftp.put(str(source), destination)
                remote_size = int(sftp.stat(destination).st_size)
            finally:
                sftp.close()
            if remote_size != source.stat().st_size:
                raise RuntimeError(f"SFTP upload size mismatch: {source}")
            return
        result = run_process(
            ["scp", str(source), f"{self.remote_host}:{destination}"],
            timeout=120,
        )
        remote_size = self.ssh(
            f"stat -c %s {shlex.quote(destination)}",
            timeout=30,
        )
        if not remote_size.isdigit() or int(remote_size) != source.stat().st_size:
            raise RuntimeError(result.stderr.strip() or f"SCP upload failed: {source}")

    def scp_from(self, source: str, destination: Path) -> bool:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.remote_transport == "paramiko":
            temporary = destination.with_name(
                f"{destination.name}.download-{os.getpid()}-{time.time_ns()}"
            )
            try:
                sftp = self._paramiko_client().open_sftp()
                try:
                    remote_size = int(sftp.stat(source).st_size)
                    sftp.get(source, str(temporary))
                finally:
                    sftp.close()
                if (
                    not temporary.exists()
                    or temporary.stat().st_size != remote_size
                ):
                    return False
                os.replace(temporary, destination)
                return True
            except (OSError, IOError):
                return False
            finally:
                if temporary.exists():
                    temporary.unlink()
        try:
            remote_size_text = self.ssh(
                f"stat -c %s {shlex.quote(source)}",
                timeout=30,
            )
            remote_size = int(remote_size_text)
        except (RuntimeError, ValueError):
            return False

        temporary = destination.with_name(
            f"{destination.name}.download-{os.getpid()}-{time.time_ns()}"
        )
        result = run_process(
            ["scp", f"{self.remote_host}:{source}", str(temporary)],
            timeout=120,
        )
        try:
            if not temporary.exists() or temporary.stat().st_size != remote_size:
                return False
            os.replace(temporary, destination)
            return True
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def round_dir(session_dir: Path, index: int, label: str) -> Path:
        path = session_dir / f"round_{index:03d}_{label}"
        for child in (
            "00_context",
            "01_query",
            "02_parameters",
            "03_submission",
            "04_runtime",
            "05_results",
            "06_agent_analysis",
        ):
            (path / child).mkdir(parents=True, exist_ok=True)
        return path

    def write_context(self, round_dir: Path, state: dict[str, Any]) -> None:
        scenario = {
            "benchmark_profile": self.benchmark_profile_name,
            "benchmark_mode": self.benchmark_mode,
            "definition": self.benchmark.get(self.benchmark_mode, {}),
        }
        save_yaml(round_dir / "00_context" / "scenario.yaml", scenario)
        save_yaml(
            round_dir / "00_context" / "image_version_manifest.yaml",
            self.image_manifest,
        )
        save_json(
            round_dir / "00_context" / "round_manifest.json",
            {
                "session_id": state["session_id"],
                "round_index": state["round_index"],
                "round_label": state["round_label"],
                "created_at": now(),
                "remote_run_id": state.get("active_run_id"),
                "task_id": state.get("active_task_id"),
                "execution_mode": state.get("execution_mode", self.execution_mode),
                "benchmark_profile": state.get(
                    "benchmark_profile", self.benchmark_profile_name
                ),
                "image_identity": state.get("image_identity", self.image_identity),
            },
        )

    def run_query(self, round_dir: Path) -> Path:
        output_path = round_dir / "01_query" / "glm5.2_search_space.yaml"
        command = [
            sys.executable,
            str(KB_ROOT / "query.py"),
            "--tag",
            "hardware=a3",
            "--tag",
            "model=moe,mla,quantized",
            "--tag",
            "deploy_topology=multi_node",
            "--tag",
            "optimize_target=throughput",
            "--where",
            "performance_impact=high",
            "--show",
            "name,valid_choices,tuning_advice.suggested_values,constraints,category",
            "--format",
            "yaml",
        ]
        if self.benchmark_mode == "legacy_random_32k1k":
            insert_at = command.index("--where")
            command[insert_at:insert_at] = [
                "--tag",
                "deploy_scenario=long_input",
            ]
        result = run_process(command, cwd=KB_ROOT, timeout=120)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(result.stderr.strip() or "Knowledge query returned no data")
        output_path.write_text(result.stdout, encoding="utf-8")
        (round_dir / "01_query" / "query_command.txt").write_text(
            subprocess.list2cmdline(command) + "\n", encoding="utf-8"
        )
        return output_path

    def candidate_env(self, label: str, candidate: dict[str, Any]) -> str:
        lines = [f"ROUND_LABEL={shlex.quote(label)}"]
        initial_baseline = self.config.get("initial_baseline", {})
        initial_label = str(initial_baseline.get("label", "a0"))
        launch_profile = (
            str(initial_baseline.get("launch_profile", "explicit_candidate"))
            if label == initial_label
            else "explicit_candidate"
        )
        lines.append(f"LAUNCH_PROFILE={shlex.quote(launch_profile)}")
        for key, env_name in self.param_to_env.items():
            value = candidate[key]
            if isinstance(value, bool):
                text = "true" if value else "false"
            elif value is None or isinstance(value, (list, dict)):
                text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            else:
                text = str(value)
            lines.append(f"{env_name}={shlex.quote(text)}")
        lines.append(f"MTP_DRAFT_MODEL_PATH={shlex.quote(self.mtp_draft_model)}")
        lines.append(
            "SAFETENSORS_LOAD_STRATEGY="
            + shlex.quote(self.safetensors_load_strategy)
        )
        lines.append(
            "SAFETENSORS_PREFETCH_NUM_THREADS="
            + str(self.safetensors_prefetch_num_threads)
        )
        lines.append(
            "SAFETENSORS_PREFETCH_BLOCK_SIZE="
            + str(self.safetensors_prefetch_block_size)
        )
        lines.append(f"BENCHMARK_MODE={shlex.quote(self.benchmark_mode)}")
        if self.benchmark_mode == "aligned_l1":
            aligned = self.benchmark["aligned_l1"]
            benchmark_env = {
                "BENCHMARK_SUITE": aligned["suite"],
                "BENCHMARK_PHASE": (
                    "" if aligned.get("phase") in {None, "all"} else aligned["phase"]
                ),
                "BENCHMARK_REPETITIONS": aligned["repetitions"],
                "BENCHMARK_SERVED_MODEL": aligned["served_model"],
                "BENCHMARK_SERVICE_PORT": aligned["service_port"],
                "BENCHMARK_TOKENIZER": aligned["tokenizer"],
                "SERVEBENCH_ROOT": aligned["servebench_root"],
                "SERVEBENCH_WORKSPACE": aligned["servebench_workspace"],
                "SERVEBENCH_DOCKER_IMAGE": aligned["docker_image"],
                "GUIDELLM_ACTIVATION": aligned["guidellm_activation"],
                "BENCHMARK_SPEC_ROOT": aligned["spec_root"],
                "BENCHMARK_DATASET_ROOT": aligned["dataset_root"],
                "BENCHMARK_PRIMARY_CONCURRENCY": aligned["primary_concurrency"],
                "BENCHMARK_CASE_RETRY_LIMIT": aligned.get("case_retry_limit", 2),
                "BENCHMARK_RUNTIME_RETRY_LIMIT": aligned.get("runtime_retry_limit", 2),
                "BENCHMARK_METRICS_RETRY_LIMIT": aligned.get("metrics_retry_limit", 2),
                "BENCHMARK_TOTAL_FULL_RETRY_LIMIT": aligned.get(
                    "total_full_retry_limit", 2
                ),
            }
            for env_name, value in benchmark_env.items():
                lines.append(f"{env_name}={shlex.quote(str(value))}")
            fingerprint = {
                "servebench_version": str(aligned["servebench_version"]),
                "servebench_commit": str(aligned["servebench_commit"]),
                "benchmark_container_image": str(aligned["docker_image"]),
                "suite_file": str(aligned["suite"]),
                "suite_sha256": str(aligned["suite_sha256"]),
                "schema_files_sha256": aligned["schema_files_sha256"],
                "tokenizer_files_sha256": aligned["tokenizer_files_sha256"],
                "dataset_manifests_sha256": aligned[
                    "dataset_manifests_sha256"
                ],
            }
            lines.append(
                "BENCHMARK_EXPECTED_FINGERPRINT_JSON="
                + shlex.quote(
                    json.dumps(fingerprint, ensure_ascii=False, sort_keys=True)
                )
            )
        return "\n".join(lines) + "\n"

    def validate_runtime_configuration(self, candidate: dict[str, Any]) -> None:
        self.validate_candidate_invariants(candidate)
        if candidate["num_speculative_tokens"] > 0 and not self.mtp_draft_model:
            raise RuntimeError(
                "mtp_draft_model is empty. Set the exact sliced MTP checkpoint "
                "path supplied by the deployment owner before launching a round."
            )

    def validate_deployment_configuration(self) -> None:
        expected_repository, expected_tag = self.image_identity["reference"].rsplit(":", 1)
        if self.benchmark_mode == "aligned_l1":
            benchmark_image = str(self.benchmark["aligned_l1"]["docker_image"])
            if "@sha256:" not in benchmark_image:
                raise RuntimeError(
                    "benchmark.aligned_l1.docker_image must be digest-qualified"
                )
        lease_path = HERE / "remote" / "lease_loop.yaml"
        lease = load_yaml(lease_path)
        if self.execution_mode == "ktp_lab":
            output_root = str(self.lab.get("output_root", ""))
            if not PurePosixPath(output_root).is_absolute():
                raise RuntimeError(
                    "lab.output_root must be an absolute server path in ktp_lab mode"
                )
            if lease.get("name") != self.lab.get("lease_name"):
                raise RuntimeError(
                    "lab.lease_name does not match remote/lease_loop.yaml name"
                )
            if (
                lease.get("image") != expected_repository
                or lease.get("image_tag") != expected_tag
            ):
                raise RuntimeError(
                    "lease_loop.yaml image does not match image_version_manifest.yaml"
                )
        experiment = load_yaml(HERE / "remote" / "experiment_loop.yaml")
        for task in experiment.get("tasks", []):
            if (
                task.get("image") != expected_repository
                or task.get("image_tag") != expected_tag
            ):
                raise RuntimeError(
                    "experiment_loop.yaml image does not match "
                    "image_version_manifest.yaml"
                )

    def assert_state_image_identity(self, state: dict[str, Any]) -> None:
        recorded = state.get("image_identity")
        if recorded != self.image_identity:
            raise RuntimeError(
                "Controller state image identity is missing or differs from the "
                "current verified image. Start a new session; do not resume this state."
            )

    def validate_candidate_invariants(self, candidate: dict[str, Any]) -> None:
        if set(candidate) != set(self.param_to_env):
            missing = sorted(set(self.param_to_env) - set(candidate))
            extra = sorted(set(candidate) - set(self.param_to_env))
            raise ValueError(
                f"Candidate schema mismatch; missing={missing}, extra={extra}"
            )
        limits = self.config["search_limits"]
        for key, value in candidate.items():
            if value not in limits[key]:
                raise ValueError(
                    f"{key}={value!r} is outside whitelist {limits[key]!r}"
                )
        if self.benchmark_mode == "aligned_l1":
            workloads = self.benchmark["aligned_l1"]["workloads"].values()
            required_tokens = max(
                int(workload["input_tokens"]) + int(workload["output_tokens"])
                for workload in workloads
            )
        else:
            legacy = self.benchmark.get(
                "legacy_random_32k1k", self.config["fixed_scenario"]
            )
            required_tokens = (
                int(legacy["input_tokens"]) + int(legacy["output_tokens"])
            )
        if candidate["max_model_len"] < required_tokens:
            raise ValueError(
                f"max_model_len={candidate['max_model_len']} is below the benchmark "
                f"input+output requirement {required_tokens}"
            )
        threshold = candidate["long_prefill_token_threshold"]
        if threshold and threshold > candidate["max_num_batched_tokens"]:
            raise ValueError(
                "long_prefill_token_threshold cannot exceed "
                "max_num_batched_tokens"
            )
        if (
            candidate.get("enable_chunked_prefill") is False
            and candidate["max_num_batched_tokens"] < candidate["max_model_len"]
        ):
            raise ValueError(
                "Disabling chunked prefill requires max_num_batched_tokens "
                "to be at least max_model_len in the pinned vLLM image"
            )
        speculative_tokens = candidate["num_speculative_tokens"]
        if speculative_tokens > 0:
            if not candidate["async_scheduling"]:
                raise ValueError(
                    "MTP speculative decoding requires async_scheduling in this workflow"
                )
            minimum_budget = candidate["max_num_seqs"] * (speculative_tokens + 1)
            if candidate["max_num_batched_tokens"] < minimum_budget:
                raise ValueError(
                    "max_num_batched_tokens is too small for max_num_seqs and "
                    f"num_speculative_tokens; require at least {minimum_budget}"
                )
        if self.config.get("resolved_search_space", {}).get("mode") == "automated":
            scenario_setting = self.config.get(
                "automated_search_limits", {}
            ).get(
                "scenario",
                "workflow/search_space_compiler/scenario.glm52-a3-aligned-l1.yaml",
            )
            scenario = load_yaml(KB_ROOT / str(scenario_setting))
            violations = validate_search_space_candidate(candidate, scenario)
            if violations:
                raise ValueError(
                    "Candidate violates compiled search-space constraints: "
                    + ", ".join(violations)
                )
        if (
            candidate.get("eplb_num_redundant_experts", 0)
            and not candidate.get("enable_eplb", False)
        ):
            raise ValueError(
                "eplb_num_redundant_experts requires enable_eplb=true"
            )
        if candidate.get("enable_eplb", False):
            raise ValueError(
                "The upstream --enable-eplb CLI is unsupported by the pinned "
                "Ascend platform. Native dynamic EPLB requires a separately "
                "reviewed additional_config integration."
            )
        capture_sizes = candidate.get("cudagraph_capture_sizes")
        maximum_capture = candidate.get("max_cudagraph_capture_size")
        if (
            isinstance(capture_sizes, list)
            and capture_sizes
            and isinstance(maximum_capture, int)
            and max(capture_sizes) != maximum_capture
        ):
            raise ValueError(
                "max_cudagraph_capture_size must equal the largest explicit "
                "cudagraph_capture_sizes value"
            )
        rule_evaluation = self.runtime_rule_evaluation(candidate)
        if rule_evaluation and not rule_evaluation["allowed"]:
            ids = [
                str(item.get("id"))
                for item in rule_evaluation.get("violations", [])
            ]
            raise ValueError(
                "Candidate rejected by frozen runtime rule store: "
                + ", ".join(ids)
            )

    def prepare_lab(self, *, submit: bool) -> str:
        self.validate_deployment_configuration()
        self.validate_runtime_configuration(self.config["baseline"])
        self.sync_remote_scripts()
        lease_name = str(self.lab["lease_name"])
        lease_yaml = str(self.lab.get("lease_yaml", "workflow/auto/lease_loop.yaml"))
        output_root = str(self.lab.get("output_root", "workflow/auto/lab_runs"))
        mode = " --submit" if submit else ""
        return self.ssh(
            f"cd {shlex.quote(self.remote_project)} && "
            f"mkdir -p {shlex.quote(output_root)} && "
            f"ktp-lab lease create -f {shlex.quote(lease_yaml)} "
            f"--output-root {shlex.quote(output_root)}{mode}",
            timeout=300,
        )

    def sync_remote_scripts(self) -> None:
        remote_source = HERE / "remote"
        self.ssh(f"mkdir -p {shlex.quote(self.remote_auto)}")
        for name in REMOTE_SCRIPT_NAMES:
            source = remote_source / name
            if not source.is_file():
                raise RuntimeError(f"Required remote script is missing locally: {source}")
            self.scp_to(source, f"{self.remote_auto}/{name}")

    def ensure_lab_available(self) -> str:
        lease_name = str(self.lab["lease_name"])
        try:
            output = self.ssh(
                f"cd {shlex.quote(self.remote_project)} && "
                f"ktp-lab status --lease {shlex.quote(lease_name)} 2>&1 || true",
                timeout=60,
            )
        except Exception as exc:
            raise RuntimeError(
                f"SSH or persistent lease {lease_name!r} is not available. "
                "Verify SSH connectivity first; if the Lease is absent, run "
                "continuous_tuning.py --prepare-lab after setting mtp_draft_model, "
                "then wait for the Lease to become ready."
            ) from exc
        normalized = output.lower()
        resource_active = bool(
            re.search(r"\bresource\s+status=active\b", normalized)
        )
        nodes_ready = bool(
            re.search(r"\bnodes=2/2\s+ready\b", normalized)
        )
        service_idle = bool(re.search(r"\bstatus\s+idle=2\b", normalized))
        # A brand-new persistent lease has no service slot until its first
        # `ktp-lab run`.  Treat that state as available; subsequent runs must
        # see both nodes idle.  Running/partial slots match neither condition
        # and therefore remain fail-closed.
        fresh_without_slots = bool(re.search(r"\bslots\s+none\b", normalized))
        if not (
            resource_active
            and nodes_ready
            and (service_idle or fresh_without_slots)
        ):
            raise RuntimeError(
                f"Persistent lease {lease_name!r} is not idle and ready:\n{output}"
            )
        return output

    def check_ready(self) -> str:
        """Run a read-only end-to-end launch preflight."""
        self.validate_deployment_configuration()
        self.validate_runtime_configuration(self.config["baseline"])
        return self.ensure_lab_available()

    def submit_lab(
        self,
        round_dir: Path,
        label: str,
        candidate: dict[str, Any],
        *,
        dry_run: bool,
    ) -> tuple[str | None, str]:
        self.validate_runtime_configuration(candidate)
        env_path = round_dir / "02_parameters" / "candidate.env"
        env_path.write_text(
            self.candidate_env(label, candidate), encoding="utf-8", newline="\n"
        )
        if dry_run:
            # The persistent lease is created once and deliberately stays
            # active between rounds. Re-running `lease create` as a dry-run
            # therefore fails on the existing name even when the lease is
            # healthy and idle. Validate the already-created lease instead;
            # this remains non-submitting and catches the same readiness
            # conditions used by a real round.
            output = self.ensure_lab_available()
            run_id = f"{label}_{dt.datetime.now():%Y%m%d_%H%M%S}"
            (round_dir / "03_submission" / "submit_output.txt").write_text(
                output + "\n", encoding="utf-8"
            )
            save_json(
                round_dir / "03_submission" / "submission.json",
                {
                    "execution_mode": "ktp_lab",
                    "lease_name": self.lab["lease_name"],
                    "run_id": run_id,
                    "submitted_at": now(),
                    "dry_run": True,
                },
            )
            return None, run_id

        self.ensure_lab_available()
        run_id = f"{label}_{dt.datetime.now():%Y%m%d_%H%M%S}"
        remote_run = f"{self.remote_auto}/runs/{run_id}"
        self.ssh(f"mkdir -p {shlex.quote(remote_run)}")
        self.scp_to(env_path, f"{remote_run}/candidate.env")

        if self.benchmark_mode == "aligned_l1":
            # Fail before paying the multi-minute vLLM startup cost. The remote
            # runner repeats this check after SERVICE_READY to catch later drift.
            preflight = self.ssh(
                f"cd {shlex.quote(self.remote_project)} && "
                "set -a && "
                f"source {shlex.quote(remote_run + '/candidate.env')} && "
                "set +a && "
                f"python3 {shlex.quote(self.remote_auto + '/validate_aligned_l1_inputs.py')}",
                timeout=180,
            )
            (round_dir / "03_submission" / "benchmark_preflight.log").write_text(
                preflight + "\n", encoding="utf-8"
            )

        pointer_path = round_dir / "02_parameters" / "lab_active_run.env"
        pointer_path.write_text(
            f"EXPERIMENT_RUN_ID={shlex.quote(run_id)}\n",
            encoding="utf-8",
            newline="\n",
        )
        self.scp_to(pointer_path, f"{self.remote_auto}/lab_active_run.env")
        self.ssh(
            f"cp {shlex.quote(self.remote_auto + '/lease_loop.yaml')} "
            f"{shlex.quote(remote_run + '/task.yaml')}"
        )

        lease_name = str(self.lab["lease_name"])
        output_root = str(self.lab.get("output_root", "workflow/auto/lab_runs"))
        output = self.ssh(
            f"cd {shlex.quote(self.remote_project)} && "
            f"ktp-lab run --lease {shlex.quote(lease_name)} "
            f"--run-id {shlex.quote(run_id)} "
            f"--output-root {shlex.quote(output_root)}",
            timeout=180,
        )
        (round_dir / "03_submission" / "submit_output.txt").write_text(
            output + "\n", encoding="utf-8"
        )
        self.scp_from(f"{remote_run}/task.yaml", round_dir / "03_submission" / "task.yaml")
        save_json(
            round_dir / "03_submission" / "submission.json",
            {
                "execution_mode": "ktp_lab",
                "lease_name": lease_name,
                "run_id": run_id,
                "submitted_at": now(),
                "dry_run": False,
            },
        )
        if self.benchmark_mode == "aligned_l1":
            self.launch_benchmark_watchdog(run_id, lease_name)
        return lease_name, run_id

    def launch_benchmark_watchdog(self, run_id: str, lease_name: str) -> None:
        """Start a remote fallback that survives a local Controller outage."""
        remote_run = f"{self.remote_auto}/runs/{run_id}"
        watchdog_pid = remote_run + "/benchmark_watchdog.pid"
        watchdog_log = remote_run + "/benchmark_watchdog.log"
        detached = (
            f"echo $$ > {shlex.quote(watchdog_pid)}; "
            f"exec bash {shlex.quote(self.remote_auto + '/benchmark_watchdog.sh')} "
            f"{shlex.quote(run_id)} {shlex.quote(lease_name)} "
            f"> {shlex.quote(watchdog_log)} 2>&1 < /dev/null"
        )
        self.ssh(f"setsid -f bash -c {shlex.quote(detached)}", timeout=60)

    def validate_change_evidence(
        self,
        decision: dict[str, Any],
        change_count: int,
    ) -> None:
        strategy = decision.get("change_strategy")
        expected_strategy = (
            "single_parameter" if change_count == 1 else "coupled_parameters"
        )
        if strategy != expected_strategy:
            raise ValueError(
                f"{change_count} changes require change_strategy={expected_strategy}"
            )
        knowledge = decision.get("knowledge_evidence", decision.get("evidence", []))
        constraint_checks = decision.get("constraint_checks", [])
        interactions = decision.get("interaction_analysis", [])
        if len(knowledge) < change_count:
            raise ValueError("Each changed parameter requires concrete knowledge/log evidence")
        if len(constraint_checks) < change_count:
            raise ValueError("Each changed parameter requires an explicit constraint check")
        if change_count > 1 and len(interactions) < change_count - 1:
            raise ValueError(
                "A coupled proposal must explain the interactions between changes"
            )
        evidence_text = [*knowledge, *constraint_checks, *interactions]
        if any(len(str(item).strip()) < 12 for item in evidence_text):
            raise ValueError("Change evidence must be concrete, not a short assertion")

    @staticmethod
    def validate_no_change_metadata(decision: dict[str, Any]) -> None:
        if decision.get("change_strategy") != "none":
            raise ValueError("A no-change decision requires change_strategy=none")
        if decision.get("interaction_analysis") or decision.get("constraint_checks"):
            raise ValueError(
                "A no-change decision must not claim change interactions or checks"
            )

    @staticmethod
    def grid_step_order(grid: list[Any]) -> list[Any]:
        """Return semantic step order, independent of baseline-first display order."""
        if grid and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in grid
        ):
            return sorted(grid)
        return list(grid)

    @classmethod
    def grid_step_distance(
        cls,
        grid: list[Any],
        before: Any,
        after: Any,
    ) -> int:
        ordered = cls.grid_step_order(grid)
        return abs(ordered.index(after) - ordered.index(before))

    def effective_change_policy(
        self,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Resolve exploration/refinement limits for the next Agent decision."""
        phase = "exploration"
        adaptive = self.adaptive_change_policy
        if adaptive.get("enabled", False) and history and len(history) > 1:
            # Once any measured candidate enters a trustworthy improvement basin,
            # keep subsequent moves small. Failed later probes do not reopen the
            # broad exploration budget.
            for end in range(2, len(history) + 1):
                prefix = history[:end]
                if prefix[-1]["metrics"].get("benchmark_mode") == "aligned_l1":
                    assessment = self.assess_aligned_l1(prefix)
                    if assessment.get("eligible_as_improvement"):
                        phase = "refinement"
                        break

        phase_config = adaptive.get(phase, {}) if adaptive.get("enabled") else {}
        return {
            "strategy_version": self.change_policy.get(
                "strategy_version", "legacy"
            ),
            "phase": phase,
            "max_parameters_per_round": int(
                phase_config.get(
                    "max_parameters_per_round", self.max_parameters_per_round
                )
            ),
            "preferred_parameters_per_round": phase_config.get(
                "preferred_parameters_per_round",
                [1, self.max_parameters_per_round],
            ),
            "minimum_parameters_per_round": int(
                phase_config.get("minimum_parameters_per_round", 1)
            ),
            "max_grid_steps_per_parameter": int(
                phase_config.get(
                    "max_grid_steps_per_parameter",
                    self.max_grid_steps_per_parameter,
                )
            ),
            "max_total_grid_steps": int(
                phase_config.get(
                    "max_total_grid_steps", self.max_total_grid_steps
                )
            ),
            "derived_parameters": self.derived_parameter_rules,
            "parameter_groups": self.change_policy.get("parameter_groups", {}),
            "high_risk_parameters": self.change_policy.get(
                "high_risk_parameters", []
            ),
        }

    def derived_changes(
        self,
        actual_changes: list[str],
        policy: dict[str, Any],
    ) -> set[str]:
        changed = set(actual_changes)
        derived: set[str] = set()
        for parameter, rule in policy.get("derived_parameters", {}).items():
            drivers = rule.get("drivers", []) if isinstance(rule, dict) else []
            if parameter in changed and any(driver in changed for driver in drivers):
                derived.add(parameter)
        return derived

    def validate_candidate(
        self,
        previous: dict[str, Any],
        candidate: dict[str, Any],
        changes: list[dict[str, Any]],
        decision: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
    ) -> None:
        self.validate_candidate_invariants(candidate)
        actual_changes = [key for key in candidate if candidate[key] != previous[key]]
        effective_policy = policy or self.effective_change_policy()
        derived_changes = self.derived_changes(actual_changes, effective_policy)
        active_changes = [key for key in actual_changes if key not in derived_changes]
        min_parameters = int(effective_policy.get("minimum_parameters_per_round", 1))
        max_parameters = int(effective_policy["max_parameters_per_round"])
        if not min_parameters <= len(active_changes) <= max_parameters:
            raise ValueError(
                f"Each {effective_policy.get('phase', 'active')} round must change "
                f"between {min_parameters} and "
                f"{max_parameters} active parameters; active={active_changes}, "
                f"derived={sorted(derived_changes)}"
            )
        total_grid_steps = 0
        for key in actual_changes:
            grid = self.config["search_limits"][key]
            step_distance = self.grid_step_distance(
                grid,
                previous[key],
                candidate[key],
            )
            if key in derived_changes:
                continue
            max_parameter_steps = int(
                effective_policy["max_grid_steps_per_parameter"]
            )
            if step_distance > max_parameter_steps:
                raise ValueError(
                    f"{key} moves {step_distance} grid steps; maximum is "
                    f"{max_parameter_steps}"
                )
            total_grid_steps += step_distance
        max_total_steps = int(effective_policy["max_total_grid_steps"])
        if total_grid_steps > max_total_steps:
            raise ValueError(
                f"Combined grid-step distance {total_grid_steps} exceeds "
                f"{max_total_steps}"
            )
        declared_items: dict[str, dict[str, Any]] = {}
        for item in changes:
            key = item["parameter"].removeprefix("--").replace("-", "_")
            if key in declared_items:
                raise ValueError(f"Duplicate declared change for {key}")
            declared_items[key] = item
        declared = set(declared_items)
        if declared != set(actual_changes):
            raise ValueError(
                f"Declared changes {sorted(declared)} do not match actual {sorted(actual_changes)}"
            )
        for key in actual_changes:
            item = declared_items[key]
            if item.get("before") != previous[key] or item.get("after") != candidate[key]:
                raise ValueError(
                    f"Declared before/after for {key} does not match candidate values"
                )
            if len(str(item.get("rationale", "")).strip()) < 12:
                raise ValueError(f"Rationale for {key} is too short to audit")
        if decision is not None:
            self.validate_change_evidence(decision, len(actual_changes))

    def submit(
        self,
        round_dir: Path,
        label: str,
        candidate: dict[str, Any],
        *,
        dry_run: bool = False,
    ) -> tuple[str | None, str]:
        self.validate_deployment_configuration()
        rule_evaluation = self.runtime_rule_evaluation(candidate)
        if rule_evaluation is not None:
            save_yaml(
                round_dir / "02_parameters" / "runtime_rule_evaluation.yaml",
                rule_evaluation,
            )
            if not rule_evaluation["allowed"]:
                raise ValueError(
                    "Candidate failed deterministic runtime-rule preflight"
                )
        self.validate_runtime_configuration(candidate)
        self.sync_remote_scripts()
        if self.execution_mode == "ktp_lab":
            return self.submit_lab(
                round_dir,
                label,
                candidate,
                dry_run=dry_run,
            )
        env_path = round_dir / "02_parameters" / "candidate.env"
        env_path.write_text(self.candidate_env(label, candidate), encoding="utf-8", newline="\n")
        remote_candidates = f"{self.remote_auto}/candidates"
        self.ssh(f"mkdir -p {shlex.quote(remote_candidates)}")
        remote_env = f"{remote_candidates}/{label}_{int(time.time())}.env"
        self.scp_to(env_path, remote_env)
        mode = " --dry-run" if dry_run else ""
        output = self.ssh(
            f"bash {shlex.quote(self.remote_auto + '/submit_candidate.sh')} "
            f"{shlex.quote(remote_env)}{mode}",
            timeout=180,
        )
        (round_dir / "03_submission" / "submit_output.txt").write_text(
            output + "\n", encoding="utf-8"
        )
        run_match = re.search(r"EXPERIMENT_RUN_ID=(\S+)", output)
        if not run_match:
            raise RuntimeError("Submit output did not contain EXPERIMENT_RUN_ID")
        run_id = run_match.group(1)
        task_match = re.search(r"\bID:\s+(\d+)", output)
        task_id = task_match.group(1) if task_match and not dry_run else None
        if not dry_run and not task_id:
            raise RuntimeError("Submit output did not contain a platform task ID")
        remote_run = f"{self.remote_auto}/runs/{run_id}"
        self.scp_from(f"{remote_run}/task.yaml", round_dir / "03_submission" / "task.yaml")
        save_json(
            round_dir / "03_submission" / "submission.json",
            {"task_id": task_id, "run_id": run_id, "submitted_at": now(), "dry_run": dry_run},
        )
        return task_id, run_id

    def collect(self, run_id: str, round_dir: Path) -> dict[str, bool]:
        remote_run = f"{self.remote_auto}/runs/{run_id}"
        found: dict[str, bool] = {}
        for name in REMOTE_ARTIFACTS:
            if name in {"metrics.json"}:
                target = round_dir / "05_results" / name
            elif name in {"candidate.env", "task.yaml", "effective_config.yaml", "vllm_common_command.txt"}:
                target = round_dir / "02_parameters" / name
            else:
                target = round_dir / "04_runtime" / name
            exists = self.ssh(
                f"test -e {shlex.quote(remote_run + '/' + name)} && echo YES || echo NO"
            ).endswith("YES")
            found[name] = False
            if exists:
                found[name] = self.scp_from(f"{remote_run}/{name}", target)
        return found

    def start_aligned_benchmark(self, run_id: str, task_id: str | None) -> None:
        if self.benchmark_mode != "aligned_l1":
            return
        if not task_id or str(task_id).isdigit():
            raise RuntimeError(
                "aligned_l1 currently requires the persistent ktp_lab lease"
            )
        remote_run = f"{self.remote_auto}/runs/{run_id}"
        runner_pid = remote_run + "/benchmark_runner.pid"
        runner_log = remote_run + "/benchmark_runner.log"
        detached_runner = (
            f"echo $$ > {shlex.quote(runner_pid)}; "
            f"exec bash {shlex.quote(self.remote_auto + '/run_aligned_l1.sh')} "
            f"{shlex.quote(run_id)} {shlex.quote(str(task_id))} "
            f"> {shlex.quote(runner_log)} 2>&1 < /dev/null"
        )
        command = (
            f"cd {shlex.quote(self.remote_project)} && "
            f"if mkdir {shlex.quote(remote_run + '/BENCHMARK_START_LOCK')} "
            "2>/dev/null; then "
            f"touch {shlex.quote(remote_run + '/BENCHMARK_STARTED')} && "
            f"setsid -f bash -c {shlex.quote(detached_runner)}; "
            "fi"
        )
        self.ssh(command, timeout=60)
        log(f"Started aligned L1 benchmark for run={run_id}")

    def should_start_aligned_benchmark(
        self,
        found: dict[str, bool],
        task: dict[str, Any],
    ) -> bool:
        """Require both a fresh Ready marker and live remote processes."""
        active_pods = task.get("active_pods")
        return bool(
            self.benchmark_mode == "aligned_l1"
            and found.get("SERVICE_READY")
            and not found.get("BENCHMARK_STARTED")
            and not task.get("terminal")
            and isinstance(active_pods, int)
            and active_pods > 0
        )

    def task_snapshot(self, task_id: str | None) -> dict[str, Any]:
        if not task_id:
            return {
                "status": None,
                "active_pods": None,
                "terminal": False,
                "partial_failure": False,
            }
        if not str(task_id).isdigit():
            try:
                output = self.ssh(
                    f"cd {shlex.quote(self.remote_project)} && "
                    f"ktp-lab process list --lease {shlex.quote(str(task_id))} 2>&1"
                )
            except RuntimeError as exc:
                log(f"Unable to query lease {task_id}: {exc}")
                return {
                    "status": "lease_query_error",
                    "active_pods": None,
                    "terminal": False,
                    "partial_failure": False,
                    "raw": str(exc),
                }
            normalized = output.lower()
            status_lines = re.findall(
                r"^\s*status\s+(.+?)\s*$",
                normalized,
                flags=re.MULTILINE,
            )
            if status_lines:
                status_summary = " ".join(status_lines)
                active_count = sum(
                    int(value)
                    for value in re.findall(
                        r"\b(?:running|starting|pending)=(\d+)\b",
                        status_summary,
                    )
                )
                inactive_count = sum(
                    int(value)
                    for value in re.findall(
                        r"\b(?:idle|failed|error|exited|stopped|completed|succeeded)"
                        r"=(\d+)\b",
                        status_summary,
                    )
                )
                processes_running = active_count > 0
                partial_failure = active_count > 0 and inactive_count > 0
            else:
                # Compatibility with older ktp-lab output that did not emit a
                # dedicated status line. Exclude explanatory NOTE lines.
                status_summary = " ".join(
                    line
                    for line in normalized.splitlines()
                    if not line.lstrip().startswith("note")
                )
                processes_running = bool(
                    re.search(r"\b(?:running|starting|pending)\b", status_summary)
                )
                active_count = 2 if processes_running else 0
                inactive_count = 0
                partial_failure = False
            processes_terminal = (
                not processes_running
                and bool(
                    re.search(
                        r"\b(?:idle|failed|error|exited|stopped|completed|succeeded)"
                        r"(?:=\d+)?\b",
                        status_summary,
                    )
                )
            )
            return {
                # The lease itself stays alive; terminal refers only to the
                # declared processes for the current round.
                "status": (
                    "LeaseProcessesTerminal"
                    if processes_terminal
                    else (
                        "LeaseProcessesPartialFailure"
                        if partial_failure
                        else "LeaseActive"
                    )
                ),
                "active_pods": active_count,
                "terminal": processes_terminal,
                "partial_failure": partial_failure,
                "process_counts": {
                    "active": active_count,
                    "inactive": inactive_count,
                },
                "raw": output,
            }
        try:
            output = self.ssh(f"ktp get {shlex.quote(str(task_id))} 2>&1")
        except RuntimeError as exc:
            log(f"Unable to query task {task_id}: {exc}")
            return {
                "status": "query_error",
                "active_pods": None,
                "terminal": False,
                "partial_failure": False,
            }
        active = re.search(r"Active Pods:\s+(\d+)", output)
        status = re.search(
            r"Status:\s+(?:\x1b\[[0-9;]*m)?([A-Za-z]+)",
            output,
        )
        status_text = status.group(1) if status else None
        active_pods = int(active.group(1)) if active else None
        # Pending/queued KTP tasks legitimately report Active Pods: 0 before
        # scheduling. Only an explicit terminal status is authoritative.
        terminal = status_text in {
            "Succeeded",
            "Completed",
            "Terminated",
            "Failed",
        }
        return {
            "status": status_text,
            "active_pods": active_pods,
            "terminal": terminal,
            "partial_failure": False,
        }

    def round_timed_out(self, state: dict[str, Any]) -> bool:
        submitted_at = state.get("round_submitted_at")
        if not submitted_at:
            return False
        started = dt.datetime.fromisoformat(submitted_at)
        elapsed = dt.datetime.now().astimezone() - started
        return elapsed.total_seconds() >= self.round_timeout_minutes * 60

    def partial_exit_is_failure(
        self,
        state: dict[str, Any],
        task: dict[str, Any],
    ) -> bool:
        if not task.get("partial_failure"):
            return False
        submitted_at = state.get("round_submitted_at")
        if not submitted_at:
            return True
        started = dt.datetime.fromisoformat(submitted_at)
        elapsed = dt.datetime.now().astimezone() - started
        return elapsed.total_seconds() >= self.partial_exit_grace_seconds

    def stop_partial_lab_processes(self, task_id: str | None) -> None:
        if not task_id or str(task_id).isdigit():
            return
        log(
            f"Lease {task_id} has a partially exited process set; "
            "stopping the remaining managed process before retry."
        )
        self.ssh(
            f"cd {shlex.quote(self.remote_project)} && "
            f"ktp-lab stop --lease {shlex.quote(str(task_id))}",
            timeout=180,
        )

    @staticmethod
    def benchmark_regime(
        metrics: dict[str, Any],
        round_dir: Path | None = None,
    ) -> str:
        """Identify stable benchmark inputs without per-run target details."""
        mode = str(metrics.get("benchmark_mode", "legacy_or_unknown"))
        if mode != "aligned_l1":
            return f"mode:{mode}"
        if round_dir is not None:
            env_path = round_dir / "02_parameters" / "candidate.env"
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    if not line.startswith("BENCHMARK_EXPECTED_FINGERPRINT_JSON="):
                        continue
                    assignment = shlex.split(line, posix=True)[0]
                    frozen = json.loads(assignment.split("=", 1)[1])
                    stable = {
                        key: frozen.get(key)
                        for key in (
                            "servebench_version",
                            "servebench_commit",
                            "suite_file",
                            "suite_sha256",
                            "schema_files_sha256",
                            "tokenizer_files_sha256",
                            "dataset_manifests_sha256",
                        )
                    }
                    encoded = json.dumps(
                        stable,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    return "aligned_l1:frozen:" + hashlib.sha256(encoded).hexdigest()
            except (OSError, ValueError, IndexError, json.JSONDecodeError):
                pass
        suite = metrics.get("l1", {}).get("suite", "unknown")
        return f"aligned_l1:suite:{suite}"

    def history_summary(self, session_dir: Path) -> list[dict[str, Any]]:
        history = []
        for metrics_path in sorted(session_dir.glob("round_*/05_results/metrics.json")):
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                params_path = metrics_path.parents[1] / "02_parameters" / "candidate_params.yaml"
                params = load_yaml(params_path) if params_path.exists() else {}
                history.append(
                    {
                        "round": metrics_path.parents[1].name,
                        "params": params,
                        "metrics": metrics,
                        "benchmark_regime": self.benchmark_regime(
                            metrics,
                            metrics_path.parents[1],
                        ),
                    }
                )
            except (OSError, ValueError, yaml.YAMLError):
                continue
        if not history:
            return history
        latest_regime = history[-1]["benchmark_regime"]
        return [
            item for item in history
            if item["benchmark_regime"] == latest_regime
        ]

    def attempted_history_summary(self, session_dir: Path) -> list[dict[str, Any]]:
        """Return every terminal candidate, including failed/invalid attempts."""
        attempts: list[dict[str, Any]] = []
        for round_dir in sorted(session_dir.glob("round_*")):
            params_path = round_dir / "02_parameters" / "candidate_params.yaml"
            if not params_path.exists():
                continue
            metrics_path = round_dir / "05_results" / "metrics.json"
            failure_path = round_dir / "05_results" / "failure.yaml"
            if not metrics_path.exists() and not failure_path.exists():
                continue
            try:
                params = load_yaml(params_path)
                item: dict[str, Any] = {
                    "round": round_dir.name,
                    "params": params,
                    "outcome": "success" if metrics_path.exists() else "failed",
                }
                if metrics_path.exists():
                    item["metrics"] = json.loads(
                        metrics_path.read_text(encoding="utf-8")
                    )
                if failure_path.exists():
                    item["failure"] = load_yaml(failure_path)
                    failure_decision_path = (
                        round_dir / "06_agent_analysis" / "failure_decision.json"
                    )
                    if failure_decision_path.exists():
                        item["failure_decision"] = json.loads(
                            failure_decision_path.read_text(encoding="utf-8")
                        )
                attempts.append(item)
            except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
                continue
        return attempts

    @staticmethod
    def primary_performance_score(item: dict[str, Any]) -> float | None:
        """Return the comparable primary score for a successful history item."""
        metrics = item.get("metrics", {})
        if metrics.get("benchmark_mode") == "aligned_l1":
            value = metrics.get("l1", {}).get(
                "primary_aggregate_output_tps_geomean"
            )
        else:
            value = metrics.get("metrics", {}).get("output_token_throughput")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    def best_accepted_anchor(
        self,
        history: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Choose the best candidate accepted by the baseline-relative gate."""
        if not history:
            return None
        baseline = history[0]
        accepted = [baseline]
        for item in history[1:]:
            if item.get("metrics", {}).get("benchmark_mode") != "aligned_l1":
                continue
            assessment = self.assess_aligned_l1([baseline, item])
            if assessment.get("eligible_as_improvement"):
                accepted.append(item)
        scored = [
            (self.primary_performance_score(item), item)
            for item in accepted
        ]
        scored = [(score, item) for score, item in scored if score is not None]
        if not scored:
            return {
                "round": baseline.get("round"),
                "params": baseline.get("params", {}),
                "primary_score": None,
                "selection_reason": "baseline fallback; no comparable score",
            }
        score, best = max(scored, key=lambda pair: pair[0])
        return {
            "round": best.get("round"),
            "params": best.get("params", {}),
            "primary_score": score,
            "accepted_rounds": [item.get("round") for item in accepted],
            "selection_reason": (
                "highest primary score among baseline and candidates accepted "
                "by the deterministic baseline-relative measurement gate"
            ),
        }

    def exploration_memory(
        self,
        history: list[dict[str, Any]],
        attempted_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Summarize search coverage and measured effects without inventing causality."""
        anchor = self.best_accepted_anchor(history)
        baseline = history[0] if history else None
        baseline_params = baseline.get("params", {}) if baseline else {}
        baseline_score = self.primary_performance_score(baseline) if baseline else None
        policy = self.effective_change_policy(history)

        coverage: dict[str, Any] = {}
        for parameter, allowed_values in self.config["search_limits"].items():
            attempted_values: list[Any] = []
            for item in attempted_history:
                params = item.get("params", {})
                if parameter in params and params[parameter] not in attempted_values:
                    attempted_values.append(params[parameter])
            coverage[parameter] = {
                "attempted_values": attempted_values,
                "untested_values": [
                    value for value in allowed_values if value not in attempted_values
                ],
            }

        direct_effects: list[dict[str, Any]] = []
        measured_combinations: list[dict[str, Any]] = []
        if baseline and baseline_score not in (None, 0):
            for item in history[1:]:
                score = self.primary_performance_score(item)
                if score is None:
                    continue
                params = item.get("params", {})
                changed = [
                    key
                    for key in self.config["search_limits"]
                    if key in params
                    and key in baseline_params
                    and params[key] != baseline_params[key]
                ]
                derived = self.derived_changes(changed, policy)
                independent = [key for key in changed if key not in derived]
                observation = {
                    "round": item.get("round"),
                    "independent_changes_vs_baseline": {
                        key: {
                            "baseline": baseline_params[key],
                            "tested": params[key],
                        }
                        for key in independent
                    },
                    "derived_changes_vs_baseline": sorted(derived),
                    "primary_score": score,
                    "gain_vs_baseline_percent": (
                        score / baseline_score - 1.0
                    ) * 100.0,
                }
                if len(independent) == 1:
                    direct_effects.append(observation)
                elif independent:
                    measured_combinations.append(observation)

        return {
            "best_accepted_anchor": anchor,
            "coverage_by_parameter": coverage,
            "direct_single_parameter_effects": direct_effects,
            "measured_multi_parameter_combinations": measured_combinations,
            "interpretation": {
                "direct_effects": (
                    "May guide value-level prioritization because exactly one "
                    "independent parameter differed from baseline."
                ),
                "multi_parameter_combinations": (
                    "May down-rank repeating the complete combination, but must "
                    "not be treated as proof that each individual value is bad."
                ),
            },
        }

    def successful_candidate(
        self,
        session_dir: Path,
        candidate: dict[str, Any],
    ) -> dict[str, Any] | None:
        for item in self.history_summary(session_dir):
            if item.get("params") == candidate:
                return item
        return None

    def candidate_was_attempted(
        self,
        session_dir: Path,
        candidate: dict[str, Any],
    ) -> bool:
        return any(
            item.get("params") == candidate
            for item in self.attempted_history_summary(session_dir)
        )

    def write_comparison(self, session_dir: Path, round_dir: Path) -> dict[str, Any]:
        history = self.history_summary(session_dir)
        payload: dict[str, Any] = {
            "generated_at": now(),
            "current_round": round_dir.name,
            "baseline_round": history[0]["round"] if history else None,
            "previous_round": history[-2]["round"] if len(history) > 1 else None,
            "numeric_metric_deltas_vs_previous": {},
            "numeric_metric_deltas_vs_baseline": {},
        }
        if not history:
            save_json(round_dir / "05_results" / "comparison.json", payload)
            return payload

        current_metrics = history[-1]["metrics"].get("metrics", {})

        def deltas(reference: dict[str, Any]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, current_value in current_metrics.items():
                reference_value = reference.get(key)
                if (
                    isinstance(current_value, (int, float))
                    and not isinstance(current_value, bool)
                    and isinstance(reference_value, (int, float))
                    and not isinstance(reference_value, bool)
                ):
                    absolute = current_value - reference_value
                    percent = (
                        absolute / reference_value * 100
                        if reference_value != 0
                        else None
                    )
                    result[key] = {
                        "current": current_value,
                        "reference": reference_value,
                        "absolute": absolute,
                        "percent": percent,
                    }
            return result

        baseline_metrics = history[0]["metrics"].get("metrics", {})
        previous_metrics = history[-2]["metrics"].get("metrics", {}) if len(history) > 1 else baseline_metrics
        payload["numeric_metric_deltas_vs_previous"] = deltas(previous_metrics)
        payload["numeric_metric_deltas_vs_baseline"] = deltas(baseline_metrics)
        save_json(round_dir / "05_results" / "comparison.json", payload)
        return payload

    def reconcile_official_source_default_baseline(
        self,
        session_dir: Path,
        round_dir: Path,
        state: dict[str, Any],
    ) -> None:
        """Replace B0 bookkeeping estimates with values resolved by vLLM.

        B0 intentionally omits serving-performance flags.  The Agent must not
        receive the typed pre-launch estimates as if they were observed values,
        so the first successful round is reconciled from the engine log before
        analysis or candidate generation can begin.
        """
        initial = self.config.get("initial_baseline", {})
        if (
            state.get("round_label") != str(initial.get("label", "a0"))
            or initial.get("launch_profile") != "official_source_defaults"
            or state.get("official_source_defaults_reconciled")
        ):
            return

        log_path = round_dir / "04_runtime" / "master.log"
        if not log_path.is_file():
            raise RuntimeError("B0 completed without master.log; defaults cannot be audited")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        resolved = dict(state["current_candidate"])
        evidence: dict[str, Any] = {
            "launch_profile": "official_source_defaults",
            "authoritative_log": str(log_path),
            "source_defaults": {
                "max_num_seqs": 256,
                "gpu_memory_utilization": 0.92,
                "async_scheduling": False,
                "enable_expert_parallel": False,
                "num_speculative_tokens": 0,
                "long_prefill_token_threshold": 0,
                "enable_eplb": False,
                "flashcomm1": False,
                "mlapo": True,
                "fused_mc2": 0,
            },
            "log_resolved": {},
        }

        def required_int(name: str, pattern: str) -> int:
            match = re.search(pattern, log_text)
            if not match:
                raise RuntimeError(
                    f"B0 master.log does not expose resolved {name}; refusing Agent handoff"
                )
            value = int(match.group(1))
            evidence["log_resolved"][name] = value
            return value

        resolved["max_model_len"] = required_int(
            "max_model_len", r"\bmax_seq_len=(\d+)"
        )
        resolved["max_num_batched_tokens"] = required_int(
            "max_num_batched_tokens", r"max_num_batched_tokens=(\d+)"
        )
        resolved["max_num_seqs"] = 256
        resolved["gpu_memory_utilization"] = 0.92
        resolved["async_scheduling"] = "Asynchronous scheduling is enabled" in log_text
        resolved["enable_expert_parallel"] = False
        resolved["num_speculative_tokens"] = 0
        resolved["long_prefill_token_threshold"] = 0
        resolved["enable_eplb"] = False
        resolved["eplb_num_redundant_experts"] = 0
        resolved["compilation_enable_sp"] = False
        resolved["flashcomm1"] = False
        if "mlapo" in resolved:
            resolved["mlapo"] = True
        if "fused_mc2" in resolved:
            resolved["fused_mc2"] = 0

        for field, pattern in (
            ("enable_prefix_caching", r"enable_prefix_caching=(True|False)"),
            ("enable_chunked_prefill", r"enable_chunked_prefill=(True|False)"),
        ):
            match = re.search(pattern, log_text)
            if not match:
                raise RuntimeError(
                    f"B0 master.log does not expose resolved {field}; refusing Agent handoff"
                )
            resolved[field] = match.group(1) == "True"
            evidence["log_resolved"][field] = resolved[field]

        mode = re.search(r"'cudagraph_mode': <CUDAGraphMode\.([A-Z_]+)", log_text)
        maximum = re.search(r"'max_cudagraph_capture_size': (\d+)", log_text)
        sizes = re.search(r"'cudagraph_capture_sizes': (\[[^\]]*\]|None)", log_text)
        if not (mode and maximum and sizes):
            raise RuntimeError(
                "B0 master.log does not expose resolved compilation defaults; "
                "refusing Agent handoff"
            )
        resolved["compilation_mode"] = mode.group(1)
        resolved["max_cudagraph_capture_size"] = int(maximum.group(1))
        resolved["cudagraph_capture_sizes"] = (
            None if sizes.group(1) == "None" else ast.literal_eval(sizes.group(1))
        )
        evidence["log_resolved"].update(
            compilation_mode=resolved["compilation_mode"],
            max_cudagraph_capture_size=resolved["max_cudagraph_capture_size"],
            cudagraph_capture_sizes=resolved["cudagraph_capture_sizes"],
        )

        for name, value in resolved.items():
            limits = self.config["search_limits"][name]
            if value not in limits:
                self.config["search_limits"][name] = [value, *limits]

        save_yaml(round_dir / "02_parameters" / "candidate_params.yaml", resolved)
        save_yaml(round_dir / "02_parameters" / "b0_effective_resolution.yaml", evidence)
        effective_path = round_dir / "02_parameters" / "effective_config.yaml"
        effective = load_yaml(effective_path) if effective_path.is_file() else {}
        effective["launch_profile"] = "official_source_defaults"
        effective["authoritative_effective_service"] = resolved
        effective["authoritative_source"] = "master.log plus pinned-source defaults"
        save_yaml(effective_path, effective)
        save_yaml(session_dir / "session_config.yaml", self.config)
        state["current_candidate"] = resolved
        state["official_source_defaults_reconciled"] = True
        state["official_source_defaults_evidence"] = str(
            round_dir / "02_parameters" / "b0_effective_resolution.yaml"
        )

    def assess_measurement(
        self,
        history: list[dict[str, Any]],
        comparison: dict[str, Any],
    ) -> dict[str, Any]:
        if (
            history
            and history[-1]["metrics"].get("benchmark_mode") == "aligned_l1"
        ):
            return self.assess_aligned_l1(history)
        policy = self.measurement_policy
        if not history:
            return {"classification": "missing_metrics", "passes_guardrails": False}
        current = history[-1]["metrics"].get("metrics", {})
        successful = int(current.get("successful_requests", 0) or 0)
        failed = int(current.get("failed_requests", 0) or 0)
        required_successes = int(
            policy.get(
                "minimum_successful_requests",
                self.config["fixed_scenario"]["num_prompts"],
            )
        )
        passes_guardrails = successful >= required_successes
        if policy.get("require_zero_failed_requests", True):
            passes_guardrails = passes_guardrails and failed == 0
        assessment: dict[str, Any] = {
            "classification": "baseline_only" if len(history) == 1 else "candidate",
            "passes_guardrails": passes_guardrails,
            "successful_requests": successful,
            "failed_requests": failed,
            "policy": policy,
            "eligible_as_improvement": False,
        }
        if len(history) == 1:
            return assessment
        deltas = comparison.get("numeric_metric_deltas_vs_baseline", {})
        throughput_gain = deltas.get("output_token_throughput", {}).get("percent")
        ttft_change = deltas.get("mean_ttft", {}).get("percent")
        tpot_change = deltas.get("mean_tpot", {}).get("percent")
        assessment.update(
            throughput_gain_percent=throughput_gain,
            mean_ttft_change_percent=ttft_change,
            mean_tpot_change_percent=tpot_change,
        )
        if any(value is None for value in (throughput_gain, ttft_change, tpot_change)):
            assessment["classification"] = "insufficient_comparison"
            return assessment
        assessment["eligible_as_improvement"] = bool(
            passes_guardrails
            and throughput_gain
            >= float(policy.get("minimum_throughput_gain_percent", 3.0))
            and ttft_change
            <= float(policy.get("maximum_ttft_regression_percent", 10.0))
            and tpot_change
            <= float(policy.get("maximum_tpot_regression_percent", 10.0))
        )
        return assessment

    def assess_aligned_l1(
        self,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        policy = self.measurement_policy.get("aligned_l1", {})
        current_payload = history[-1]["metrics"]
        current_l1 = current_payload.get("l1", {})
        repetitions = int(current_l1.get("repetition_count", 0) or 0)
        minimum_repetitions = int(policy.get("minimum_repetitions", 3))
        absolute_gate = bool(current_l1.get("all_repetitions_gate_passed"))
        if policy.get("require_all_repetitions", True):
            absolute_gate = absolute_gate and repetitions >= minimum_repetitions
        assessment: dict[str, Any] = {
            "benchmark_mode": "aligned_l1",
            "classification": "baseline_only" if len(history) == 1 else "candidate",
            "passes_guardrails": absolute_gate,
            "eligible_as_improvement": False,
            "repetition_count": repetitions,
            "policy": policy,
            "violations": [],
        }
        if not absolute_gate:
            assessment["classification"] = "absolute_gate_failed"
            assessment["violations"].append(
                "not all required L1 repetitions passed the 12-case evidence gate"
            )
            return assessment
        if len(history) == 1:
            return assessment

        baseline_payload = history[0]["metrics"]
        baseline_l1 = baseline_payload.get("l1", {})
        if not baseline_l1.get("all_repetitions_gate_passed"):
            assessment["classification"] = "invalid_baseline"
            assessment["violations"].append("baseline did not pass the L1 evidence gate")
            return assessment

        baseline_score = baseline_l1.get(
            "primary_aggregate_output_tps_geomean"
        )
        current_score = current_l1.get(
            "primary_aggregate_output_tps_geomean"
        )
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
            for value in (baseline_score, current_score)
        ):
            assessment["classification"] = "insufficient_comparison"
            assessment["violations"].append("primary L1 score is missing or invalid")
            return assessment

        baseline_cases = {
            (case.get("workload"), case.get("concurrency")): case
            for case in baseline_l1.get("cases", [])
            if isinstance(case, dict)
        }
        current_cases = {
            (case.get("workload"), case.get("concurrency")): case
            for case in current_l1.get("cases", [])
            if isinstance(case, dict)
        }
        if set(baseline_cases) != set(current_cases) or not current_cases:
            assessment["classification"] = "insufficient_comparison"
            assessment["violations"].append(
                "candidate and baseline L1 case matrices differ"
            )
            return assessment

        latency_limits = {
            "ttft_p50_ms": float(policy.get("maximum_ttft_p50_ratio", 1.20)),
            "ttft_p90_ms": float(policy.get("maximum_ttft_p90_ratio", 1.20)),
            "tpot_p50_ms": float(policy.get("maximum_tpot_p50_ratio", 1.15)),
            "tpot_p90_ms": float(policy.get("maximum_tpot_p90_ratio", 1.15)),
        }
        latency_ratios: dict[str, dict[str, float]] = {}
        primary_concurrency = int(current_l1.get("primary_concurrency", 32))
        throughput_floor = 1.0 - float(
            policy.get(
                "maximum_single_workload_throughput_regression_percent", 5.0
            )
        ) / 100.0
        workload_throughput_ratios: dict[str, float] = {}
        for key, current_case in current_cases.items():
            baseline_case = baseline_cases[key]
            case_name = f"{key[0]}@c{key[1]}"
            latency_ratios[case_name] = {}
            for metric, limit in latency_limits.items():
                current_value = float(current_case[metric])
                baseline_value = float(baseline_case[metric])
                ratio = current_value / baseline_value if baseline_value > 0 else math.inf
                latency_ratios[case_name][metric] = ratio
                if ratio > limit:
                    assessment["violations"].append(
                        f"{case_name} {metric} ratio={ratio:.4f} > {limit:.4f}"
                    )
            if key[1] == primary_concurrency:
                current_tps = float(current_case["aggregate_output_tps"])
                baseline_tps = float(baseline_case["aggregate_output_tps"])
                ratio = current_tps / baseline_tps if baseline_tps > 0 else 0.0
                workload_throughput_ratios[str(key[0])] = ratio
                if ratio < throughput_floor:
                    assessment["violations"].append(
                        f"{case_name} aggregate output TPS ratio={ratio:.4f} "
                        f"< {throughput_floor:.4f}"
                    )

        gain_percent = (float(current_score) / float(baseline_score) - 1.0) * 100
        baseline_cv = float(baseline_l1.get("primary_score_cv_percent", 0.0) or 0.0)
        current_cv = float(current_l1.get("primary_score_cv_percent", 0.0) or 0.0)
        noise_floor = float(policy.get("noise_cv_multiplier", 2.0)) * max(
            baseline_cv, current_cv
        )
        required_gain = max(
            float(policy.get("minimum_throughput_gain_percent", 3.0)),
            noise_floor,
        )
        if gain_percent < required_gain:
            assessment["violations"].append(
                f"primary throughput gain={gain_percent:.4f}% "
                f"< required={required_gain:.4f}%"
            )
        assessment.update(
            primary_score=float(current_score),
            baseline_primary_score=float(baseline_score),
            throughput_gain_percent=gain_percent,
            baseline_primary_score_cv_percent=baseline_cv,
            candidate_primary_score_cv_percent=current_cv,
            noise_adjusted_required_gain_percent=required_gain,
            latency_ratios=latency_ratios,
            workload_throughput_ratios=workload_throughput_ratios,
        )
        assessment["passes_guardrails"] = not assessment["violations"]
        assessment["eligible_as_improvement"] = not assessment["violations"]
        return assessment

    def wait_for_task_release(self, task_id: str | None) -> None:
        if not task_id:
            return
        if not str(task_id).isdigit():
            # MASTER_DONE is written immediately before the master exits. Wait
            # for both declared processes to release their lease slots, while
            # deliberately keeping the lease itself allocated.
            for _ in range(12):
                snapshot = self.task_snapshot(task_id)
                if snapshot.get("terminal") or snapshot.get("active_pods") == 0:
                    return
                time.sleep(5)
            raise RuntimeError(
                f"Lease {task_id} still reports active processes; refusing overlap"
            )
        for _ in range(60):
            snapshot = self.task_snapshot(task_id)
            if snapshot["terminal"]:
                return
            time.sleep(10)
        log(f"Task {task_id} did not release after 10 minutes; requesting stop.")
        self.ssh(f"ktp stop {shlex.quote(str(task_id))}")
        for _ in range(12):
            if self.task_snapshot(task_id)["terminal"]:
                return
            time.sleep(10)
        raise RuntimeError(f"Task {task_id} still has active Pods; refusing overlap")

    @staticmethod
    def evidence_text(path: Path, *, max_chars: int = 60000) -> str:
        if not path.exists():
            return "[missing]"
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) <= max_chars:
            return text
        head = text[: max_chars // 3]
        tail = text[-(max_chars - len(head)) :]
        return head + "\n[...truncated by controller...]\n" + tail

    def build_analysis_evidence(
        self,
        round_dir: Path,
        history: list[dict[str, Any]],
        previous: dict[str, Any],
        attempted_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        comparison = json.loads(
            (round_dir / "05_results" / "comparison.json").read_text(encoding="utf-8")
        )
        evidence = {
            "current_candidate": previous,
            "scenario": load_yaml(round_dir / "00_context" / "scenario.yaml"),
            "image_identity": load_yaml(
                round_dir / "00_context" / "image_version_manifest.yaml"
            ),
            "search_space": self.evidence_text(
                round_dir / "01_query" / "glm5.2_search_space.yaml"
            ),
            "submitted_task": self.evidence_text(
                round_dir / "02_parameters" / "task.yaml"
            ),
            "effective_config": load_yaml(
                round_dir / "02_parameters" / "effective_config.yaml"
            ),
            "effective_command": self.evidence_text(
                round_dir / "02_parameters" / "vllm_common_command.txt"
            ),
            "formal_log": self.evidence_text(
                round_dir / "04_runtime" / "formal.log",
                max_chars=30000,
            ),
            "master_log": self.evidence_text(
                round_dir / "04_runtime" / "master.log",
                max_chars=30000,
            ),
            "worker_log": self.evidence_text(
                round_dir / "04_runtime" / "worker.log",
                max_chars=30000,
            ),
            "metrics": json.loads(
                (round_dir / "05_results" / "metrics.json").read_text(encoding="utf-8")
            ),
            "comparison": comparison,
            "measurement_assessment": self.assess_measurement(history, comparison),
            "history": history,
            "attempted_history": attempted_history or history,
            "exploration_memory": self.exploration_memory(
                history,
                attempted_history or history,
            ),
            "tag_knowledge": self.evidence_text(
                KB_ROOT / "SKILL-with-tag.md",
                max_chars=50000,
            ),
            "parameter_portraits": self.evidence_text(
                round_dir.parent
                / "00_search_space"
                / "parameter_portraits.agent.yaml",
                max_chars=120000,
            ),
            "runtime_rules": (
                {
                    "rules": self.runtime_rule_store.data.get("rules", []),
                    "quarantines": self.runtime_rule_store.data.get(
                        "quarantines", []
                    ),
                    "proposals": self.runtime_rule_store.data.get(
                        "proposals", []
                    ),
                }
                if self.runtime_rule_store
                else None
            ),
        }
        save_json(round_dir / "06_agent_analysis" / "evidence_bundle.json", evidence)
        return evidence

    def run_agent_decision_with_reselection(
        self,
        session_dir: Path,
        round_dir: Path,
        base_prompt: str,
        validate_decision: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        """Run Agent selection and retry deterministic candidate rejections."""
        analysis_dir = round_dir / "06_agent_analysis"
        decision_path = analysis_dir / "decision.json"
        rejections: list[dict[str, Any]] = []
        total_attempts = self.max_candidate_reselections + 1

        for attempt in range(1, total_attempts + 1):
            rejection_feedback = ""
            if rejections:
                rejection_feedback = f"""

Deterministic controller rejection feedback from earlier attempts in this same
selection cycle follows. Do not repeat any rejected complete candidate. Correct
the exact violations while preserving the measured evidence and Search Limits:
{json.dumps(rejections, ensure_ascii=False, indent=2)}
"""
            prompt = base_prompt + rejection_feedback
            (analysis_dir / "agent_prompt.md").write_text(prompt, encoding="utf-8")
            (analysis_dir / f"agent_prompt.attempt_{attempt:02d}.md").write_text(
                prompt,
                encoding="utf-8",
            )
            result = run_structured_agent(
                self.agent_config,
                prompt=prompt,
                schema_path=self.decision_schema_path(
                    session_dir, "agent_decision.schema.json"
                ),
                output_path=decision_path,
                cwd=KB_ROOT,
                allowed_dir=round_dir,
            )
            events_path = analysis_dir / f"agent_events.attempt_{attempt:02d}.jsonl"
            stderr_path = analysis_dir / f"agent_stderr.attempt_{attempt:02d}.log"
            events_path.write_text(result.stdout, encoding="utf-8")
            stderr_path.write_text(result.stderr, encoding="utf-8")
            (analysis_dir / "agent_events.jsonl").write_text(
                result.stdout,
                encoding="utf-8",
            )
            (analysis_dir / "agent_stderr.log").write_text(
                result.stderr,
                encoding="utf-8",
            )
            if result.returncode != 0 or not decision_path.exists():
                raise RuntimeError(
                    f"{result.provider} Agent analysis failed: {result.stderr[-2000:]}"
                )

            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            try:
                validate_decision(decision)
                save_json(
                    analysis_dir / "candidate_reselection_audit.json",
                    {
                        "completed_at": now(),
                        "attempts": attempt,
                        "allowed_reselections": self.max_candidate_reselections,
                        "rejections": rejections,
                        "final_status": "accepted",
                    },
                )
                return decision
            except (KeyError, ValueError) as exc:
                rejection = {
                    "attempt": attempt,
                    "rejected_at": now(),
                    "reason": f"{type(exc).__name__}: {exc}",
                    "candidate": decision.get("candidate"),
                    "changes": decision.get("changes"),
                }
                rejections.append(rejection)
                save_json(
                    analysis_dir / f"decision.rejected_{attempt:02d}.json",
                    {
                        **rejection,
                        "decision": decision,
                    },
                )
                save_json(
                    analysis_dir / "candidate_reselection_audit.json",
                    {
                        "updated_at": now(),
                        "attempts": attempt,
                        "allowed_reselections": self.max_candidate_reselections,
                        "rejections": rejections,
                        "final_status": (
                            "reselecting" if attempt < total_attempts else "paused"
                        ),
                    },
                )
                log(
                    f"Agent candidate rejected by deterministic validation "
                    f"(attempt {attempt}/{total_attempts}): {exc}"
                )
                if attempt == total_attempts:
                    reasons = "; ".join(item["reason"] for item in rejections)
                    raise RepeatedCandidateRejection(
                        f"Agent produced {total_attempts} consecutively rejected "
                        f"candidates: {reasons}"
                    ) from exc

        raise AssertionError("unreachable candidate reselection state")

    def analyze(
        self,
        session_dir: Path,
        round_dir: Path,
        previous: dict[str, Any],
    ) -> dict[str, Any]:
        history = self.history_summary(session_dir)
        attempted_history = self.attempted_history_summary(session_dir)
        self.write_comparison(session_dir, round_dir)
        save_json(
            round_dir / "06_agent_analysis" / "history_input.json",
            attempted_history,
        )
        self.refresh_runtime_rules(session_dir, round_dir)
        evidence = self.build_analysis_evidence(
            round_dir,
            history,
            previous,
            attempted_history,
        )
        selection_policy = self.effective_change_policy(history)
        evidence["selection_policy"] = selection_policy
        save_json(
            round_dir / "06_agent_analysis" / "evidence_bundle.json",
            evidence,
        )
        if self.benchmark_mode == "aligned_l1":
            repetition_count = int(
                self.benchmark["aligned_l1"]["repetitions"]
            )
            repetition_aggregation = (
                "the single complete repetition"
                if repetition_count == 1
                else f"the median across {repetition_count} complete repetitions"
            )
            benchmark_goal = f"""Goal: improve the strict aggregate output-token throughput score for the
frozen ServeBench tuning-fixed v3 L1 matrix. The matrix has four workloads
(1024/256, 8192/512, 1024/1024, 256/2048), fixed C1/C16/C32 concurrency,
fixed JSONL prompts, temperature=0, and {repetition_count} complete repetition(s).
The primary score is {repetition_aggregation} of the C32 workload geometric mean.
TTFT/TPOT P50/P90, zero errors/incomplete requests, exact token shapes,
per-workload throughput, and run-to-run CV are deterministic guardrails."""
        else:
            benchmark_goal = """Goal: improve measured output throughput under the historical
random 32K-centered input / 1K-centered output / 8 prompts / 0.2 req/s /
temperature=0 workload. TTFT, TPOT, success rate, and memory are guardrails.
Treat this small legacy measurement as exploratory evidence."""
        prompt = f"""You are the Codex tuning analyst for a measured GLM-5.2 vLLM-Ascend experiment.

The controller has embedded all required read-only evidence below. Treat it as
authoritative. If additional inspection is useful, you may execute local read-only
commands to read the listed files. Never write or edit files, run remote commands,
access the network, submit jobs, stop jobs, or change external state.

{benchmark_goal}
Use measurement_assessment as the deterministic acceptance gate. A candidate is not
an improvement unless eligible_as_improvement=true. Use exploration_memory as the
cross-round decision memory. Anchor new proposals on best_accepted_anchor instead
of blindly continuing from a rejected branch; because the submitted candidate is
expressed relative to the exact current candidate, declare any required rollback
changes explicitly. Prefer informative untested values of high-impact active
parameters. Down-rank a directly measured negative single-parameter direction, and
do not repeat a measured negative multi-parameter combination unless a concrete new
interaction hypothesis justifies it. Never infer that each value in a confounded
multi-parameter result is independently harmful. Use the smallest defensible change
set.

Return only JSON matching the supplied schema. Set action=continue when another
evidence-based experiment is warranted. Preserve the complete current candidate.
The frozen Agent strategy profile is {self.strategy_profile_name}:
{yaml.safe_dump(self.strategy_profile, allow_unicode=True, sort_keys=False)}
The active selection phase and limits are:
{yaml.safe_dump(selection_policy, allow_unicode=True, sort_keys=False)}
In exploration, prefer the configured 2-3 active parameters when evidence supports
a coherent faster probe; in refinement, prefer 1-2 active parameters. A derived
parameter changed together with one of its declared drivers does not consume an
active-parameter slot or grid-step budget, but it must still be declared in changes
and satisfy every hard invariant. Treat listed high-risk parameters conservatively:
normally change only one high-risk active parameter in a round, and combine it only
when a documented prerequisite or inseparable interaction requires the combination.
Use multiple parameters only when knowledge, runtime
constraints, or a proven interaction makes a joint experiment materially more valid
than independent trials. Do not bundle unrelated guesses. For multiple changes set
change_strategy=coupled_parameters, explain at least one interaction per additional
parameter, and give explicit constraint checks for every changed parameter.
For numeric parameters, grid distance is computed in ascending numeric order;
the whitelist below may put the current baseline first for display and must not
be interpreted as step order. The effective grid-step order is:
{yaml.safe_dump({key: self.grid_step_order(values) for key, values in self.config['search_limits'].items()}, allow_unicode=True, sort_keys=False)}
Set action=stop_complete, preserve the current candidate, and return an
empty changes array, change_strategy=none, and empty interaction_analysis and
constraint_checks when the useful whitelist search space is exhausted, no safe
untested change remains, or further testing is not justified by the measurements.
Use only values allowed by this whitelist:
{yaml.safe_dump(self.config['search_limits'], allow_unicode=True, sort_keys=False)}

Do not change model, DP=2, TP=16, Pod/NPU topology, network, ports, image, benchmark,
quantization, or fixed Ascend environment.
Do not repeat any successful or failed configuration already present in
attempted_history/history_input.json. Treat candidates classified parameter_invalid
or parameter_oom as excluded combinations. Cite concrete metric evidence and
knowledge-base constraints. Before changing any parameter, read its entry in
parameter_portraits, inspect related_parameters, and cite the applicable natural-language
constraints in knowledge_evidence and constraint_checks. Never claim improvement
without data.
For every declared change, before and after must exactly match the current and proposed
candidate values.

The exact current candidate is:
{json.dumps(previous, ensure_ascii=False, indent=2)}

Embedded evidence:
{json.dumps(evidence, ensure_ascii=False)}
"""
        def validate_decision(decision: dict[str, Any]) -> None:
            action = decision["action"]
            if action == "stop_complete":
                if decision["candidate"] != previous or decision["changes"]:
                    raise ValueError(
                        "stop_complete must preserve the current candidate and have no changes"
                    )
                self.validate_no_change_metadata(decision)
                return
            self.validate_candidate(
                previous,
                decision["candidate"],
                decision["changes"],
                decision,
                selection_policy,
            )
            if self.candidate_was_attempted(session_dir, decision["candidate"]):
                raise ValueError(
                    "Codex proposed a configuration already present in experiment history"
                )

        decision = self.run_agent_decision_with_reselection(
            session_dir,
            round_dir,
            prompt,
            validate_decision,
        )
        if decision["action"] == "stop_complete":
            return decision
        self.write_selected_portrait_evidence(
            round_dir,
            decision["changes"],
        )
        save_yaml(
            round_dir / "06_agent_analysis" / "next_candidate.yaml",
            {
                "generated_at": now(),
                "based_on_round": round_dir.name,
                "changes": decision["changes"],
                "candidate": decision["candidate"],
            },
        )
        return decision

    def analyze_failure(
        self,
        session_dir: Path,
        round_dir: Path,
        current: dict[str, Any],
    ) -> dict[str, Any]:
        history = self.attempted_history_summary(session_dir)
        save_json(round_dir / "06_agent_analysis" / "history_input.json", history)
        failure_evidence = {
            "current_candidate": current,
            "history": history,
            "scenario": self.evidence_text(round_dir / "00_context" / "scenario.yaml"),
            "search_space": self.evidence_text(
                round_dir / "01_query" / "glm5.2_search_space.yaml"
            ),
            "effective_config": self.evidence_text(
                round_dir / "02_parameters" / "effective_config.yaml"
            ),
            "run_status": self.evidence_text(
                round_dir / "04_runtime" / "run_status.json"
            ),
            "master_log": self.evidence_text(
                round_dir / "04_runtime" / "master.log", max_chars=50000
            ),
            "worker_log": self.evidence_text(
                round_dir / "04_runtime" / "worker.log", max_chars=50000
            ),
            "warmup_log": self.evidence_text(
                round_dir / "04_runtime" / "warmup.log", max_chars=30000
            ),
            "formal_log": self.evidence_text(
                round_dir / "04_runtime" / "formal.log", max_chars=30000
            ),
            "benchmark_runner_log": self.evidence_text(
                round_dir / "04_runtime" / "benchmark_runner.log",
                max_chars=30000,
            ),
            "parameter_portraits": self.evidence_text(
                session_dir
                / "00_search_space"
                / "parameter_portraits.agent.yaml",
                max_chars=120000,
            ),
            "runtime_rules": (
                {
                    "rules": self.runtime_rule_store.data.get("rules", []),
                    "quarantines": self.runtime_rule_store.data.get(
                        "quarantines", []
                    ),
                    "proposals": self.runtime_rule_store.data.get(
                        "proposals", []
                    ),
                }
                if self.runtime_rule_store
                else None
            ),
        }
        save_json(
            round_dir / "06_agent_analysis" / "failure_evidence_bundle.json",
            failure_evidence,
        )
        prompt = f"""You are the Codex failure analyst for a GLM-5.2 vLLM-Ascend experiment
that ended without metrics.json.

All required evidence is embedded below. You may execute local read-only commands for
additional inspection. Never write or edit files, run remote commands, access the
network, submit jobs, stop jobs, or change external state.

Classify the root cause using only log evidence:
- parameter_invalid or parameter_oom: a launch parameter is invalid or causes resource failure.
- transient_infrastructure/network_or_hccl: platform, Pod, network, HCCL, timeout,
  or another transient condition; normally retry the identical candidate.
- image_or_dependency/model_or_runtime_bug/benchmark_failure: not safely repairable by
  changing the tuning whitelist; pause for human unless the evidence clearly proves a
  safe parameter correction.
- unknown: pause for human.

For parameter_invalid/parameter_oom, choose action=adjust_parameters and use the
smallest directly corrective change set, up to {self.max_parameters_per_round}
parameters. Multiple changes are allowed only when the logs prove that the correction
is coupled. Explain the interaction and give an explicit constraint check for every
changed parameter. Grid-step budgets are:
{yaml.safe_dump(self.change_policy, allow_unicode=True, sort_keys=False)}
For retry_same or pause_for_human, return the current candidate unchanged, an empty
changes array, change_strategy=none, and empty interaction_analysis and
constraint_checks.

Allowed values:
{yaml.safe_dump(self.config['search_limits'], allow_unicode=True, sort_keys=False)}

Never modify topology, model, image, network, paths, benchmark, or system state. Do not
edit files. Return only schema-valid JSON. If the log evidence is
not decisive, do not guess: use pause_for_human.

Embedded evidence:
{json.dumps(failure_evidence, ensure_ascii=False)}
"""
        analysis_dir = round_dir / "06_agent_analysis"
        prompt_path = analysis_dir / "failure_agent_prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        decision_path = analysis_dir / "failure_decision.json"
        result = run_structured_agent(
            self.agent_config,
            prompt=prompt,
            schema_path=self.decision_schema_path(
                session_dir, "failure_decision.schema.json"
            ),
            output_path=decision_path,
            cwd=KB_ROOT,
            allowed_dir=round_dir,
        )
        (analysis_dir / "failure_agent_events.jsonl").write_text(
            result.stdout, encoding="utf-8"
        )
        (analysis_dir / "failure_agent_stderr.log").write_text(
            result.stderr, encoding="utf-8"
        )
        if result.returncode != 0 or not decision_path.exists():
            raise RuntimeError(
                f"{result.provider} failure analysis failed: {result.stderr[-2000:]}"
            )
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        self.validate_failure_decision(session_dir, decision, current)
        self.write_selected_portrait_evidence(
            round_dir,
            decision["changes"],
            prefix="failure_selected",
        )
        self.refresh_runtime_rules(session_dir, round_dir)
        return decision

    def validate_failure_decision(
        self,
        session_dir: Path,
        decision: dict[str, Any],
        current: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Validate a failure decision and return its known-success rollback, if any."""
        action = decision["action"]
        candidate = decision["candidate"]
        changes = decision["changes"]
        if action == "adjust_parameters":
            if decision["classification"] not in {"parameter_invalid", "parameter_oom"}:
                raise ValueError("Only a proven parameter failure may adjust parameters")
            if not decision["safe_to_automate"]:
                raise ValueError("Agent marked parameter adjustment unsafe")
            self.validate_candidate(current, candidate, changes, decision)
            known_success = self.successful_candidate(session_dir, candidate)
            if self.candidate_was_attempted(session_dir, candidate) and not known_success:
                raise ValueError(
                    "Failure recovery proposed a previously failed candidate"
                )
            return known_success
        else:
            if candidate != current or changes:
                raise ValueError(f"{action} must preserve the candidate and have no changes")
            self.validate_no_change_metadata(decision)
        return None

    def deterministic_benchmark_retry(
        self,
        round_dir: Path,
        current: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return a bounded same-candidate retry for one clean missing request."""
        if self.benchmark_mode != "aligned_l1":
            return None
        runtime_dir = round_dir / "04_runtime"
        if not (runtime_dir / "SERVICE_READY").exists():
            return None

        successful: int
        planned: int
        retry_plan_path = runtime_dir / "benchmark_case_retry_plan.json"
        try:
            retry_plan = json.loads(retry_plan_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            retry_plan = {}
        if retry_plan.get("retryable") is True:
            successful = int(retry_plan["successful"])
            planned = int(retry_plan["planned"])
        else:
            # Compatibility fallback for old rounds produced before the remote
            # metrics compiler emitted a structured retry plan.
            log_path = runtime_dir / "benchmark_runner.log"
            if not log_path.is_file():
                return None
            text = log_path.read_text(encoding="utf-8", errors="replace")
            case_exits = re.findall(r"CASE COMPLETED .*? runner_exit=(\d+)", text)
            repetitions = int(self.benchmark["aligned_l1"]["repetitions"])
            expected_case_exits = 24 * repetitions
            if (
                len(case_exits) != expected_case_exits
                or any(code != "0" for code in case_exits)
            ):
                return None
            matches = re.findall(
                r"zero-error request gate failed: \{'successful': (\d+), "
                r"'incomplete': (\d+), 'errored': (\d+), 'total': (\d+)\}, "
                r"planned=(\d+), minimum_successful=(\d+)",
                text,
            )
            if len(matches) != 1:
                return None
            successful, incomplete, errored, total, planned, minimum = map(
                int, matches[0]
            )
            if not (
                incomplete == 0
                and errored == 0
                and total == successful
                and planned - successful == 1
                and successful < minimum <= planned
            ):
                return None

        summary = (
            "The strict aligned-L1 metrics gate rejected one formal case with "
            f"{successful}/{planned} requests; the identical candidate will be "
            "retried within the controller's bounded retry budget."
        )
        return {
            "summary": summary,
            "classification": "benchmark_failure",
            "root_cause": (
                "Exactly one request was absent from otherwise clean benchmark "
                "accounting; no incomplete or errored requests were accepted."
            ),
            "evidence": [
                "SERVICE_READY was present before the benchmark.",
                (
                    f"The structured/fallback request gate reported "
                    f"successful={successful}, planned={planned}."
                ),
            ],
            "action": "retry_same",
            "safe_to_automate": True,
            "change_strategy": "none",
            "interaction_analysis": [],
            "constraint_checks": [],
            "changes": [],
            "candidate": current,
        }

    def saved_failure_decision(
        self,
        session_dir: Path,
        round_dir: Path,
        current: dict[str, Any],
    ) -> dict[str, Any] | None:
        decision_path = round_dir / "06_agent_analysis" / "failure_decision.json"
        if not decision_path.exists():
            return None
        try:
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            self.validate_failure_decision(session_dir, decision, current)
            return decision
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            return None

    def analyze_after_known_good_rollback(
        self,
        session_dir: Path,
        failed_round_dir: Path,
        rollback: dict[str, Any],
        rollback_round: str,
        failure: dict[str, Any],
    ) -> dict[str, Any]:
        """Choose an untested next candidate after rejecting an invalid candidate."""
        attempted_history = self.attempted_history_summary(session_dir)
        successful_history = self.history_summary(session_dir)
        rollback_dir = session_dir / rollback_round
        evidence = {
            "known_good_round": rollback_round,
            "known_good_candidate": rollback,
            "known_good_metrics": json.loads(
                (rollback_dir / "05_results" / "metrics.json").read_text(
                    encoding="utf-8"
                )
            ),
            "known_good_comparison": json.loads(
                (rollback_dir / "05_results" / "comparison.json").read_text(
                    encoding="utf-8"
                )
            ),
            "failed_round": failed_round_dir.name,
            "failed_candidate": load_yaml(
                failed_round_dir / "02_parameters" / "candidate_params.yaml"
            ),
            "failure_decision": failure,
            "successful_history": successful_history,
            "attempted_history": attempted_history,
            "search_space": self.evidence_text(
                failed_round_dir / "01_query" / "glm5.2_search_space.yaml"
            ),
            "tag_knowledge": self.evidence_text(
                KB_ROOT / "SKILL-with-tag.md",
                max_chars=50000,
            ),
            "parameter_portraits": self.evidence_text(
                session_dir
                / "00_search_space"
                / "parameter_portraits.agent.yaml",
                max_chars=120000,
            ),
            "runtime_rules": (
                {
                    "rules": self.runtime_rule_store.data.get("rules", []),
                    "quarantines": self.runtime_rule_store.data.get(
                        "quarantines", []
                    ),
                    "proposals": self.runtime_rule_store.data.get(
                        "proposals", []
                    ),
                }
                if self.runtime_rule_store
                else None
            ),
        }
        analysis_dir = failed_round_dir / "06_agent_analysis"
        save_json(analysis_dir / "recovery_evidence_bundle.json", evidence)
        prompt = f"""You are the Codex tuning analyst recovering from a proven invalid
GLM-5.2 vLLM-Ascend candidate.

The failed candidate has already been rejected and must never be resubmitted. The
failure analyst recommended a rollback to a previously measured, known-good candidate.
Do not resubmit that known-good candidate either. Starting from the known-good
candidate, choose the smallest safe, evidence-based, untested change set of 1 to
{self.max_parameters_per_round} parameters, or return stop_complete if no useful safe
untested change remains.

All successful and failed configurations are listed in attempted_history. Never repeat
any of them. Treat parameter_invalid and parameter_oom configurations as excluded.
Preserve the fixed model, image, DP=2, TP=16, Pod/NPU topology, network, paths,
benchmark (including temperature=0), quantization, and Ascend environment. Multiple
    changes require a real interaction or constraint coupling, not independent guesses.
Explain at least one interaction per additional parameter and check every relevant
constraint. Grid-step budgets are:
{yaml.safe_dump(self.change_policy, allow_unicode=True, sort_keys=False)}

For stop_complete, preserve the known-good candidate, return an empty changes array,
change_strategy=none, and empty interaction_analysis and constraint_checks.

Allowed values:
{yaml.safe_dump(self.config['search_limits'], allow_unicode=True, sort_keys=False)}

Return only JSON matching the supplied schema. The exact known-good candidate is:
{json.dumps(rollback, ensure_ascii=False, indent=2)}

Embedded evidence:
{json.dumps(evidence, ensure_ascii=False)}
"""
        prompt_path = analysis_dir / "recovery_agent_prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        decision_path = analysis_dir / "recovery_decision.json"
        result = run_structured_agent(
            self.agent_config,
            prompt=prompt,
            schema_path=self.decision_schema_path(
                session_dir, "agent_decision.schema.json"
            ),
            output_path=decision_path,
            cwd=KB_ROOT,
            allowed_dir=failed_round_dir,
        )
        (analysis_dir / "recovery_agent_events.jsonl").write_text(
            result.stdout, encoding="utf-8"
        )
        (analysis_dir / "recovery_agent_stderr.log").write_text(
            result.stderr, encoding="utf-8"
        )
        if result.returncode != 0 or not decision_path.exists():
            raise RuntimeError(
                f"{result.provider} recovery analysis failed: {result.stderr[-2000:]}"
            )
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision["action"] == "stop_complete":
            if decision["candidate"] != rollback or decision["changes"]:
                raise ValueError(
                    "stop_complete must preserve the known-good candidate"
                )
            self.validate_no_change_metadata(decision)
            return decision
        self.write_selected_portrait_evidence(
            failed_round_dir,
            decision["changes"],
            prefix="recovery_selected",
        )
        self.validate_candidate(
            rollback,
            decision["candidate"],
            decision["changes"],
            decision,
        )
        if self.candidate_was_attempted(session_dir, decision["candidate"]):
            raise ValueError(
                "Recovery analysis proposed an already-attempted candidate"
            )
        save_yaml(
            analysis_dir / "recovery_next_candidate.yaml",
            {
                "generated_at": now(),
                "based_on_round": rollback_round,
                "rejected_round": failed_round_dir.name,
                "changes": decision["changes"],
                "candidate": decision["candidate"],
            },
        )
        return decision

    def prepare_and_submit_round(
        self,
        session_dir: Path,
        state: dict[str, Any],
        *,
        index: int,
        label: str,
        candidate: dict[str, Any],
    ) -> tuple[Path, str | None, str]:
        next_dir = self.round_dir(session_dir, index, label)
        self.write_context(
            next_dir,
            {
                **state,
                "round_index": index,
                "round_label": label,
                "active_run_id": None,
                "active_task_id": None,
            },
        )
        self.run_query(next_dir)
        save_yaml(next_dir / "02_parameters" / "candidate_params.yaml", candidate)
        task_id, run_id = self.submit(next_dir, label, candidate, dry_run=False)
        return next_dir, task_id, run_id

    def create_session(self) -> tuple[Path, dict[str, Any]]:
        if STATE_FILE.exists():
            try:
                previous_state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                previous_dir = Path(previous_state.get("session_dir", ""))
                if previous_dir.is_dir():
                    save_json(
                        previous_dir
                        / f"controller_state_archived_{dt.datetime.now():%Y%m%d_%H%M%S}.json",
                        previous_state,
                    )
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        session_id = "glm52_continuous_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = ARCHIVE_ROOT / session_id
        session_dir.mkdir(parents=True, exist_ok=False)
        write_session_search_space(
            session_dir,
            result=self.search_space_result,
            config=self.config,
        )
        self.write_decision_schemas(session_dir)
        self.initialize_session_sidecars(session_dir)
        state = {
            "session_id": session_id,
            "session_dir": str(session_dir),
            "status": "initialized",
            "round_index": 0,
            "candidate_index": 0,
            "round_label": str(self.config.get("initial_baseline", {}).get("label", "a0")),
            "active_task_id": None,
            "active_run_id": None,
            "execution_mode": self.execution_mode,
            "image_identity": self.image_identity,
            "search_limits_mode": self.config.get("resolved_search_space", {}).get(
                "mode", "manual"
            ),
            "strategy_profile": self.strategy_profile_name,
            "agent_provider": self.agent_config.get("provider", "codex"),
            "benchmark_profile": self.benchmark_profile_name,
            "active_search_parameters": list(
                self.config.get("resolved_search_space", {}).get(
                    "active_tunable_parameters",
                    [
                        name
                        for name, values in self.config["search_limits"].items()
                        if len(values) > 1
                    ],
                )
            ),
            "lease_name": (
                self.lab.get("lease_name")
                if self.execution_mode == "ktp_lab"
                else None
            ),
            "current_candidate": self.config["baseline"],
            "failure_retries": 0,
            "failure_adjustments": 0,
            "round_submitted_at": None,
            "created_at": now(),
            "updated_at": now(),
        }
        save_json(STATE_FILE, state)
        save_yaml(session_dir / "session_config.yaml", self.config)
        save_yaml(session_dir / "image_version_manifest.yaml", self.image_manifest)
        return session_dir, state

    def save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = now()
        save_json(STATE_FILE, state)

    def reanalyze_current(self) -> dict[str, Any]:
        if not STATE_FILE.exists():
            raise RuntimeError("No controller state exists to reanalyze")
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        self.assert_state_image_identity(state)
        session_dir = Path(state["session_dir"])
        self.load_session_sidecars(session_dir)
        round_dir = self.round_dir(
            session_dir,
            int(state["round_index"]),
            str(state["round_label"]),
        )
        metrics_path = round_dir / "05_results" / "metrics.json"
        if not metrics_path.exists():
            raise RuntimeError(f"Current round has no saved metrics: {metrics_path}")
        decision = self.analyze(session_dir, round_dir, state["current_candidate"])
        state["analysis_status"] = "ready"
        state["analysis_decision_path"] = str(
            round_dir / "06_agent_analysis" / "decision.json"
        )
        state["analysis_action"] = decision["action"]
        self.save_state(state)
        log(
            f"Reanalyzed saved round {state['round_label']}; "
            f"action={decision['action']}. No experiment was submitted."
        )
        return decision

    def retry_paused_current(self) -> dict[str, Any]:
        """Operator-authorized same-candidate diagnostic retry."""
        if not STATE_FILE.exists():
            raise RuntimeError("No controller state exists to retry")
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        allowed_statuses = {
            "paused_for_human",
            "paused_after_repeated_infrastructure_failure",
            "paused_controller_error",
        }
        if state.get("status") not in allowed_statuses:
            raise RuntimeError(
                "Current state is not paused for operator recovery: "
                f"{state.get('status')!r}"
            )
        self.assert_state_image_identity(state)
        session_dir = Path(state["session_dir"])
        self.load_session_sidecars(session_dir)
        failed_round = self.round_dir(
            session_dir,
            int(state["round_index"]),
            str(state["round_label"]),
        )
        failure_path = failed_round / "05_results" / "failure.yaml"
        metrics_path = failed_round / "05_results" / "metrics.json"
        if not failure_path.exists() or metrics_path.exists():
            raise RuntimeError(
                "Paused round must contain failure.yaml and no metrics.json"
            )
        task = self.task_snapshot(state.get("active_task_id"))
        if not task.get("terminal") and int(task.get("active_pods", 0)) > 0:
            raise RuntimeError(
                "Paused task still has active processes; refusing an overlapping retry"
            )

        retry_number = int(state.get("failure_retries", 0)) + 1
        next_index = int(state["round_index"]) + 1
        next_label = f"a{int(state['candidate_index'])}r{retry_number}"
        save_json(
            failed_round / "06_agent_analysis" / "operator_retry_authorization.json",
            {
                "authorized_at": now(),
                "source_status": state["status"],
                "failed_round": failed_round.name,
                "retry_label": next_label,
                "candidate": state["current_candidate"],
                "reason": (
                    "Operator authorized one identical-candidate retry to distinguish "
                    "a transient runtime failure from a reproducible candidate-related "
                    "failure."
                ),
            },
        )
        _, task_id, run_id = self.prepare_and_submit_round(
            session_dir,
            state,
            index=next_index,
            label=next_label,
            candidate=state["current_candidate"],
        )
        state.update(
            round_index=next_index,
            round_label=next_label,
            active_task_id=task_id,
            active_run_id=run_id,
            failure_retries=retry_number,
            round_submitted_at=now(),
            status="running",
            recovery_source_round=failed_round.name,
            recovery_reason="operator_authorized_same_candidate_diagnostic_retry",
        )
        self.save_state(state)
        log(
            f"Operator-authorized retry submitted {next_label} "
            f"task={task_id} run={run_id}"
        )
        return state

    def saved_analysis_decision(
        self,
        session_dir: Path,
        round_dir: Path,
        previous: dict[str, Any],
    ) -> dict[str, Any] | None:
        decision_path = round_dir / "06_agent_analysis" / "decision.json"
        if not decision_path.exists():
            return None
        try:
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            if decision["action"] == "stop_complete":
                if decision["candidate"] != previous or decision["changes"]:
                    return None
                self.validate_no_change_metadata(decision)
            else:
                selection_policy = self.effective_change_policy(
                    self.history_summary(session_dir)
                )
                self.validate_candidate(
                    previous,
                    decision["candidate"],
                    decision["changes"],
                    decision,
                    selection_policy,
                )
                if self.candidate_was_attempted(
                    session_dir,
                    decision["candidate"],
                ):
                    return None
            return decision
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            return None

    def dry_run_validation(self) -> None:
        session_dir, state = self.create_session()
        state["status"] = "dry_run"
        round_dir = self.round_dir(session_dir, 0, "a0_dryrun")
        self.write_context(round_dir, state)
        self.run_query(round_dir)
        save_yaml(
            round_dir / "02_parameters" / "candidate_params.yaml",
            self.config["baseline"],
        )
        _, run_id = self.submit(
            round_dir, "a0dryrun", self.config["baseline"], dry_run=True
        )
        state["active_run_id"] = run_id
        state["status"] = "dry_run_complete"
        self.save_state(state)
        log(f"Dry-run complete: {session_dir}")

    def start(self, *, resume: bool = False) -> None:
        if resume:
            if not STATE_FILE.exists():
                raise RuntimeError("No controller state exists to resume")
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            self.assert_state_image_identity(state)
            session_dir = Path(state["session_dir"])
            if not session_dir.is_dir():
                raise RuntimeError(f"Session directory does not exist: {session_dir}")
            self.load_session_sidecars(session_dir)
            self.write_decision_schemas(session_dir)
            if not state.get("active_task_id") or not state.get("active_run_id"):
                archived_round = self.round_dir(
                    session_dir,
                    int(state["round_index"]),
                    str(state["round_label"]),
                )
                resumable_statuses = {
                    "stopped_after_current_round",
                    "stopped_after_failed_round",
                }
                has_terminal_artifact = (
                    (archived_round / "05_results" / "metrics.json").exists()
                    or (archived_round / "05_results" / "failure.yaml").exists()
                )
                if (
                    state.get("status") not in resumable_statuses
                    or not has_terminal_artifact
                ):
                    raise RuntimeError(
                        "Controller state has no active task/run or resumable "
                        "archived round"
                    )
            state["status"] = "running"
            state["controller_error"] = None
            self.save_state(state)
            log(
                f"Resumed session={state['session_id']} "
                f"task={state['active_task_id']} run={state['active_run_id']}"
            )
        else:
            if STATE_FILE.exists():
                try:
                    previous_state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    previous_state = {}
                previous_task = previous_state.get("active_task_id")
                if previous_task:
                    snapshot = self.task_snapshot(str(previous_task))
                    if not snapshot["terminal"]:
                        raise RuntimeError(
                            "The previous controller state still references active "
                            f"task {previous_task} (status={snapshot['status']}, "
                            f"active_pods={snapshot['active_pods']}). Refusing to "
                            "start an overlapping tuning session."
                        )
            session_dir, state = self.create_session()
            initial_label = str(
                self.config.get("initial_baseline", {}).get("label", "a0")
            )
            round_dir = self.round_dir(session_dir, 0, initial_label)
            self.write_context(round_dir, state)
            self.run_query(round_dir)
            save_yaml(
                round_dir / "02_parameters" / "candidate_params.yaml",
                state["current_candidate"],
            )
            task_id, run_id = self.submit(
                round_dir, initial_label, state["current_candidate"], dry_run=False
            )
            state.update(
                status="running",
                active_task_id=task_id,
                active_run_id=run_id,
                round_submitted_at=now(),
            )
            self.save_state(state)
            log(f"Submitted {initial_label.upper()} task={task_id} run={run_id}")

        while True:
            stop_requested = STOP_FILE.exists()
            if stop_requested and state["status"] != "stop_requested":
                state["status"] = "stop_requested"
                self.save_state(state)
                log(
                    "STOP_REQUESTED detected. The active round will be archived; "
                    "no new experiment will be submitted."
                )

            round_dir = self.round_dir(
                session_dir, state["round_index"], state["round_label"]
            )
            found = self.collect(state["active_run_id"], round_dir)
            task = None
            if (
                self.benchmark_mode == "aligned_l1"
                and found.get("SERVICE_READY")
                and not found.get("BENCHMARK_STARTED")
            ):
                task = self.task_snapshot(state.get("active_task_id"))
                if self.should_start_aligned_benchmark(found, task):
                    self.start_aligned_benchmark(
                        state["active_run_id"], state.get("active_task_id")
                    )
            if found.get("metrics.json"):
                self.wait_for_task_release(state.get("active_task_id"))
                if stop_requested:
                    self.write_comparison(session_dir, round_dir)
                    state.update(
                        status="stopped_after_current_round",
                        active_task_id=None,
                    )
                    self.save_state(state)
                    log("Active round completed and was archived; controller stopped.")
                    return
                self.reconcile_official_source_default_baseline(
                    session_dir, round_dir, state
                )
                self.save_state(state)
                log(f"Round {state['round_label']} completed; invoking Codex analysis.")
                decision = None
                if state.get("analysis_status") == "ready":
                    decision = self.saved_analysis_decision(
                        session_dir,
                        round_dir,
                        state["current_candidate"],
                    )
                    if decision:
                        log("Reusing the validated saved Codex decision.")
                if decision is None:
                    decision = self.analyze(
                        session_dir, round_dir, state["current_candidate"]
                    )
                if decision["action"] == "stop_complete":
                    state.update(
                        status="completed_by_agent",
                        active_task_id=None,
                        completion_summary=decision["summary"],
                    )
                    self.save_state(state)
                    log("Codex determined that tuning is complete; no new round submitted.")
                    return
                next_index = state["round_index"] + 1
                next_candidate_index = state["candidate_index"] + 1
                next_label = f"a{next_candidate_index}"
                next_candidate = decision["candidate"]
                _, task_id, run_id = self.prepare_and_submit_round(
                    session_dir,
                    state,
                    index=next_index,
                    label=next_label,
                    candidate=next_candidate,
                )
                state.update(
                    round_index=next_index,
                    candidate_index=next_candidate_index,
                    round_label=next_label,
                    active_task_id=task_id,
                    active_run_id=run_id,
                    current_candidate=next_candidate,
                    failure_retries=0,
                    failure_adjustments=0,
                    round_submitted_at=now(),
                    status="running",
                )
                self.save_state(state)
                log(f"Submitted {next_label} task={task_id} run={run_id}")
            else:
                task = task or self.task_snapshot(state.get("active_task_id"))
                timed_out = self.round_timed_out(state)
                partial_exit = self.partial_exit_is_failure(state, task)
                failed = (
                    found.get("MASTER_DONE")
                    or task["terminal"]
                    or partial_exit
                    or timed_out
                )
                if not failed:
                    time.sleep(self.poll_seconds)
                    continue
                if partial_exit:
                    self.stop_partial_lab_processes(state.get("active_task_id"))
                self.wait_for_task_release(state.get("active_task_id"))
                reasons = []
                if found.get("MASTER_DONE"):
                    reasons.append("MASTER_DONE exists but metrics.json is absent")
                if task["terminal"]:
                    reasons.append(
                        f"platform task is terminal: status={task['status']}, "
                        f"active_pods={task['active_pods']}"
                    )
                if partial_exit:
                    reasons.append(
                        "persistent lease process set partially exited: "
                        f"status={task['status']}, counts={task.get('process_counts')}"
                    )
                if timed_out:
                    reasons.append(
                        f"round exceeded {self.round_timeout_minutes} minutes"
                    )
                save_yaml(
                    round_dir / "05_results" / "failure.yaml",
                    {
                        "detected_at": now(),
                        "reason": "; ".join(reasons),
                        "task": task,
                    },
                )
                if stop_requested:
                    state.update(
                        status="stopped_after_failed_round",
                        active_task_id=None,
                        last_failure_classification="operator_stop_before_metrics",
                        last_failure_summary="; ".join(reasons),
                    )
                    self.save_state(state)
                    log("Active round failed and was archived; controller stopped.")
                    return
                log(f"Round {state['round_label']} failed; invoking Codex failure analysis.")
                failure = self.deterministic_benchmark_retry(
                    round_dir,
                    state["current_candidate"],
                )
                if failure is not None:
                    save_json(
                        round_dir
                        / "06_agent_analysis"
                        / "deterministic_recovery_decision.json",
                        failure,
                    )
                    log(
                        "Deterministic recovery classified a one-request benchmark "
                        "shortfall as safe for bounded same-candidate retry."
                    )
                else:
                    failure = self.saved_failure_decision(
                        session_dir,
                        round_dir,
                        state["current_candidate"],
                    ) or self.analyze_failure(
                        session_dir,
                        round_dir,
                        state["current_candidate"],
                    )
                action = failure["action"]
                if action == "pause_for_human":
                    state.update(
                        status="paused_for_human",
                        last_failure_classification=failure["classification"],
                        last_failure_summary=failure["summary"],
                    )
                    self.save_state(state)
                    log(
                        "Paused because the failure is not safe to repair automatically: "
                        + failure["summary"]
                    )
                    return

                rollback_success = None
                if action == "adjust_parameters":
                    rollback_success = self.successful_candidate(
                        session_dir,
                        failure["candidate"],
                    )
                if rollback_success:
                    recovery = self.analyze_after_known_good_rollback(
                        session_dir,
                        round_dir,
                        failure["candidate"],
                        rollback_success["round"],
                        failure,
                    )
                    save_json(
                        round_dir
                        / "06_agent_analysis"
                        / "recovery_resolution.json",
                        {
                            "resolved_at": now(),
                            "rejected_round": round_dir.name,
                            "rejected_candidate": state["current_candidate"],
                            "rollback_round": rollback_success["round"],
                            "rollback_candidate": failure["candidate"],
                            "next_action": recovery["action"],
                        },
                    )
                    if recovery["action"] == "stop_complete":
                        state.update(
                            status="tuning_complete",
                            active_task_id=None,
                            active_run_id=None,
                            current_candidate=failure["candidate"],
                            completion_summary=recovery["summary"],
                            last_failure_classification=failure["classification"],
                            last_failure_summary=failure["summary"],
                            recovered_best_round=rollback_success["round"],
                        )
                        self.save_state(state)
                        log(
                            "Invalid candidate rejected; Codex determined that "
                            "tuning is complete at the known-good rollback."
                        )
                        return

                    next_index = state["round_index"] + 1
                    next_candidate_index = state["candidate_index"] + 1
                    next_label = f"a{next_candidate_index}"
                    next_candidate = recovery["candidate"]
                    _, task_id, run_id = self.prepare_and_submit_round(
                        session_dir,
                        state,
                        index=next_index,
                        label=next_label,
                        candidate=next_candidate,
                    )
                    state.update(
                        round_index=next_index,
                        candidate_index=next_candidate_index,
                        round_label=next_label,
                        active_task_id=task_id,
                        active_run_id=run_id,
                        current_candidate=next_candidate,
                        failure_retries=0,
                        failure_adjustments=0,
                        round_submitted_at=now(),
                        status="running",
                        last_failure_classification=failure["classification"],
                        last_failure_summary=failure["summary"],
                        recovered_best_round=rollback_success["round"],
                    )
                    self.save_state(state)
                    log(
                        f"Invalid candidate rejected; submitted {next_label} "
                        f"from known-good {rollback_success['round']} "
                        f"task={task_id} run={run_id}"
                    )
                    time.sleep(self.poll_seconds)
                    continue

                next_index = state["round_index"] + 1
                if action == "adjust_parameters":
                    state["failure_retries"] = 0
                    state["failure_adjustments"] += 1
                    next_candidate = failure["candidate"]
                    next_label = (
                        f"a{state['candidate_index']}f{state['failure_adjustments']}"
                    )
                    next_status = "recovering_parameter_failure"
                else:
                    state["failure_retries"] += 1
                    if state["failure_retries"] > 2:
                        state.update(
                            status="paused_after_repeated_infrastructure_failure",
                            last_failure_classification=failure["classification"],
                            last_failure_summary=failure["summary"],
                        )
                        self.save_state(state)
                        log("Paused after three identical infrastructure retries.")
                        return
                    next_candidate = state["current_candidate"]
                    next_label = (
                        f"a{state['candidate_index']}r{state['failure_retries']}"
                    )
                    next_status = "retrying_infrastructure_failure"

                _, task_id, run_id = self.prepare_and_submit_round(
                    session_dir,
                    state,
                    index=next_index,
                    label=next_label,
                    candidate=next_candidate,
                )
                state.update(
                    round_index=next_index,
                    round_label=next_label,
                    active_task_id=task_id,
                    active_run_id=run_id,
                    current_candidate=next_candidate,
                    round_submitted_at=now(),
                    status=next_status,
                    last_failure_classification=failure["classification"],
                    last_failure_summary=failure["summary"],
                )
                self.save_state(state)
                log(f"Failure recovery submitted {next_label} task={task_id} run={run_id}")
            time.sleep(self.poll_seconds)


def acquire_controller_lock() -> int:
    try:
        descriptor = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            existing_pid = int(LOCK_FILE.read_text(encoding="ascii").strip())
        except ValueError:
            existing_pid = -1
        if not process_is_running(existing_pid):
            archived = LOCK_FILE.with_name(
                f"{LOCK_FILE.name}.stale-{dt.datetime.now():%Y%m%d_%H%M%S}"
            )
            os.replace(LOCK_FILE, archived)
            return acquire_controller_lock()
        raise RuntimeError(
            f"A continuous controller is already running with PID {existing_pid}"
        )
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    return descriptor


def process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def release_controller_lock(descriptor: int) -> None:
    os.close(descriptor)
    try:
        if LOCK_FILE.read_text(encoding="ascii").strip() == str(os.getpid()):
            LOCK_FILE.unlink()
    except FileNotFoundError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start", action="store_true", help="start a new continuous session")
    group.add_argument("--resume", action="store_true", help="resume the active session from state.json")
    group.add_argument("--dry-run", action="store_true", help="validate without submitting")
    group.add_argument(
        "--check-only",
        action="store_true",
        help="validate local configuration, SSH connectivity, and idle Lease without writes",
    )
    group.add_argument(
        "--prepare-lab",
        action="store_true",
        help="submit the persistent ktp-lab lease after validating configuration",
    )
    group.add_argument(
        "--reanalyze-current",
        action="store_true",
        help="reanalyze saved metrics without submitting another experiment",
    )
    group.add_argument(
        "--retry-paused-current",
        action="store_true",
        help="retry the paused candidate after an operator-authorized external fix",
    )
    group.add_argument("--status", action="store_true", help="print controller state")
    parser.add_argument(
        "--strategy-profile",
        help="strategy profile for a new Session; resume uses the frozen Session value",
    )
    parser.add_argument(
        "--agent-provider",
        choices=["codex", "anthropic", "openai_compatible", "command"],
        help="Agent provider for a new Session; credentials stay in environment variables",
    )
    parser.add_argument(
        "--benchmark-profile",
        help="benchmark profile for a new Session; resume uses the frozen Session value",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.status:
        if not STATE_FILE.exists():
            print("No controller state exists.")
            return 0
        print(STATE_FILE.read_text(encoding="utf-8"))
        return 0
    search_space_result = None
    if (
        args.resume
        or args.reanalyze_current
        or args.retry_paused_current
    ) and STATE_FILE.exists():
        if args.strategy_profile or args.agent_provider or args.benchmark_profile:
            raise RuntimeError(
                "Strategy, Agent provider, and Benchmark profile are frozen in "
                "session_config.yaml; "
                "start a new Session to change them"
            )
        frozen_state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        frozen_config = Path(frozen_state["session_dir"]) / "session_config.yaml"
        if not frozen_config.is_file():
            raise RuntimeError(
                f"Frozen Session configuration is missing: {frozen_config}"
            )
        config = load_yaml(frozen_config)
    else:
        raw_config = load_yaml(HERE / "config.yaml")
        if args.strategy_profile:
            raw_config.setdefault("strategy", {})["profile"] = args.strategy_profile
        if args.agent_provider:
            raw_config.setdefault("agent", {})["provider"] = args.agent_provider
        if args.benchmark_profile:
            raw_config.setdefault("benchmark", {})["profile"] = args.benchmark_profile
        config, search_space_result = resolve_search_limits(
            raw_config,
            project_root=KB_ROOT,
            archive_root=ARCHIVE_ROOT,
        )
    controller = Controller(
        config,
        dry_run=args.dry_run,
        search_space_result=search_space_result,
    )
    validate_agent_credentials(controller.agent_config)
    if args.check_only:
        try:
            controller.check_ready()
        except Exception as exc:
            print(f"End-to-end preflight failed: {exc}", file=sys.stderr)
            return 2
        print("Controller, SSH, and persistent Lease preflight: OK")
        return 0
    lock_descriptor = acquire_controller_lock()
    try:
        if args.prepare_lab:
            controller.validate_runtime_configuration(config["baseline"])
            print(controller.prepare_lab(submit=True))
        elif args.reanalyze_current:
            print(
                json.dumps(
                    controller.reanalyze_current(),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.retry_paused_current:
            print(
                json.dumps(
                    controller.retry_paused_current(),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            # A recovery command must rejoin the normal controller loop after
            # submitting the replacement round. Otherwise the remote task runs
            # unattended and no later metrics/Agent decision can advance the
            # closed loop.
            controller.start(resume=True)
        elif args.dry_run:
            controller.dry_run_validation()
        else:
            if STOP_FILE.exists():
                raise RuntimeError(
                    f"{STOP_FILE} exists. Rename it before starting a new session."
                )
            try:
                controller.start(resume=args.resume)
            except Exception as exc:
                if STATE_FILE.exists():
                    try:
                        failed_state = json.loads(
                            STATE_FILE.read_text(encoding="utf-8")
                        )
                        paused_status = (
                            "paused_after_repeated_candidate_rejection"
                            if isinstance(exc, RepeatedCandidateRejection)
                            else "paused_controller_error"
                        )
                        failed_state.update(
                            status=paused_status,
                            controller_error=f"{type(exc).__name__}: {exc}",
                            updated_at=now(),
                        )
                        save_json(STATE_FILE, failed_state)
                        if isinstance(exc, RepeatedCandidateRejection):
                            log(
                                "Controller paused after consecutive deterministic "
                                f"candidate rejections: {exc}"
                            )
                        else:
                            log(
                                "Controller paused after an internal error: "
                                f"{type(exc).__name__}: {exc}"
                            )
                    except (OSError, ValueError, json.JSONDecodeError):
                        pass
                raise
    finally:
        release_controller_lock(lock_descriptor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
