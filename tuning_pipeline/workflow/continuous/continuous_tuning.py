#!/usr/bin/env python3
"""Continuous vLLM-Ascend tuning controller.

The default mode runs on Windows beside the local knowledge base.  An isolated
server-autonomous mode can run the same deterministic controller on Linux with
its own state root and local execution transport.  Experiment artifacts are
never deleted by this program.
"""

from __future__ import annotations

import argparse
import ast
import base64
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
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import yaml

if str(Path(__file__).resolve().parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from .executor_adapter import (
        EXECUTOR_ADAPTER_API_VERSION,
        resolve_executor_adapter,
        validate_snapshot as validate_executor_snapshot,
    )
    from .search_space_adapter import resolve_search_limits, write_session_search_space
    from .model_loading_profile import resolve_model_loading_profile
    from .runtime_profile import (
        apply_topology_baseline_binding,
        resolve_runtime_profile,
        validate_runtime_selections,
    )
    from .topology_profile import resolve_topology_profile
    from .topology_advisor import build_plan as build_topology_plan, load_document as load_topology_document
    from .agent_provider import (
        resolve_agent_profile,
        run_structured_agent,
        validate_agent_credentials,
    )
except ImportError:  # Direct script execution.
    from executor_adapter import (
        EXECUTOR_ADAPTER_API_VERSION,
        resolve_executor_adapter,
        validate_snapshot as validate_executor_snapshot,
    )
    from search_space_adapter import resolve_search_limits, write_session_search_space
    from model_loading_profile import resolve_model_loading_profile
    from runtime_profile import (
        apply_topology_baseline_binding,
        resolve_runtime_profile,
        validate_runtime_selections,
    )
    from topology_profile import resolve_topology_profile
    from topology_advisor import build_plan as build_topology_plan, load_document as load_topology_document
    from agent_provider import (
        resolve_agent_profile,
        run_structured_agent,
        validate_agent_credentials,
    )
from workflow.search_space_compiler.compiler import (
    validate_candidate as validate_search_space_candidate,
)
from workflow.registry_builder.candidate_validator import validate_trial_candidate
from workflow.registry_builder.compatibility import CompatibilityValidator
from workflow.registry_builder.pipeline import compile_generic_runtime_payload
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
ACTIVATION_FILE = HERE / "activation.approved.yaml"
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
    "RFORK_TRANSFER_VERIFIED",
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
    "node_checkpoint_prefetch.py",
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
    "benchmark_driver.py",
    "benchmark_result.schema.json",
)


class RepeatedCandidateRejection(RuntimeError):
    """Raised after the Agent repeatedly fails deterministic candidate validation."""


class AgentProtocolError(RuntimeError):
    """The Agent transport/structured-output contract may succeed on retry."""


class RecoverableControllerIOError(RuntimeError):
    """A read-only control-plane operation may succeed after process restart."""


class LeaseNotReadyError(RuntimeError):
    """The persistent Lease may recover without changing experiment state."""


def session_budget_pause_status(
    candidate_index: int,
    pause_after_candidate_index: int | None,
    *,
    topology_feasibility_only: bool,
) -> str | None:
    """Return the terminal slice status only after the requested metrics index."""
    if (
        pause_after_candidate_index is None
        or candidate_index < pause_after_candidate_index
    ):
        return None
    return (
        "topology_feasibility_passed"
        if topology_feasibility_only
        else "budget_paused"
    )


MANUAL_INTERVENTION_SIGNATURES = re.compile(
    r"image.*(?:digest|identity|mismatch|not approved)|(?:digest|commit).*mismatch|"
    r"permission denied|access denied|no such file or directory[^\n]{0,200}"
    r"(?:model|image|checkpoint|tokenizer)|missing (?:model|image|"
    r"credential|api key|secret)|invalid (?:credential|api key)|authentication failed|"
    r"(?:frozen|immutable) configuration.*mismatch|state.*(?:inconsistent|corrupt)|"
    r"topology.*mismatch|model path.*(?:missing|unavailable|not found)",
    re.IGNORECASE,
)


def controller_exception_is_recoverable(exc: BaseException) -> bool:
    """Classify process-restart-safe control-plane failures conservatively."""
    if isinstance(
        exc,
        (
            AgentProtocolError,
            RecoverableControllerIOError,
            LeaseNotReadyError,
            RepeatedCandidateRejection,
        ),
    ):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, subprocess.SubprocessError)):
        return True
    if isinstance(exc, OSError) and not isinstance(exc, PermissionError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    text = str(exc)
    # ktp-lab ranks share the run directory.  Its atomic JSON writer currently
    # derives the temporary suffix from the container-local PID, so two ranks
    # can occasionally choose the same name (for example request.json.tmp-959)
    # and race on the final rename.  This is a transient control-plane write
    # collision, not a missing model/artifact, even when the workspace path
    # itself contains the word ``model`` (for example /mnt/host-model/...).
    if re.search(
        r"FileNotFoundError:.*\.json\.tmp-[^'\"\s]+['\"]?\s*->\s*"
        r"['\"][^'\"\n]+\.json",
        text,
        re.IGNORECASE | re.DOTALL,
    ):
        return True
    if MANUAL_INTERVENTION_SIGNATURES.search(text):
        return False
    return bool(
        re.search(
            r"ssh|scp|sftp|paramiko|connection|timed? ?out|timeout|temporar|"
            r"heartbeat|lease.*(?:ready|active|admission)|ktp-lab|pod|network|"
            r"transport|remote (?:command|status|artifact)|http.*(?:429|5\d\d)",
            text,
            re.IGNORECASE,
        )
    )


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
    "enable_balance_scheduling": "ADDITIONAL_CONFIG_ENABLE_BALANCE_SCHEDULING",
    "enable_reduce_sample": "ADDITIONAL_CONFIG_ENABLE_REDUCE_SAMPLE",
    "speculative_config__enforce_eager": "SPECULATIVE_CONFIG_ENFORCE_EAGER_JSON",
}

B0_LAUNCH_PROFILE = "official_source_defaults_deployable"

# This is the controller's authoritative failure-action contract.  Keep the
# Agent prompt and deterministic validator derived from these sets so a model
# cannot propose an action that the controller later rejects because the two
# layers silently drifted apart.
FAILURE_ADJUSTABLE_CLASSIFICATIONS = frozenset(
    {
        "parameter_invalid",
        "parameter_oom",
        "model_or_runtime_bug",
        "network_or_hccl",
        "transient_infrastructure",
        "image_or_dependency",
        "unknown",
    }
)
FAILURE_RETRYABLE_CLASSIFICATIONS = frozenset(
    {"transient_infrastructure", "network_or_hccl", "benchmark_failure"}
)
FAILURE_DIAGNOSTIC_RETRY_CLASSIFICATIONS = frozenset(
    {
        "parameter_invalid",
        "parameter_oom",
        "transient_infrastructure",
        "network_or_hccl",
        "image_or_dependency",
        "model_or_runtime_bug",
        "benchmark_failure",
        "unknown",
    }
)
FAILURE_ALL_CLASSIFICATIONS = frozenset(
    {
        "parameter_invalid",
        "parameter_oom",
        "transient_infrastructure",
        "network_or_hccl",
        "image_or_dependency",
        "model_or_runtime_bug",
        "benchmark_failure",
        "unknown",
    }
)
FAILURE_ACTION_CLASSIFICATIONS = {
    "adjust_parameters": FAILURE_ADJUSTABLE_CLASSIFICATIONS,
    "retry_same": FAILURE_RETRYABLE_CLASSIFICATIONS,
    "diagnostic_retry_same": FAILURE_DIAGNOSTIC_RETRY_CLASSIFICATIONS,
    "pause_for_human": FAILURE_ALL_CLASSIFICATIONS,
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


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Path) -> dict[str, Any]:
    """Load a config and an optional relative ``base_config`` overlay."""
    path = path.resolve()
    value = load_yaml(path)
    base_setting = value.pop("base_config", None)
    if not base_setting:
        return value
    base_path = (path.parent / str(base_setting)).resolve()
    if base_path == path:
        raise RuntimeError("Configuration cannot extend itself")
    return deep_merge(load_config(base_path), value)


def resolve_initial_baseline_definition(
    config: dict[str, Any], project_root: Path
) -> dict[str, Any]:
    """Materialize an explicit candidate from its sole baseline definition.

    B0 intentionally keeps typed source-default estimates in ``config.yaml`` and
    has no ``reference_parameters`` section.  Expert A0 definitions do carry
    that section; when present it replaces any inherited/overlaid ``baseline``
    mapping so autonomous and Windows controllers cannot maintain a second,
    drifting copy of the same candidate.
    """
    resolved = copy.deepcopy(config)
    initial = resolved.get("initial_baseline", {})
    if not isinstance(initial, dict):
        raise ValueError("initial_baseline must be a mapping")
    definition = initial.get("definition")
    if not definition:
        return resolved
    definition_path = Path(str(definition))
    if not definition_path.is_absolute():
        definition_path = project_root / definition_path
    definition_path = definition_path.resolve()
    if not definition_path.is_file():
        raise FileNotFoundError(f"Initial baseline definition does not exist: {definition_path}")
    document = load_yaml(definition_path)
    if not isinstance(document, dict):
        raise ValueError(f"Initial baseline definition must be a mapping: {definition_path}")
    reference = document.get("reference_parameters")
    if reference is None:
        return resolved
    if not isinstance(reference, dict) or not reference:
        raise ValueError(
            f"reference_parameters must be a non-empty mapping: {definition_path}"
        )
    declared_profile = document.get("launch_profile")
    configured_profile = initial.get("launch_profile")
    if declared_profile and configured_profile != declared_profile:
        raise ValueError(
            "Initial baseline launch_profile differs from its definition: "
            f"config={configured_profile!r}, definition={declared_profile!r}"
        )
    resolved["baseline"] = copy.deepcopy(reference)
    data = definition_path.read_bytes()
    resolved.setdefault("initial_baseline", {})["resolved_definition"] = {
        "baseline_id": document.get("baseline_id"),
        "path": (
            definition_path.relative_to(project_root.resolve()).as_posix()
            if definition_path.is_relative_to(project_root.resolve())
            else str(definition_path)
        ),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    return resolved


def configure_runtime_root(path: Path | None) -> None:
    """Redirect mutable controller state without changing legacy defaults."""
    if path is None:
        return
    root = path.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    global ARCHIVE_ROOT, STATE_FILE, STOP_FILE, LOG_FILE, LOCK_FILE
    ARCHIVE_ROOT = root / "experiments"
    STATE_FILE = root / "state.json"
    STOP_FILE = root / "STOP_REQUESTED"
    LOCK_FILE = root / "controller.lock"
    if "VLLMTKB_CONTROLLER_LOG" not in os.environ:
        LOG_FILE = root / "logs" / "controller" / "controller.log"


def validate_activation_approval(
    image_manifest: dict[str, Any] | None = None,
    *,
    approval_path: Path = ACTIVATION_FILE,
    approval: dict[str, Any] | None = None,
) -> None:
    """Require explicit approval for the exact runtime image and source pair."""
    manifest = image_manifest or load_yaml(IMAGE_MANIFEST_FILE)
    if approval is None and not approval_path.is_file():
        raise RuntimeError(
            "Remote activation is locked. Copy activation.example.yaml to "
            "activation.approved.yaml only after verifying the real runtime image."
        )
    approval = approval or load_yaml(approval_path)
    if approval.get("approved") is not True:
        raise RuntimeError(f"Remote activation is not approved in {approval_path}")
    target = approval.get("target")
    if not isinstance(target, dict):
        raise RuntimeError(f"Remote activation target is missing in {approval_path}")
    manifest_target = manifest.get("target_image", {})
    versions = manifest.get("versions", {})
    expected = {
        "image": (
            f"{manifest_target.get('repository', '')}:"
            f"{manifest_target.get('tag', '')}"
        ),
        "image_digest": manifest_target.get("digest"),
        "vllm_commit": versions.get("vllm", {}).get("commit"),
        "vllm_ascend_commit": versions.get("vllm_ascend", {}).get("commit"),
    }
    incomplete = [name for name, value in expected.items() if not value]
    if incomplete:
        raise RuntimeError(
            "Image version manifest is incomplete for activation validation: "
            + ", ".join(incomplete)
        )
    mismatches = {
        name: {"approved": target.get(name), "expected": value}
        for name, value in expected.items()
        if target.get(name) != value
    }
    if mismatches:
        raise RuntimeError(
            "Remote activation does not match image_version_manifest.yaml: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    evidence = approval.get("evidence")
    required_evidence = {
        "approved_at": approval.get("approved_at"),
        "approved_by": approval.get("approved_by"),
        "evidence.package_versions": (
            evidence.get("package_versions") if isinstance(evidence, dict) else None
        ),
        "evidence.source_commit_probe": (
            evidence.get("source_commit_probe") if isinstance(evidence, dict) else None
        ),
    }
    missing_evidence = [name for name, value in required_evidence.items() if not value]
    if missing_evidence:
        raise RuntimeError(
            "Remote activation evidence is incomplete: " + ", ".join(missing_evidence)
        )


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_name(f".{path.name}.write-{os.getpid()}-{time.time_ns()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(value, allow_unicode=True, sort_keys=False)
    temporary = path.with_name(f".{path.name}.write-{os.getpid()}-{time.time_ns()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_controller_state() -> dict[str, Any]:
    """Read primary state, falling back to the last known-good snapshot."""
    try:
        value = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Controller state must be a JSON object")
        return value
    except (OSError, ValueError, json.JSONDecodeError) as primary_error:
        backup = STATE_FILE.with_name(STATE_FILE.name + ".previous")
        try:
            value = json.loads(backup.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Backup Controller state must be a JSON object")
        except (OSError, ValueError, json.JSONDecodeError) as backup_error:
            raise RuntimeError(
                "Controller state and its last-known-good backup are unreadable: "
                f"primary={primary_error}; backup={backup_error}"
            ) from backup_error
        save_json(STATE_FILE, value)
        save_json(
            STATE_FILE.with_name("state_recovery.audit.json"),
            {
                "recovered_at": now(),
                "source": str(backup),
                "primary_error": f"{type(primary_error).__name__}: {primary_error}",
            },
        )
        return value


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
        offline_dry_run: bool = False,
        search_space_result: dict[str, Any] | None = None,
        pause_after_candidate_index: int | None = None,
        topology_feasibility_only: bool = False,
    ):
        runtime_config, self.runtime_identity = resolve_runtime_profile(
            config, KB_ROOT, apply_bindings=False
        )
        loading_config, self.model_loading_profile = resolve_model_loading_profile(
            runtime_config, KB_ROOT
        )
        self.config, self.topology = resolve_topology_profile(loading_config, KB_ROOT)
        validate_runtime_selections(self.config)
        topology_setting = self.config.get("topology", {})
        topology_profiles_value = (
            topology_setting.get("profiles_file")
            or "workflow/continuous/topology_profiles.yaml"
        )
        topology_profiles_path = Path(str(topology_profiles_value))
        if not topology_profiles_path.is_absolute():
            topology_profiles_path = KB_ROOT / topology_profiles_path
        runtime_model = self.runtime_identity.get("model_contract", {})
        model_contract_name = (
            f"{runtime_model.get('variant')}-{runtime_model.get('weight_format')}"
        )
        topology_document = load_topology_document(topology_profiles_path)
        selection_defaults = topology_document.get("selection", {})
        self.topology_plan = build_topology_plan(
            topology_document,
            model_contract=model_contract_name,
            available_nodes=int(
                selection_defaults.get("available_nodes", self.topology["nodes"])
            ),
            npu_per_node=int(
                selection_defaults.get("npu_per_node", self.topology["npu_per_node"])
            ),
        )
        campaign_settings = self.config.get("topology_campaign", {})
        self.topology_campaign_enabled = bool(
            isinstance(campaign_settings, dict)
            and campaign_settings.get("enabled") is True
        )
        self.topology_fixed_mode = bool(
            isinstance(campaign_settings, dict)
            and campaign_settings.get("enabled") is False
        )
        if self.topology_fixed_mode:
            selected_profile = str(self.config.get("topology", {}).get("profile", ""))
            selected_candidate = next(
                (
                    copy.deepcopy(item)
                    for item in self.topology_plan.get("candidates", [])
                    if str(item.get("profile")) == selected_profile
                ),
                None,
            )
            if selected_candidate is None:
                raise ValueError(
                    f"Fixed topology profile is absent from the topology plan: {selected_profile!r}"
                )
            session_baseline = self.config.get("initial_baseline", {})
            if isinstance(session_baseline, dict):
                selected_candidate["session_baseline_label"] = session_baseline.get(
                    "label"
                )
                selected_candidate["session_baseline_definition"] = (
                    session_baseline.get("definition")
                )
            self.topology_plan = {
                "schema_version": "vllmtkb-topology-plan/v1",
                "stage": "fixed_topology_session",
                "decision_owner": "operator_policy",
                "controller_role": "identity_freeze_and_metrics_gate",
                "requires_new_session_per_profile": True,
                "selected_profile": selected_profile,
                "selection_reason": (
                    f"{selected_profile} is operator-frozen for parameter-only "
                    "chain validation; "
                    "the dormant outer Campaign cannot allocate another topology."
                ),
                "candidates": [selected_candidate],
                "eligible_profiles": [selected_profile],
            }
        self.config["topology_plan"] = copy.deepcopy(self.topology_plan)
        config = self.config
        effective_runtime = {
            "topology_profile": self.config.get("topology", {}).get("profile"),
            "search_space_profile": self.config.get("search_space", {}).get("profile"),
            "strategy_profile": self.config.get("strategy", {}).get("profile"),
            "benchmark_profile": self.config.get("benchmark", {}).get("profile"),
            "baseline_definition": self.config.get("initial_baseline", {}).get(
                "definition"
            ),
        }
        effective_bytes = json.dumps(
            effective_runtime, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        self.runtime_identity["effective_selections"] = effective_runtime
        self.runtime_identity["effective_sha256"] = hashlib.sha256(
            effective_bytes
        ).hexdigest()
        self.config["runtime"]["identity"] = copy.deepcopy(self.runtime_identity)
        self.dry_run = dry_run
        self.offline_dry_run = offline_dry_run
        self.search_space_result = search_space_result
        self.pause_after_candidate_index = pause_after_candidate_index
        self.topology_feasibility_only = topology_feasibility_only
        if pause_after_candidate_index is not None and pause_after_candidate_index < 0:
            raise ValueError("pause_after_candidate_index must be >= 0")
        if topology_feasibility_only and pause_after_candidate_index != 0:
            raise ValueError(
                "topology_feasibility_only requires pause_after_candidate_index=0"
            )
        if set(config["baseline"]) != set(config["search_limits"]):
            raise ValueError("baseline and search_limits parameter schemas differ")
        self.candidate_schema = set(config["baseline"])
        self.conditional_search_exclusions = copy.deepcopy(
            config.get("conditional_search_exclusions", [])
        )
        automatic_validation = config.get("automatic_registry_validation")
        self.automatic_registry_validation = (
            copy.deepcopy(automatic_validation)
            if isinstance(automatic_validation, dict)
            else None
        )
        active_injections = (
            self.automatic_registry_validation.get("active_injections", {})
            if self.automatic_registry_validation
            else {}
        )
        self.generic_runtime_injections = {
            str(name): copy.deepcopy(injection)
            for name, injection in active_injections.items()
            if name not in ALL_PARAM_TO_ENV
        }
        unknown_parameters = sorted(
            self.candidate_schema
            - set(ALL_PARAM_TO_ENV)
            - set(self.generic_runtime_injections)
        )
        if unknown_parameters:
            raise ValueError(
                f"No runtime injection contract for parameters: {unknown_parameters}"
            )
        self.param_to_env = {
            name: ALL_PARAM_TO_ENV[name]
            for name in config["baseline"]
            if name in ALL_PARAM_TO_ENV
        }
        self.automatic_compatibility = None
        if self.automatic_registry_validation:
            self.automatic_compatibility = CompatibilityValidator(
                scenario=self.automatic_registry_validation["scenario"],
                policy=self.automatic_registry_validation["compatibility_policy"],
            )
        self.remote_host = config["remote_host"]
        self.remote_transport = str(config.get("remote_transport", "native_ssh"))
        if self.remote_transport not in {"native_ssh", "paramiko", "local"}:
            raise ValueError(f"Unsupported remote_transport={self.remote_transport!r}")
        self.operation_mode = str(config.get("operation_mode", "windows_remote"))
        if self.remote_transport == "local" and self.operation_mode != "server_autonomous":
            raise ValueError(
                "remote_transport=local is restricted to operation_mode=server_autonomous"
            )
        if self.operation_mode == "server_autonomous" and self.remote_transport != "local":
            raise ValueError("server_autonomous mode requires remote_transport=local")
        self._paramiko_client_cache: Any | None = None
        self.remote_project = config["remote_project"]
        self.allowed_write_root = str(
            config.get("autonomous", {}).get("allowed_write_root", "")
        ).strip()
        if self.remote_transport == "local":
            if os.name == "nt":
                raise RuntimeError("Local transport is supported only on the Linux server")
            if not self.allowed_write_root:
                raise ValueError(
                    "server_autonomous mode requires autonomous.allowed_write_root"
                )
            project = Path(self.remote_project).resolve()
            allowed = Path(self.allowed_write_root).resolve()
            if not project.is_relative_to(allowed):
                raise ValueError(
                    "server_autonomous remote_project must stay under allowed_write_root"
                )
        self.remote_auto = f"{self.remote_project}/workflow/auto"
        legacy_aligned = config.get("benchmark", {}).get("aligned_l1", {})
        deployment = {
            "model_path": legacy_aligned.get("tokenizer", "/models/share/GLM-5.2-w8a8"),
            "served_model_name": legacy_aligned.get("served_model", "glm-5"),
            "service_port": legacy_aligned.get("service_port", 8000),
            "quantization": "ascend",
            "network_interface": "bond4.3000",
            "vllm_compat_version": "0.21.0",
            "init_env_script": "/models/share/init_env.sh",
            "cann_env_script": "/usr/local/Ascend/cann/set_env.sh",
            **dict(config.get("deployment", {})),
        }
        self.deployment = {
            name: str(value).strip() for name, value in deployment.items()
        }
        model_contract = self.runtime_identity.get("model_contract", {})
        self.runtime_guardrail = (
            f"platform=Ascend, model_family={model_contract.get('family')}, "
            f"model_variant={model_contract.get('variant')}, "
            f"weight_format={model_contract.get('weight_format')}, "
            f"topology_profile={self.config['topology']['profile']}, "
            f"DP={self.topology['data_parallel_size']}, "
            f"TP={self.topology['tensor_parallel_size']}, "
            f"nodes={self.topology['nodes']}, "
            f"NPU_per_node={self.topology['npu_per_node']}"
        )
        empty_deployment = [
            name for name, value in self.deployment.items() if not value
        ]
        if empty_deployment:
            raise ValueError(
                "deployment settings cannot be empty: " + ", ".join(empty_deployment)
            )
        for name, value in self.deployment.items():
            if "\n" in value or "\r" in value:
                raise ValueError(f"deployment.{name} contains a newline")
        for name in ("model_path", "init_env_script", "cann_env_script"):
            if not PurePosixPath(self.deployment[name]).is_absolute():
                raise ValueError(f"deployment.{name} must be an absolute server path")
        if not self.deployment["service_port"].isdigit() or not (
            1 <= int(self.deployment["service_port"]) <= 65535
        ):
            raise ValueError("deployment.service_port must be between 1 and 65535")
        self.config["deployment"] = dict(self.deployment)
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
        agent_settings = dict(self.agent_config.get("settings", {}))
        self.max_agent_protocol_retries = int(
            agent_settings.get("max_protocol_retries", 2)
        )
        if not 0 <= self.max_agent_protocol_retries <= 5:
            raise ValueError("Agent max_protocol_retries must be between 0 and 5")
        self.max_controller_recovery_attempts = int(
            config.get("max_controller_recovery_attempts", 3)
        )
        if not 1 <= self.max_controller_recovery_attempts <= 10:
            raise ValueError("max_controller_recovery_attempts must be between 1 and 10")
        recovery_policy = dict(config.get("failure_recovery", {}))
        self.hard_terminal_only = bool(
            recovery_policy.get("hard_terminal_only", False)
        )
        self.max_same_candidate_retries = int(
            recovery_policy.get("same_candidate_retries", 4)
        )
        self.max_agent_diagnostic_retries = int(
            recovery_policy.get("agent_diagnostic_retries", 1)
        )
        self.max_parameter_failure_adjustments = int(
            recovery_policy.get("parameter_adjustments", 3)
        )
        self.max_total_failure_recovery_rounds = int(
            recovery_policy.get("total_recovery_rounds", 6)
        )
        self.max_recovery_parameter_changes = int(
            recovery_policy.get("max_recovery_parameter_changes", 2)
        )
        if not 1 <= self.max_recovery_parameter_changes <= 4:
            raise ValueError(
                "failure_recovery.max_recovery_parameter_changes must be between 1 and 4"
            )
        raw_recovery_parameters = recovery_policy.get("recovery_parameters", {})
        if not isinstance(raw_recovery_parameters, dict):
            raise ValueError("failure_recovery.recovery_parameters must be a mapping")
        self.recovery_parameter_registry: dict[str, dict[str, Any]] = {}
        self.runtime_recovery_values: dict[str, Any] = {}
        self.recovery_runtime_injections: dict[str, dict[str, Any]] = {}
        for raw_name, raw_definition in raw_recovery_parameters.items():
            name = str(raw_name)
            if not isinstance(raw_definition, dict):
                raise ValueError(f"Recovery parameter {name!r} must be a mapping")
            allowed_values = copy.deepcopy(raw_definition.get("allowed_values", []))
            if not isinstance(allowed_values, list) or not allowed_values:
                raise ValueError(
                    f"Recovery parameter {name!r} requires non-empty allowed_values"
                )
            initial_value = copy.deepcopy(raw_definition.get("initial_value"))
            if initial_value not in allowed_values:
                raise ValueError(
                    f"Recovery parameter {name!r} initial_value is outside allowed_values"
                )
            injection = copy.deepcopy(raw_definition.get("injection"))
            if not isinstance(injection, dict) or injection.get("kind") != "json_path":
                raise ValueError(
                    f"Recovery parameter {name!r} requires a json_path injection"
                )
            path = injection.get("path")
            if not isinstance(path, list) or not path or not all(
                isinstance(part, str) and part for part in path
            ):
                raise ValueError(
                    f"Recovery parameter {name!r} has an invalid injection path"
                )
            self.recovery_parameter_registry[name] = {
                "allowed_values": allowed_values,
                "initial_value": initial_value,
                "injection": injection,
            }
            self.runtime_recovery_values[name] = copy.deepcopy(initial_value)
            self.recovery_runtime_injections[name] = injection
        overlap = set(self.generic_runtime_injections) & set(
            self.recovery_runtime_injections
        )
        if overlap:
            raise ValueError(
                "Recovery-only parameters must not also be Active Search parameters: "
                f"{sorted(overlap)}"
            )
        for name, value, upper in (
            ("same_candidate_retries", self.max_same_candidate_retries, 8),
            ("agent_diagnostic_retries", self.max_agent_diagnostic_retries, 3),
            ("parameter_adjustments", self.max_parameter_failure_adjustments, 6),
            ("total_recovery_rounds", self.max_total_failure_recovery_rounds, 12),
        ):
            if not 0 <= value <= upper:
                raise ValueError(f"failure_recovery.{name} must be between 0 and {upper}")
        self.mtp_draft_model = str(config.get("mtp_draft_model", "")).strip()
        self.model_loading = copy.deepcopy(self.model_loading_profile)
        self.model_loading_backend = str(self.model_loading.get("backend", "dtfs_page_cache"))
        self.model_load_format = str(self.model_loading.get("load_format", "auto"))
        self.require_rfork_transfer = bool(
            self.model_loading.get("require_transfer_hit", False)
        )
        self.safetensors_load_strategy = str(
            self.model_loading.get("safetensors_load_strategy", "prefetch")
        ).strip()
        self.safetensors_prefetch_num_threads = int(
            self.model_loading.get("safetensors_prefetch_num_threads", 8)
        )
        self.safetensors_prefetch_block_size = int(
            self.model_loading.get("safetensors_prefetch_block_size", 16 * 1024 * 1024)
        )
        self.safetensors_prefetch_mode = str(
            self.model_loading.get("safetensors_prefetch_mode", "node_blocking")
        ).strip()
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
        if self.safetensors_prefetch_mode not in {
            "node_blocking",
            "vllm_background",
            "disabled",
        }:
            raise ValueError(
                "model_loading.safetensors_prefetch_mode must be node_blocking "
                "vllm_background, or disabled"
            )
        self.model_loader_extra_config: dict[str, Any] = {}
        if self.model_loading_backend == "rfork":
            self.model_loader_extra_config = {
                name: self.model_loading[name]
                for name in (
                    "model_url",
                    "model_deploy_strategy_name",
                    "rfork_scheduler_url",
                    "rfork_seed_timeout_sec",
                    "rfork_seed_key_separator",
                )
                if name in self.model_loading
            }
        self.lab = config.get("lab", {})
        autonomous_lease_wait = (
            1800 if self.operation_mode == "server_autonomous" else 0
        )
        autonomous_submission_retries = (
            6 if self.operation_mode == "server_autonomous" else 0
        )
        self.lease_readiness_wait_seconds = int(
            self.lab.get("readiness_wait_seconds", autonomous_lease_wait)
        )
        self.lease_readiness_poll_seconds = int(
            self.lab.get("readiness_poll_seconds", 30)
        )
        self.lease_submission_retry_limit = int(
            self.lab.get(
                "submission_readiness_retry_limit",
                autonomous_submission_retries,
            )
        )
        if self.lease_readiness_wait_seconds < 0:
            raise ValueError("lab.readiness_wait_seconds cannot be negative")
        if not 1 <= self.lease_readiness_poll_seconds <= 300:
            raise ValueError(
                "lab.readiness_poll_seconds must be between 1 and 300"
            )
        if not 0 <= self.lease_submission_retry_limit <= 60:
            raise ValueError(
                "lab.submission_readiness_retry_limit must be between 0 and 60"
            )
        image_setting = self.config.get("image_identity", {})
        if not isinstance(image_setting, dict):
            raise ValueError("image_identity must be a mapping")
        frozen_manifest = image_setting.get("resolved_manifest")
        frozen_approval = image_setting.get("resolved_activation")
        if isinstance(frozen_manifest, dict) and isinstance(frozen_approval, dict):
            self.image_manifest = copy.deepcopy(frozen_manifest)
            self.activation_approval = copy.deepcopy(frozen_approval)
            self.image_manifest_path = str(image_setting.get("manifest", "frozen"))
            self.activation_path = str(image_setting.get("activation", "frozen"))
        else:
            manifest_setting = Path(
                str(image_setting.get("manifest", IMAGE_MANIFEST_FILE))
            )
            activation_setting = Path(
                str(image_setting.get("activation", ACTIVATION_FILE))
            )
            if not manifest_setting.is_absolute():
                manifest_setting = KB_ROOT / manifest_setting
            if not activation_setting.is_absolute():
                activation_setting = KB_ROOT / activation_setting
            self.image_manifest = load_yaml(manifest_setting)
            self.activation_approval = load_yaml(activation_setting)
            self.image_manifest_path = str(manifest_setting)
            self.activation_path = str(activation_setting)
            self.config["image_identity"] = {
                "manifest": str(image_setting.get("manifest", manifest_setting)),
                "activation": str(image_setting.get("activation", activation_setting)),
                "resolved_manifest": copy.deepcopy(self.image_manifest),
                "resolved_activation": copy.deepcopy(self.activation_approval),
            }
        target = self.image_manifest.get("target_image", {})
        versions = self.image_manifest.get("versions", {})
        self.image_identity = {
            "reference": f"{target.get('repository', '')}:{target.get('tag', '')}",
            "digest": target.get("digest"),
            "vllm_commit": versions.get("vllm", {}).get("commit"),
            "vllm_ascend_commit": versions.get("vllm_ascend", {}).get("commit"),
        }
        if not all(self.image_identity.values()):
            raise ValueError(f"Incomplete image identity in {self.image_manifest_path}")
        self.measurement_policy = config.get("measurement_policy", {})
        benchmark_settings = dict(config.get("benchmark", {}))
        if not benchmark_settings.get("profile"):
            raise ValueError("benchmark.profile is required")
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
        if self.benchmark_mode not in {
            "aligned_l1",
            "vllm_bench_serve",
            "custom_adapter",
        }:
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
        self.config.setdefault("benchmark", {})["profile"] = self.benchmark_profile_name
        self.config["benchmark"][definition_key] = dict(definition)
        self.config["benchmark"]["resolved_profile"] = dict(self.benchmark_profile)
        self.benchmark_identity = {
            "schema_version": "vllmtkb-benchmark-identity/v1",
            "profile": self.benchmark_profile_name,
            "mode": self.benchmark_mode,
            "definition": dict(definition),
        }
        identity_bytes = json.dumps(
            self.benchmark_identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.benchmark_identity["sha256"] = hashlib.sha256(identity_bytes).hexdigest()
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
            expected_formal_cases = int(aligned.get("expected_formal_cases", 12))
            if expected_formal_cases < 1:
                raise ValueError("aligned L1 expected_formal_cases must be positive")
            target_seconds = int(aligned.get("target_benchmark_seconds", 600))
            if not 60 <= target_seconds <= 43200:
                raise ValueError(
                    "aligned L1 target_benchmark_seconds must be 60..43200"
                )
        elif self.benchmark_mode == "vllm_bench_serve":
            public = self.benchmark[self.benchmark_mode]
            required = ("input_tokens", "output_tokens", "num_prompts", "request_rate")
            missing = [field for field in required if public.get(field) is None]
            if missing:
                raise ValueError(
                    f"benchmark.vllm_bench_serve is missing required fields: {missing}"
                )
            if any(int(public[field]) < 1 for field in required[:3]):
                raise ValueError("public vLLM benchmark token and prompt counts must be positive")
        elif self.benchmark_mode == "custom_adapter":
            custom = self.benchmark[self.benchmark_mode]
            adapter_path = PurePosixPath(str(custom.get("adapter_path", "")))
            roots = [PurePosixPath(str(item)) for item in custom.get("allowlisted_roots", [])]
            if (
                not str(adapter_path)
                or adapter_path.is_absolute()
                or ".." in adapter_path.parts
                or adapter_path.suffix != ".py"
            ):
                raise ValueError(
                    "benchmark.custom_adapter.adapter_path must be a relative .py path"
                )
            if not roots or not any(
                adapter_path == root or root in adapter_path.parents for root in roots
            ):
                raise ValueError(
                    "benchmark.custom_adapter.adapter_path is outside allowlisted_roots"
                )
            local_adapter = (KB_ROOT / Path(*adapter_path.parts)).resolve()
            if not local_adapter.is_file() or not local_adapter.is_relative_to(
                KB_ROOT.resolve()
            ):
                raise ValueError(
                    "benchmark.custom_adapter.adapter_path does not exist in the project"
                )
            timeout_seconds = int(custom.get("timeout_seconds", 3600))
            if not 1 <= timeout_seconds <= 43200:
                raise ValueError("custom benchmark timeout_seconds must be 1..43200")
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
            raise ValueError(
                f"Strategy {self.strategy_profile_name!r} is not integrated"
            )
        strategy_measurement_policy = self.strategy_profile.get("measurement_policy")
        if strategy_measurement_policy is not None:
            if not isinstance(strategy_measurement_policy, dict):
                raise ValueError("strategy measurement_policy must be a mapping")
            self.measurement_policy = deep_merge(
                self.measurement_policy,
                strategy_measurement_policy,
            )
            # Freeze the effective policy into the Session. A later edit to the
            # profile registry cannot silently change an in-flight experiment.
            self.config["measurement_policy"] = copy.deepcopy(
                self.measurement_policy
            )
        self.config.setdefault("strategy", {})["profile"] = self.strategy_profile_name
        self.config["strategy"]["resolved_profile"] = dict(self.strategy_profile)
        # Keep the legacy policy label synchronized with the selected frozen
        # Strategy Profile. It is still included in Agent context and audit
        # records, so leaving the config default here would label V3 as V2.
        self.change_policy["strategy_version"] = self.strategy_profile_name
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
        self.derived_parameter_rules = deep_merge(
            {
                # A positive MTP depth cannot launch without its method. This
                # field is a mechanical runtime companion, not an independent
                # optimization axis with its own portrait.
                "speculative_config__method": {
                    "drivers": ["num_speculative_tokens"]
                }
            },
            self.change_policy.get("derived_parameters", {}),
        )
        self.max_candidate_reselections = int(
            self.change_policy.get("max_candidate_reselections", 2)
        )
        if not 0 <= self.max_candidate_reselections <= 5:
            raise ValueError("max_candidate_reselections must be between 0 and 5")
        if not 1 <= self.max_parameters_per_round <= len(self.candidate_schema):
            raise ValueError("max_parameters_per_round is outside the candidate schema")
        if self.max_grid_steps_per_parameter < 1 or self.max_total_grid_steps < 1:
            raise ValueError("change-policy grid step budgets must be positive")
        if self.execution_mode not in {"ktp", "ktp_lab", "executor_adapter"}:
            raise ValueError(f"Unsupported execution_mode={self.execution_mode!r}")
        if self.execution_mode == "ktp_lab" and not self.lab.get("lease_name"):
            raise ValueError("lab.lease_name is required in ktp_lab mode")
        self.executor_adapter, self.executor_identity = resolve_executor_adapter(
            self.config, KB_ROOT
        )
        if (
            self.execution_mode == "executor_adapter"
            and self.benchmark_mode == "aligned_l1"
            and self.executor_adapter is not None
            and not self.executor_adapter.supports("start_benchmark")
        ):
            raise ValueError(
                "aligned_l1 requires executor_adapter.capabilities.start_benchmark=true"
            )
        self.config["executor_identity"] = copy.deepcopy(self.executor_identity)
        self.sidecar_settings = config.get("sidecars", {})
        self.sidecars_enabled = bool(self.sidecar_settings.get("enabled", False))
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
            str(Path(value).expanduser()) for value in resolved.get("identityfile", [])
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
            'exit "$__continuous_rc"'
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
        output = output[: match.start()].strip()
        if remote_rc != 0:
            details = "\n".join(part for part in (output, errors) if part)
            raise RuntimeError(
                details or f"Remote command failed ({remote_rc}): {command}"
            )
        return output

    def ssh(self, command: str, *, timeout: int = 120) -> str:
        if self.remote_transport == "local":
            result = run_process(["bash", "-lc", command], timeout=timeout)
            if result.returncode != 0:
                details = "\n".join(
                    part for part in (result.stdout.strip(), result.stderr.strip()) if part
                )
                raise RuntimeError(
                    details or f"Local server command failed ({result.returncode}): {command}"
                )
            return result.stdout.strip()
        if self.remote_transport == "paramiko":
            return self._ssh_paramiko(command, timeout=timeout)
        marker = "__CONTINUOUS_REMOTE_RC__="
        wrapped = (
            "{\n"
            f"{command}\n"
            "} 2>&1\n"
            "__continuous_rc=$?\n"
            f"printf '\\n{marker}%s\\n' \"$__continuous_rc\"\n"
            'exit "$__continuous_rc"'
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
        output = result.stdout[: match.start()].strip()
        if remote_rc != 0:
            details = "\n".join(
                part for part in (output, result.stderr.strip()) if part
            )
            raise RuntimeError(
                details or f"Remote command failed ({remote_rc}): {command}"
            )
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
            schema["properties"]["changes"]["maxItems"] = (
                self.max_parameters_per_round
            )
            candidate_schema = schema["properties"]["candidate"]
            candidate_schema["required"] = list(self.config["search_limits"])
            candidate_schema["properties"] = candidate_properties
            if filename == "failure_decision.schema.json":
                recovery_schema = schema["properties"]["recovery_changes"]
                recovery_schema["maxItems"] = self.max_recovery_parameter_changes
                recovery_schema["items"]["properties"]["parameter"] = {
                    "type": "string",
                    "enum": sorted(self.recovery_parameter_registry),
                }
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
                    key = json.dumps(relation, ensure_ascii=False, sort_keys=True)
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
                    key = json.dumps(selected, ensure_ascii=False, sort_keys=True)
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
            include_one_hop=bool(self.sidecar_settings.get("include_one_hop", True)),
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
            raise RuntimeError(
                f"Frozen Session runtime rule store is missing: {store_path}"
            )
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
        result = self.validate_selected_portrait_evidence(changes)
        save_yaml(
            round_dir / "06_agent_analysis" / f"{prefix}_parameter_portraits.yaml",
            result,
        )

    def validate_selected_portrait_evidence(
        self,
        changes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Validate evidence before accepting an Agent candidate.

        Keeping this check inside candidate reselection prevents a valid round
        from crashing the whole controller merely because one proposed axis has
        incomplete portrait metadata.
        """
        if not self.portrait_retriever or not changes:
            return {}
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
        changed_names = {
            str(item["parameter"]).removeprefix("--").replace("-", "_")
            for item in changes
        }
        unresolved_changed = [
            name
            for name in unresolved_changed
            if not (
                name in self.derived_parameter_rules
                and any(
                    driver in changed_names
                    for driver in self.derived_parameter_rules[name].get(
                        "drivers", []
                    )
                )
            )
        ]
        if unresolved_changed:
            raise ValueError(
                "No parameter portrait evidence for changed parameters: "
                f"{unresolved_changed}"
            )
        return result

    def refresh_runtime_rules(self, session_dir: Path, round_dir: Path) -> None:
        if not self.runtime_rule_store:
            return
        history_path = round_dir / "06_agent_analysis" / "runtime_rule_history.json"
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

    def failure_schema_path(self, session_dir: Path) -> Path:
        """Add recovery actions to an older Session without widening candidates.

        The candidate schema remains the Session-frozen authority. Only the
        Controller recovery action contract is migrated, into a separate
        audited runtime schema, so an in-flight Session can gain safer recovery
        without silently changing its Search Limits.
        """
        frozen = self.decision_schema_path(session_dir, "failure_decision.schema.json")
        frozen_schema = json.loads(frozen.read_text(encoding="utf-8"))
        current_schema = json.loads(
            (HERE / "failure_decision.schema.json").read_text(encoding="utf-8")
        )
        current_schema["properties"]["candidate"] = frozen_schema["properties"][
            "candidate"
        ]
        current_schema["properties"]["changes"]["maxItems"] = (
            self.max_parameters_per_round
        )
        recovery_schema = current_schema["properties"]["recovery_changes"]
        recovery_schema["maxItems"] = self.max_recovery_parameter_changes
        recovery_schema["items"]["properties"]["parameter"] = {
            "type": "string",
            "enum": sorted(self.recovery_parameter_registry),
        }
        migrated = frozen.with_name("failure_decision.runtime.schema.json")
        save_json(migrated, current_schema)
        save_json(
            migrated.with_name("failure_schema_migration.audit.json"),
            {
                "migrated_at": now(),
                "policy": "agent_owned_failure_recovery_v3",
                "source": str(frozen),
                "candidate_schema_preserved": True,
                "action_contract_source": "current_controller_with_frozen_candidate_schema",
                "added_recovery_parameters": sorted(
                    self.recovery_parameter_registry
                ),
            },
        )
        return migrated

    def agent_schema_path(self, session_dir: Path) -> Path:
        """Apply current Agent change budgets without widening frozen values."""
        frozen = self.decision_schema_path(session_dir, "agent_decision.schema.json")
        frozen_schema = json.loads(frozen.read_text(encoding="utf-8"))
        current_schema = json.loads(
            (HERE / "agent_decision.schema.json").read_text(encoding="utf-8")
        )
        current_schema["properties"]["candidate"] = frozen_schema["properties"][
            "candidate"
        ]
        current_schema["properties"]["changes"]["maxItems"] = (
            self.max_parameters_per_round
        )
        migrated = frozen.with_name("agent_decision.runtime.schema.json")
        save_json(migrated, current_schema)
        save_json(
            migrated.with_name("agent_schema_migration.audit.json"),
            {
                "migrated_at": now(),
                "policy": "agent_owned_selection_contract_v2",
                "source": str(frozen),
                "candidate_schema_preserved": True,
                "max_parameters_per_round": self.max_parameters_per_round,
            },
        )
        return migrated

    def scp_to(self, source: Path, destination: str) -> None:
        if self.remote_transport == "local":
            target = Path(destination).resolve()
            allowed = Path(self.allowed_write_root).resolve()
            if not target.is_relative_to(allowed):
                raise RuntimeError(f"Local upload escapes allowed_write_root: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(
                f"{target.name}.upload-{os.getpid()}-{time.time_ns()}"
            )
            shutil.copyfile(source, temporary)
            if temporary.stat().st_size != source.stat().st_size:
                raise RuntimeError(f"Local upload size mismatch: {source}")
            os.replace(temporary, target)
            return
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
        if self.remote_transport == "local":
            origin = Path(source).resolve()
            allowed = Path(self.allowed_write_root).resolve()
            if not origin.is_relative_to(allowed) or not origin.is_file():
                return False
            temporary = destination.with_name(
                f"{destination.name}.download-{os.getpid()}-{time.time_ns()}"
            )
            shutil.copyfile(origin, temporary)
            if temporary.stat().st_size != origin.stat().st_size:
                return False
            os.replace(temporary, destination)
            return True
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
                if not temporary.exists() or temporary.stat().st_size != remote_size:
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
        run_process(
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
                "executor_identity": state.get(
                    "executor_identity", self.executor_identity
                ),
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
            "performance_impact=high,medium",
            "--show",
            "name,valid_choices,tuning_advice.suggested_values,constraints,category",
            "--format",
            "yaml",
        ]
        result = run_process(command, cwd=KB_ROOT, timeout=120)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(
                result.stderr.strip() or "Knowledge query returned no data"
            )
        output_path.write_text(result.stdout, encoding="utf-8")
        (round_dir / "01_query" / "query_command.txt").write_text(
            subprocess.list2cmdline(command) + "\n", encoding="utf-8"
        )
        return output_path

    def candidate_env(
        self,
        label: str,
        candidate: dict[str, Any],
        *,
        launch_profile: str | None = None,
    ) -> str:
        lines = [f"ROUND_LABEL={shlex.quote(label)}"]
        initial_baseline = self.config.get("initial_baseline", {})
        initial_label = str(initial_baseline.get("label", "a0"))
        if launch_profile is None:
            launch_profile = (
                str(initial_baseline.get("launch_profile", "explicit_candidate"))
                if label == initial_label
                else "explicit_candidate"
            )
        if launch_profile not in {B0_LAUNCH_PROFILE, "explicit_candidate"}:
            raise ValueError(f"Unsupported launch profile: {launch_profile!r}")
        lines.append(f"LAUNCH_PROFILE={shlex.quote(launch_profile)}")
        topology_env = {
            "TOPOLOGY_PROFILE": self.config["topology"]["profile"],
            "TOPOLOGY_NODES": self.topology["nodes"],
            "NPU_PER_NODE": self.topology["npu_per_node"],
            "DATA_PARALLEL_SIZE": self.topology["data_parallel_size"],
            "DATA_PARALLEL_SIZE_LOCAL": self.topology["data_parallel_size_local"],
            "TENSOR_PARALLEL_SIZE": self.topology["tensor_parallel_size"],
            "DATA_PARALLEL_RPC_PORT": self.topology["data_parallel_rpc_port"],
            "WORKER_DATA_PARALLEL_START_RANK": self.topology[
                "worker_data_parallel_start_rank"
            ],
            "WORKER_REPLICAS": self.topology["worker_replicas"],
            "EXECUTOR_REMOTE_CONTRACT": self.topology["resolved_executor"][
                "remote_contract"
            ],
        }
        for env_name, value in topology_env.items():
            lines.append(f"{env_name}={shlex.quote(str(value))}")
        for key, env_name in self.param_to_env.items():
            value = candidate[key]
            if isinstance(value, bool):
                text = "true" if value else "false"
            elif value is None or isinstance(value, (list, dict)):
                text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            else:
                text = str(value)
            lines.append(f"{env_name}={shlex.quote(text)}")
        runtime_injections = {
            **self.generic_runtime_injections,
            **self.recovery_runtime_injections,
        }
        if runtime_injections:
            runtime_values = {
                **candidate,
                **self.runtime_recovery_values,
            }
            payload = compile_generic_runtime_payload(
                runtime_values, runtime_injections
            )
            encoded = base64.b64encode(
                json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).decode("ascii")
            lines.append("RUNTIME_INJECTION_MODE=generated_v1")
            lines.append("RUNTIME_INJECTION_PAYLOAD_B64=" + shlex.quote(encoded))
        else:
            lines.append("RUNTIME_INJECTION_MODE=native_v1")
        lines.append(
            "RUNTIME_RECOVERY_PARAMETERS_JSON="
            + shlex.quote(
                json.dumps(
                    self.runtime_recovery_values,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        )
        lines.append(f"MTP_DRAFT_MODEL_PATH={shlex.quote(self.mtp_draft_model)}")
        deployment_env = {
            "MODEL_PATH": self.deployment["model_path"],
            "SERVED_MODEL_NAME": self.deployment["served_model_name"],
            "SERVICE_PORT": self.deployment["service_port"],
            "MODEL_QUANTIZATION": self.deployment["quantization"],
            "NIC_NAME": self.deployment["network_interface"],
            "VLLM_COMPAT_VERSION": self.deployment["vllm_compat_version"],
            "INIT_ENV_SCRIPT": self.deployment["init_env_script"],
            "CANN_ENV_SCRIPT": self.deployment["cann_env_script"],
            "RUNTIME_CACHE_ROOT": self.deployment.get("cache_root", ""),
            "FIXED_CLI_ARGS_JSON": json.dumps(
                self.deployment.get("fixed_cli_args", []),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "FIXED_ADDITIONAL_CONFIG_JSON": json.dumps(
                self.deployment.get("fixed_additional_config", {}),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "FIXED_ENVIRONMENT_JSON": json.dumps(
                self.deployment.get("fixed_environment", {}),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
        for env_name, value in deployment_env.items():
            lines.append(f"{env_name}={shlex.quote(str(value))}")
        lab_output_root = str(
            self.lab.get("output_root", f"{self.remote_auto}/lab_runs")
        )
        lines.append("LAB_OUTPUT_ROOT=" + shlex.quote(lab_output_root))
        lines.append(
            "SAFETENSORS_LOAD_STRATEGY=" + shlex.quote(self.safetensors_load_strategy)
        )
        lines.append(
            "SAFETENSORS_PREFETCH_NUM_THREADS="
            + str(self.safetensors_prefetch_num_threads)
        )
        lines.append(
            "SAFETENSORS_PREFETCH_BLOCK_SIZE="
            + str(self.safetensors_prefetch_block_size)
        )
        lines.append(
            "SAFETENSORS_PREFETCH_MODE="
            + shlex.quote(self.safetensors_prefetch_mode)
        )
        lines.append("MODEL_LOADING_BACKEND=" + shlex.quote(self.model_loading_backend))
        lines.append("MODEL_LOAD_FORMAT=" + shlex.quote(self.model_load_format))
        lines.append(
            "MODEL_LOADER_EXTRA_CONFIG_JSON="
            + shlex.quote(
                json.dumps(
                    self.model_loader_extra_config,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        )
        lines.append(
            "REQUIRE_RFORK_TRANSFER="
            + ("true" if self.require_rfork_transfer else "false")
        )
        lines.append(f"BENCHMARK_MODE={shlex.quote(self.benchmark_mode)}")
        lines.append(f"BENCHMARK_PROFILE={shlex.quote(self.benchmark_profile_name)}")
        lines.append(
            "BENCHMARK_IDENTITY_JSON="
            + shlex.quote(
                json.dumps(self.benchmark_identity, ensure_ascii=False, sort_keys=True)
            )
        )
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
                "BENCHMARK_SUITE_ID": aligned.get("suite_id", "tuning-fixed"),
                "BENCHMARK_EXPECTED_FORMAL_CASES": aligned.get(
                    "expected_formal_cases", 12
                ),
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
                "dataset_manifests_sha256": aligned["dataset_manifests_sha256"],
            }
            lines.append(
                "BENCHMARK_EXPECTED_FINGERPRINT_JSON="
                + shlex.quote(
                    json.dumps(fingerprint, ensure_ascii=False, sort_keys=True)
                )
            )
        elif self.benchmark_mode in {"vllm_bench_serve", "custom_adapter"}:
            definition = self.benchmark[self.benchmark_mode]
            encoded = base64.b64encode(
                json.dumps(
                    definition, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).decode("ascii")
            lines.append("BENCHMARK_DEFINITION_B64=" + shlex.quote(encoded))
        return "\n".join(lines) + "\n"

    def validate_runtime_configuration(self, candidate: dict[str, Any]) -> None:
        self.validate_candidate_invariants(candidate)
        if candidate["num_speculative_tokens"] > 0 and not self.mtp_draft_model:
            raise RuntimeError(
                "mtp_draft_model is empty. Set the exact sliced MTP checkpoint "
                "path supplied by the deployment owner before launching a round."
            )

    def validate_deployment_configuration(self) -> None:
        validate_activation_approval(
            self.image_manifest,
            approval_path=Path(self.activation_path),
            approval=self.activation_approval,
        )
        expected_repository, expected_tag = self.image_identity["reference"].rsplit(
            ":", 1
        )
        if self.benchmark_mode == "aligned_l1":
            benchmark_image = str(self.benchmark["aligned_l1"]["docker_image"])
            if "@sha256:" not in benchmark_image:
                raise RuntimeError(
                    "benchmark.aligned_l1.docker_image must be digest-qualified"
                )
            if (
                str(self.benchmark["aligned_l1"]["served_model"])
                != self.deployment["served_model_name"]
            ):
                raise RuntimeError(
                    "benchmark.aligned_l1.served_model must match "
                    "deployment.served_model_name"
                )
            if int(self.benchmark["aligned_l1"]["service_port"]) != int(
                self.deployment["service_port"]
            ):
                raise RuntimeError(
                    "benchmark.aligned_l1.service_port must match "
                    "deployment.service_port"
                )
        # External schedulers receive the same frozen image/topology identity
        # through the v1 adapter context. Their own manifests are validated by
        # check_ready/submit; ktp YAML rendering remains untouched for legacy
        # modes and is deliberately not imposed on another scheduler.
        if self.execution_mode == "executor_adapter":
            return
        lease = self.render_remote_control_document("lease_loop.yaml")
        if self.execution_mode == "ktp_lab":
            output_root = str(self.lab.get("output_root", ""))
            if not PurePosixPath(output_root).is_absolute():
                raise RuntimeError(
                    "lab.output_root must be an absolute server path in ktp_lab mode"
                )
            if self.remote_transport == "local":
                allowed = Path(self.allowed_write_root).resolve()
                if not Path(output_root).resolve().is_relative_to(allowed):
                    raise RuntimeError(
                        "server_autonomous lab.output_root must stay under "
                        "autonomous.allowed_write_root"
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
        experiment = self.render_remote_control_document("experiment_loop.yaml")
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
        recorded_runtime = state.get("runtime_identity")
        if recorded_runtime is not None and recorded_runtime != self.runtime_identity:
            raise RuntimeError(
                "Controller state runtime-adapter identity differs from the frozen "
                "Session. Start a new Session; do not cross model/image/topology "
                "boundaries during resume."
            )
        recorded_executor = state.get("executor_identity")
        if self.execution_mode == "executor_adapter" and recorded_executor is None:
            raise RuntimeError(
                "Controller state is missing the frozen executor-adapter identity. "
                "Start a new Session; do not resume without scheduler identity."
            )
        if recorded_executor is not None and recorded_executor != self.executor_identity:
            raise RuntimeError(
                "Controller state executor-adapter identity differs from the frozen "
                "Session. Start a new Session; do not change scheduler bridges "
                "during resume."
            )

    def executor_context(self) -> dict[str, Any]:
        """Return the non-secret, frozen contract visible to an external executor."""

        return {
            "api_version": EXECUTOR_ADAPTER_API_VERSION,
            "operation_mode": self.operation_mode,
            "remote_host": self.remote_host,
            "remote_transport": self.remote_transport,
            "remote_project": self.remote_project,
            "remote_auto": self.remote_auto,
            "topology": copy.deepcopy(self.topology),
            "deployment": copy.deepcopy(self.deployment),
            "image_identity": copy.deepcopy(self.image_identity),
            "runtime_identity": copy.deepcopy(self.runtime_identity),
            "benchmark": {
                "profile": self.benchmark_profile_name,
                "mode": self.benchmark_mode,
                "identity": copy.deepcopy(self.benchmark_identity),
            },
            "artifact_contract": {
                "remote_run_template": self.remote_auto + "/runs/{run_id}",
                "artifacts": list(REMOTE_ARTIFACTS),
            },
            "round_timeout_minutes": self.round_timeout_minutes,
            "poll_seconds": self.poll_seconds,
        }

    def validate_candidate_invariants(self, candidate: dict[str, Any]) -> None:
        if set(candidate) != self.candidate_schema:
            missing = sorted(self.candidate_schema - set(candidate))
            extra = sorted(set(candidate) - self.candidate_schema)
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
        elif self.benchmark_mode == "vllm_bench_serve":
            public_benchmark = self.benchmark["vllm_bench_serve"]
            required_tokens = int(public_benchmark["input_tokens"]) + int(
                public_benchmark["output_tokens"]
            )
        else:
            required_tokens = 0
        if required_tokens and candidate["max_model_len"] < required_tokens:
            raise ValueError(
                f"max_model_len={candidate['max_model_len']} is below the benchmark "
                f"input+output requirement {required_tokens}"
            )
        threshold = candidate["long_prefill_token_threshold"]
        if threshold and threshold > candidate["max_num_batched_tokens"]:
            raise ValueError(
                "long_prefill_token_threshold cannot exceed " "max_num_batched_tokens"
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
        resolved_mode = self.config.get("resolved_search_space", {}).get("mode")
        if resolved_mode in {"automated", "curated_registry"}:
            profile = self.config.get("search_space", {}).get("resolved_profile", {})
            scenario_setting = profile.get(
                "scenario",
                self.config.get("automated_search_limits", {}).get(
                    "scenario",
                    "workflow/search_space_compiler/scenario.glm52-a3-aligned-l1.yaml",
                ),
            )
            scenario = load_yaml(KB_ROOT / str(scenario_setting))
            violations = validate_search_space_candidate(candidate, scenario)
            if violations:
                raise ValueError(
                    "Candidate violates compiled search-space constraints: "
                    + ", ".join(violations)
                )
        if self.automatic_registry_validation and self.automatic_compatibility:
            active = set(
                self.config.get("resolved_search_space", {}).get(
                    "active_tunable_parameters", []
                )
            )
            report = validate_trial_candidate(
                candidate={name: candidate[name] for name in active},
                compiled=self.automatic_registry_validation["compiled"],
                scenario=self.automatic_registry_validation["scenario"],
                compatibility=self.automatic_compatibility,
                context_candidate=candidate,
            )
            if not report["valid"]:
                raise ValueError(
                    "Candidate violates automatic-registry validation: "
                    + json.dumps(report["violations"], ensure_ascii=False)
                )
        if candidate.get("eplb_num_redundant_experts", 0) and not candidate.get(
            "enable_eplb", False
        ):
            raise ValueError("eplb_num_redundant_experts requires enable_eplb=true")
        if candidate.get("enable_eplb", False):
            raise ValueError(
                "The upstream --enable-eplb CLI is unsupported by the pinned "
                "Ascend platform. Native dynamic EPLB requires a separately "
                "reviewed additional_config integration."
            )
        if candidate.get("fused_mc2", 0) and not candidate.get(
            "enable_expert_parallel", False
        ):
            raise ValueError("fused_mc2 requires enable_expert_parallel=true")
        if candidate.get("fused_mc2") == 2 and candidate.get(
            "num_speculative_tokens", 0
        ) <= 0:
            raise ValueError("fused_mc2=2 requires speculative decoding")
        if candidate.get("enable_balance_scheduling", False):
            data_parallel_size = int(self.topology["data_parallel_size"])
            if data_parallel_size <= 1:
                raise ValueError("enable_balance_scheduling requires DP > 1")
        draft_eager = candidate.get("speculative_config__enforce_eager")
        if draft_eager is True and candidate.get("num_speculative_tokens", 0) <= 0:
            raise ValueError(
                "speculative draft enforce_eager only applies when speculative decoding is enabled"
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
                str(item.get("id")) for item in rule_evaluation.get("violations", [])
            ]
            raise ValueError(
                "Candidate rejected by frozen runtime rule store: " + ", ".join(ids)
            )

    def prepare_lab(self, *, submit: bool) -> str:
        self.validate_deployment_configuration()
        self.validate_runtime_configuration(self.config["baseline"])
        if self.execution_mode == "executor_adapter":
            assert self.executor_adapter is not None
            result = self.executor_adapter.invoke(
                "prepare",
                context=self.executor_context(),
                payload={"submit": submit},
            )
            return str(result.get("message", "External executor prepare completed."))
        self.ensure_no_blocked_leases()
        self.sync_remote_scripts()
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

    def render_remote_control_document(self, name: str) -> dict[str, Any]:
        """Render server-specific control YAML from the repository template."""
        if name not in {"lease_loop.yaml", "experiment_loop.yaml"}:
            raise ValueError(f"Unsupported remote control document: {name}")
        document = load_yaml(HERE / "remote" / name)
        repository, tag = self.image_identity["reference"].rsplit(":", 1)
        if name == "lease_loop.yaml":
            document["name"] = str(self.lab["lease_name"])
            document["image"] = repository
            document["image_tag"] = tag
        document["min_available"] = self.topology["nodes"]
        rendered_tasks = []
        for task in document.get("tasks", []):
            task_name = str(task.get("name", "")).lower()
            if task_name not in {"master", "worker"}:
                raise ValueError(
                    f"Unsupported remote task role {task_name!r} in {name}"
                )
            script = (
                "run_master_loop.sh" if task_name == "master" else "run_worker_loop.sh"
            )
            task["command"] = f"bash {self.remote_auto}/{script}"
            task["npu"] = self.topology["npu_per_node"]
            if task_name == "worker":
                if self.topology["worker_replicas"] == 0:
                    continue
                task["replicas"] = self.topology["worker_replicas"]
            if name == "experiment_loop.yaml":
                task["image"] = repository
                task["image_tag"] = tag
            rendered_tasks.append(task)
        document["tasks"] = rendered_tasks
        return document

    def sync_remote_scripts(self) -> None:
        remote_source = HERE / "remote"
        self.ssh(f"mkdir -p {shlex.quote(self.remote_auto)}")
        with tempfile.TemporaryDirectory(prefix="vllmtkb-remote-control-") as temporary:
            rendered_root = Path(temporary)
            for name in REMOTE_SCRIPT_NAMES:
                if (
                    self.execution_mode == "executor_adapter"
                    and name in {"lease_loop.yaml", "experiment_loop.yaml"}
                ):
                    continue
                source = remote_source / name
                if not source.is_file():
                    raise RuntimeError(
                        f"Required remote script is missing locally: {source}"
                    )
                if name in {"lease_loop.yaml", "experiment_loop.yaml"}:
                    source = rendered_root / name
                    save_yaml(source, self.render_remote_control_document(name))
                self.scp_to(source, f"{self.remote_auto}/{name}")
        if self.benchmark_mode == "custom_adapter":
            adapter = PurePosixPath(
                str(self.benchmark["custom_adapter"]["adapter_path"])
            )
            remote_adapter = PurePosixPath(self.remote_project) / adapter
            self.ssh(f"mkdir -p {shlex.quote(str(remote_adapter.parent))}")
            self.scp_to(KB_ROOT / Path(*adapter.parts), str(remote_adapter))

    def lease_status(self) -> str:
        lease_name = str(self.lab["lease_name"])
        try:
            return self.ssh(
                f"cd {shlex.quote(self.remote_project)} && "
                f"ktp-lab status --lease {shlex.quote(lease_name)} 2>&1 || true",
                timeout=60,
            )
        except Exception as exc:
            raise RuntimeError(
                f"SSH or persistent lease {lease_name!r} is not available. "
                "Verify SSH connectivity first; if the Lease is absent, run "
                ".\\scripts\\prepare-remote.ps1 from the repository root, then "
                "wait for the Lease to become ready."
            ) from exc

    def ensure_lab_available(self) -> str:
        self.ensure_no_blocked_leases()
        lease_name = str(self.lab["lease_name"])
        output = self.lease_status()
        normalized = output.lower()
        nodes = self.topology["nodes"]
        resource_active = bool(re.search(r"\bresource\s+status=active\b", normalized))
        nodes_ready = bool(
            re.search(rf"\bnodes={nodes}/{nodes}\s+ready\b", normalized)
        )
        service_idle = bool(re.search(rf"\bstatus\s+idle={nodes}\b", normalized))
        # A brand-new persistent lease has no service slot until its first
        # `ktp-lab run`.  Treat that state as available; subsequent runs must
        # see both nodes idle.  Running/partial slots match neither condition
        # and therefore remain fail-closed.
        fresh_without_slots = bool(re.search(r"\bslots\s+none\b", normalized))
        if not (
            resource_active and nodes_ready and (service_idle or fresh_without_slots)
        ):
            raise LeaseNotReadyError(
                f"Persistent lease {lease_name!r} is not idle and ready:\n{output}"
            )
        return output

    def lease_readiness_deadline(self) -> float | None:
        if self.lease_readiness_wait_seconds <= 0:
            return None
        return time.monotonic() + self.lease_readiness_wait_seconds

    def wait_for_lab_available(self, *, deadline: float | None = None) -> str:
        """Wait for a transient Lease outage without hiding hard safety errors."""
        if deadline is None:
            deadline = self.lease_readiness_deadline()
        first_failure: str | None = None
        attempts = 0
        while True:
            try:
                output = self.ensure_lab_available()
                if attempts:
                    log(
                        f"Persistent Lease recovered after {attempts} readiness "
                        "checks; continuing the pending round."
                    )
                return output
            except LeaseNotReadyError as exc:
                attempts += 1
                first_failure = first_failure or str(exc)
                if deadline is None:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LeaseNotReadyError(
                        "Persistent Lease did not recover within "
                        f"{self.lease_readiness_wait_seconds} seconds. "
                        "The pending round was not submitted. First status:\n"
                        f"{first_failure}\nLast status:\n{exc}"
                    ) from exc
                if attempts == 1 or attempts % 10 == 0:
                    log(
                        "Persistent Lease is temporarily not Ready; waiting up to "
                        f"{math.ceil(remaining)} more seconds before safely pausing."
                    )
                time.sleep(min(self.lease_readiness_poll_seconds, remaining))

    @staticmethod
    def is_transient_protocol_readiness_error(exc: BaseException) -> bool:
        return "resource admission requires control protocol v2 workers" in str(exc)

    def run_lab_submission_with_readiness_retry(
        self,
        command: str,
        *,
        deadline: float | None,
    ) -> str:
        """Retry only the pre-admission protocol error caused by a heartbeat race."""
        retries = 0
        while True:
            try:
                return self.ssh(command, timeout=180)
            except RuntimeError as exc:
                if (
                    not self.is_transient_protocol_readiness_error(exc)
                    or retries >= self.lease_submission_retry_limit
                ):
                    raise
                retries += 1
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise LeaseNotReadyError(
                            "Lease readiness deadline expired while ktp-lab was "
                            "waiting for control protocol v2 worker heartbeats. "
                            "The pending round was not admitted."
                        ) from exc
                    time.sleep(min(self.lease_readiness_poll_seconds, remaining))
                log(
                    "ktp-lab could not see fresh control protocol v2 worker "
                    f"heartbeats during admission; readiness retry {retries}/"
                    f"{self.lease_submission_retry_limit}."
                )
                self.wait_for_lab_available(deadline=deadline)

    def ensure_no_blocked_leases(self) -> None:
        """Prevent the autonomous mode from overlapping declared main-chain leases."""
        if self.operation_mode != "server_autonomous":
            return
        for lease_name in self.lab.get("blocked_lease_names", []):
            lease_name = str(lease_name).strip()
            if not lease_name or lease_name == str(self.lab.get("lease_name", "")):
                continue
            output = self.ssh(
                f"ktp-lab status --lease {shlex.quote(lease_name)} 2>&1 || true",
                timeout=60,
            )
            if re.search(r"\bresource\s+status=active\b", output.lower()):
                raise RuntimeError(
                    "Autonomous mode is isolated and will not compete with the "
                    f"active main-chain Lease {lease_name!r}."
                )

    def check_ready(self, *, require_idle_lease: bool = True) -> str:
        """Run a read-only end-to-end launch preflight."""
        self.validate_deployment_configuration()
        self.validate_runtime_configuration(self.config["baseline"])
        if self.execution_mode == "executor_adapter":
            assert self.executor_adapter is not None
            result = self.executor_adapter.invoke(
                "check_ready",
                context=self.executor_context(),
                payload={"require_idle": require_idle_lease},
            )
            return str(result.get("message", "External executor is ready."))
        if require_idle_lease:
            return self.ensure_lab_available()
        self.ensure_no_blocked_leases()
        output = self.lease_status()
        normalized = output.lower()
        nodes = self.topology["nodes"]
        if not (
            re.search(r"\bresource\s+status=active\b", normalized)
            and re.search(rf"\bnodes={nodes}/{nodes}\s+ready\b", normalized)
        ):
            raise RuntimeError(
                f"The active Session Lease is reachable but its {nodes}-node "
                f"resource is not ready:\n{output}"
            )
        return output

    def submit_lab(
        self,
        round_dir: Path,
        label: str,
        candidate: dict[str, Any],
        *,
        dry_run: bool,
        launch_profile: str | None = None,
    ) -> tuple[str | None, str]:
        self.validate_runtime_configuration(candidate)
        env_path = round_dir / "02_parameters" / "candidate.env"
        env_path.write_text(
            self.candidate_env(
                label, candidate, launch_profile=launch_profile
            ),
            encoding="utf-8",
            newline="\n",
        )
        if dry_run:
            # The persistent lease is created once and deliberately stays
            # active between rounds. Re-running `lease create` as a dry-run
            # therefore fails on the existing name even when the lease is
            # healthy and idle. Validate the already-created lease instead;
            # this remains non-submitting and catches the same readiness
            # conditions used by a real round.
            output = (
                "Lease preflight intentionally skipped by --offline-dry-run."
                if self.offline_dry_run
                else self.ensure_lab_available()
            )
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

        readiness_deadline = self.lease_readiness_deadline()
        self.wait_for_lab_available(deadline=readiness_deadline)
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
        submission_command = (
            f"cd {shlex.quote(self.remote_project)} && "
            f"ktp-lab run --lease {shlex.quote(lease_name)} "
            f"--run-id {shlex.quote(run_id)} "
            f"--output-root {shlex.quote(output_root)}"
        )
        output = self.run_lab_submission_with_readiness_retry(
            submission_command,
            deadline=readiness_deadline,
        )
        (round_dir / "03_submission" / "submit_output.txt").write_text(
            output + "\n", encoding="utf-8"
        )
        self.scp_from(
            f"{remote_run}/task.yaml", round_dir / "03_submission" / "task.yaml"
        )
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
        if decision.get("exploration_intent") == "none":
            raise ValueError("A changed candidate requires a non-none exploration_intent")
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
            raise ValueError(
                "Each changed parameter requires concrete knowledge/log evidence"
            )
        if len(constraint_checks) < change_count:
            raise ValueError(
                "Each changed parameter requires an explicit constraint check"
            )
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
        if decision.get("exploration_intent") not in {None, "none"}:
            raise ValueError("A no-change decision requires exploration_intent=none")

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

    @staticmethod
    def hierarchical_probe_measurement_budget(probe: dict[str, Any]) -> tuple[int, int]:
        defaults = {
            "mtp_enablement": (1, 2),
            "moe_communication": (1, 2),
            "scheduler_capacity": (2, 3),
            "compilation_graph": (1, 2),
            "ascend_communication_refinement": (1, 2),
        }
        default_minimum, default_maximum = defaults.get(
            str(probe.get("name")), (1, 2)
        )
        minimum = int(probe.get("minimum_successful_measurements", default_minimum))
        maximum = int(probe.get("maximum_successful_measurements", default_maximum))
        if not 1 <= minimum <= maximum <= 5:
            raise ValueError("hierarchical probe measurement budget is invalid")
        return minimum, maximum

    def hierarchical_search_state(
        self,
        history: list[dict[str, Any]],
        ordered_probes: list[dict[str, Any]],
        hierarchy: dict[str, Any],
    ) -> dict[str, Any]:
        """Replay measured rounds through adaptive layers, then rank revisits."""
        floor = float(hierarchy.get("promising_incremental_gain_percent", -3.0))
        probe_index = 0
        measurements_in_probe = 0
        observations: list[dict[str, Any]] = []
        cross_layer_rounds = 0
        for position, item in enumerate(history[1:], start=1):
            if probe_index >= len(ordered_probes):
                cross_layer_rounds += 1
                continue
            probe = ordered_probes[probe_index]
            measurements_in_probe += 1
            current_score = self.primary_performance_score(item)
            prior_history = history[:position]
            anchor = self.best_accepted_anchor(prior_history)
            anchor_item = next(
                (
                    prior
                    for prior in prior_history
                    if prior.get("round") == (anchor or {}).get("round")
                ),
                prior_history[0] if prior_history else None,
            )
            anchor_score = (
                self.primary_performance_score(anchor_item)
                if anchor_item is not None
                else None
            )
            incremental_gain = (
                (current_score / anchor_score - 1.0) * 100.0
                if current_score is not None
                and anchor_score is not None
                and anchor_score > 0
                else None
            )
            observations.append(
                {
                    "probe_index": probe_index,
                    "probe": probe.get("name"),
                    "round": item.get("round"),
                    "measurement_number": measurements_in_probe,
                    "incremental_gain_vs_entry_anchor_percent": incremental_gain,
                }
            )
            minimum, maximum = self.hierarchical_probe_measurement_budget(probe)
            should_exit = measurements_in_probe >= maximum or (
                measurements_in_probe >= minimum
                and (incremental_gain is None or incremental_gain < floor)
            )
            if should_exit:
                probe_index += 1
                measurements_in_probe = 0

        layer_scores: list[dict[str, Any]] = []
        for index, probe in enumerate(ordered_probes):
            gains = [
                item["incremental_gain_vs_entry_anchor_percent"]
                for item in observations
                if item["probe_index"] == index
                and item["incremental_gain_vs_entry_anchor_percent"] is not None
            ]
            if gains:
                layer_scores.append(
                    {
                        "probe_index": index,
                        "probe": probe.get("name"),
                        "best_incremental_gain_percent": max(gains),
                    }
                )
        ranked_revisits = sorted(
            layer_scores,
            key=lambda item: item["best_incremental_gain_percent"],
            reverse=True,
        )
        promising_revisits = [
            item
            for item in ranked_revisits
            if item["best_incremental_gain_percent"] >= floor
        ] or ranked_revisits[:1]
        return {
            "probe_index": probe_index,
            "measurements_in_probe": measurements_in_probe,
            "promising_incremental_gain_percent": floor,
            "observations": observations,
            "ranked_cross_layer_revisits": promising_revisits,
            "cross_layer_rounds": cross_layer_rounds,
        }

    def effective_change_policy(
        self,
        history: list[dict[str, Any]] | None = None,
        attempted_history: list[dict[str, Any]] | None = None,
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
                else:
                    assessment = self.assess_measurement(
                        prefix,
                        self.pairwise_metric_comparison(prefix[0], prefix[-1]),
                    )
                if assessment.get("eligible_as_improvement"):
                    phase = "refinement"
                    break

        hierarchy = self.strategy_profile.get("hierarchy", {})
        ordered_probes = (
            hierarchy.get("ordered_probes", [])
            if isinstance(hierarchy, dict)
            else []
        )
        hierarchy_state = self.hierarchical_search_state(
            history or [], ordered_probes, hierarchy
        ) if ordered_probes else {}
        probe_index = int(hierarchy_state.get("probe_index", 0))
        # A successful early feature probe is an anchor, not a reason to skip
        # the remaining high-impact families. Refinement starts after the
        # ordered probe curriculum has terminal evidence for every stage.
        if ordered_probes and probe_index < len(ordered_probes):
            phase = "exploration"
        elif ordered_probes:
            phase = "refinement"

        phase_config = adaptive.get(phase, {}) if adaptive.get("enabled") else {}
        effective = {
            "strategy_version": self.change_policy.get("strategy_version", "legacy"),
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
                phase_config.get("max_total_grid_steps", self.max_total_grid_steps)
            ),
            "derived_parameters": self.derived_parameter_rules,
            "parameter_groups": self.change_policy.get("parameter_groups", {}),
            "high_risk_parameters": self.change_policy.get("high_risk_parameters", []),
            "approved_high_risk_interaction_groups": copy.deepcopy(
                self.change_policy.get("approved_high_risk_interaction_groups", {})
            ),
            "conditional_failure_exclusions": copy.deepcopy(
                self.conditional_search_exclusions
            ),
            "outer_topology_stage": (
                copy.deepcopy(self.strategy_profile.get("topology_stage", {}))
                if not self.topology_fixed_mode
                else {
                    "order": 0,
                    "mode": "fixed_session",
                    "selected_profile": self.config.get("topology", {}).get("profile"),
                    "decision_owner": "operator_policy",
                    "controller_role": "identity_freeze_and_metrics_gate",
                    "requires_new_session_per_topology": True,
                    "completion_rule": (
                        "Do not propose or compare topology changes in this Session; "
                        "optimize only the frozen serving-parameter Search Limits."
                    ),
                }
            ),
        }
        autonomous = self.strategy_profile.get("autonomous_cross_layer", {})
        budget = (
            autonomous.get("exploration_budget", {})
            if isinstance(autonomous, dict)
            else {}
        )
        if isinstance(budget, dict) and budget.get("programmatic_tracking"):
            effective["measured_exploration_budget_state"] = (
                self.measured_exploration_budget_state(
                    history or [], attempted_history or [], budget
                )
            )
        if phase == "exploration" and ordered_probes:
            if probe_index < len(ordered_probes):
                probe = ordered_probes[probe_index]
                if isinstance(probe, dict):
                    effective.update(
                        hierarchical_stage="ordered_probe",
                        hierarchical_probe_index=probe_index,
                        hierarchical_probe=copy.deepcopy(probe),
                    )
                    default_probe_budgets = {
                        "mtp_enablement": [1, 1],
                        "moe_communication": [1, 2],
                        "scheduler_capacity": [2, 3],
                        "compilation_graph": [1, 2],
                        "ascend_communication_refinement": [1, 2],
                    }
                    preferred = probe.get(
                        "independent_parameters_per_round",
                        default_probe_budgets.get(str(probe.get("name"))),
                    )
                    if (
                        isinstance(preferred, list)
                        and len(preferred) == 2
                        and all(isinstance(value, int) for value in preferred)
                    ):
                        preferred_minimum, maximum = preferred
                        minimum = int(
                            probe.get(
                                "minimum_independent_parameters",
                                preferred_minimum,
                            )
                        )
                        if not 1 <= minimum <= maximum:
                            raise ValueError(
                                "hierarchical probe independent parameter range is invalid"
                            )
                        effective.update(
                            preferred_parameters_per_round=preferred,
                            minimum_parameters_per_round=minimum,
                            max_parameters_per_round=maximum,
                        )
                    minimum_measurements, maximum_measurements = (
                        self.hierarchical_probe_measurement_budget(probe)
                    )
                    effective.update(
                        successful_measurements_in_probe=hierarchy_state.get(
                            "measurements_in_probe", 0
                        ),
                        minimum_successful_measurements=minimum_measurements,
                        maximum_successful_measurements=maximum_measurements,
                        probe_exit_policy={
                            "exit_at_maximum": True,
                            "early_exit_after_minimum_when_incremental_gain_below_percent":
                                hierarchy_state.get(
                                    "promising_incremental_gain_percent", -3.0
                                ),
                            "failed_or_incomplete_rounds_do_not_consume_budget": True,
                        },
                        hierarchical_observations=hierarchy_state.get(
                            "observations", []
                        ),
                    )
        elif ordered_probes:
            revisits = hierarchy_state.get("ranked_cross_layer_revisits", [])
            layer_evidence = []
            for revisit in revisits:
                probe_index_for_evidence = int(revisit["probe_index"])
                layer_evidence.append(
                    {
                        **copy.deepcopy(revisit),
                        "probe": copy.deepcopy(
                            ordered_probes[probe_index_for_evidence]
                        ),
                    }
                )
            effective.update(
                hierarchical_stage="cross_layer_refinement",
                cross_layer_selection_owner="agent",
                controller_preselected_layer=None,
                autonomous_cross_layer=copy.deepcopy(
                    self.strategy_profile.get("autonomous_cross_layer", {})
                ),
                cross_layer_evidence=layer_evidence,
                ranked_cross_layer_revisits=revisits,
                hierarchical_observations=hierarchy_state.get("observations", []),
            )
        return effective

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
        for exclusion in self.conditional_search_exclusions:
            if not isinstance(exclusion, dict):
                continue
            conditions = exclusion.get("conditions", {})
            if not isinstance(conditions, dict):
                continue
            # Reject only when the historical record covers the complete
            # current schema and every value matches. Partial overlap remains
            # advisory evidence; it never globally bans a parameter value.
            if self.candidate_schema.issubset(conditions) and all(
                candidate[name] == conditions[name]
                for name in self.candidate_schema
            ):
                raise ValueError(
                    "Candidate exactly matches a conditionally excluded "
                    f"{exclusion.get('failure_classification', 'failed')} combination "
                    f"from {exclusion.get('trial_id', 'history')}"
                )
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
            max_parameter_steps = int(effective_policy["max_grid_steps_per_parameter"])
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
            if (
                item.get("before") != previous[key]
                or item.get("after") != candidate[key]
            ):
                raise ValueError(
                    f"Declared before/after for {key} does not match candidate values"
                )
            if len(str(item.get("rationale", "")).strip()) < 12:
                raise ValueError(f"Rationale for {key} is too short to audit")
        if decision is not None:
            self.validate_change_evidence(decision, len(actual_changes))
            if (
                self.strategy_profile_name
                in {
                    "hierarchical_agentic_frontier_v3",
                    "hierarchical_agentic_guided_v4",
                }
                and previous.get("num_speculative_tokens", 0) > 0
                and candidate.get("num_speculative_tokens") == 0
            ):
                if decision.get("exploration_intent") != "diagnostic_ablation":
                    raise ValueError(
                        "Disabling MTP is allowed only as an explicit diagnostic_ablation"
                    )
                state = effective_policy.get("measured_exploration_budget_state", {})
                if int(state.get("diagnostic_ablation_count", 0)) >= int(
                    state.get("diagnostic_ablation_maximum_per_session", 1)
                ):
                    raise ValueError(
                        "The per-Session MTP diagnostic-ablation budget is exhausted"
                    )

    def submit(
        self,
        round_dir: Path,
        label: str,
        candidate: dict[str, Any],
        *,
        dry_run: bool = False,
        launch_profile: str | None = None,
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
        if self.execution_mode == "executor_adapter":
            return self.submit_executor_adapter(
                round_dir,
                label,
                candidate,
                dry_run=dry_run,
                launch_profile=launch_profile,
            )
        if self.execution_mode == "ktp_lab":
            return self.submit_lab(
                round_dir,
                label,
                candidate,
                dry_run=dry_run,
                launch_profile=launch_profile,
            )
        env_path = round_dir / "02_parameters" / "candidate.env"
        env_path.write_text(
            self.candidate_env(
                label, candidate, launch_profile=launch_profile
            ),
            encoding="utf-8",
            newline="\n",
        )
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
        self.scp_from(
            f"{remote_run}/task.yaml", round_dir / "03_submission" / "task.yaml"
        )
        save_json(
            round_dir / "03_submission" / "submission.json",
            {
                "task_id": task_id,
                "run_id": run_id,
                "submitted_at": now(),
                "dry_run": dry_run,
            },
        )
        return task_id, run_id

    def submit_executor_adapter(
        self,
        round_dir: Path,
        label: str,
        candidate: dict[str, Any],
        *,
        dry_run: bool,
        launch_profile: str | None = None,
    ) -> tuple[str | None, str]:
        """Submit through the frozen v1 bridge without exposing Controller authority."""

        assert self.executor_adapter is not None
        env_path = round_dir / "02_parameters" / "candidate.env"
        env_path.write_text(
            self.candidate_env(label, candidate, launch_profile=launch_profile),
            encoding="utf-8",
            newline="\n",
        )
        result = self.executor_adapter.invoke(
            "submit",
            context=self.executor_context(),
            payload={
                "label": label,
                "dry_run": dry_run,
                "launch_profile": launch_profile,
                "candidate_env_path": str(env_path.resolve()),
                "round_dir": str(round_dir.resolve()),
            },
        )
        run_id = str(result.get("run_id", "")).strip()
        if not run_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
            raise RuntimeError(
                "Executor submit response must contain a filesystem-safe run_id"
            )
        raw_task_id = result.get("task_id")
        task_id = None if raw_task_id is None else str(raw_task_id).strip()
        if not dry_run and not task_id:
            raise RuntimeError(
                "Executor submit response must contain task_id for a real submission"
            )
        output = str(result.get("message") or result.get("submit_output") or "")
        (round_dir / "03_submission" / "submit_output.txt").write_text(
            output + ("\n" if output else ""), encoding="utf-8"
        )
        task_document = result.get("task")
        if task_document is not None:
            if not isinstance(task_document, dict):
                raise RuntimeError("Executor submit task must be a mapping")
            save_yaml(round_dir / "03_submission" / "task.yaml", task_document)
        save_json(
            round_dir / "03_submission" / "submission.json",
            {
                "execution_mode": "executor_adapter",
                "executor_identity": self.executor_identity,
                "task_id": task_id,
                "run_id": run_id,
                "submitted_at": now(),
                "dry_run": dry_run,
            },
        )
        return task_id, run_id

    def collect(self, run_id: str, round_dir: Path) -> dict[str, bool]:
        remote_run = f"{self.remote_auto}/runs/{run_id}"
        found: dict[str, bool] = {}
        for name in REMOTE_ARTIFACTS:
            if name in {"metrics.json"}:
                target = round_dir / "05_results" / name
            elif name in {
                "candidate.env",
                "task.yaml",
                "effective_config.yaml",
                "vllm_common_command.txt",
            }:
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

    def collect_with_retry(self, run_id: str, round_dir: Path) -> dict[str, bool]:
        """Absorb short control-plane outages without changing experiment state."""
        failures: list[str] = []
        for attempt in range(1, 4):
            try:
                return self.collect(run_id, round_dir)
            except Exception as exc:
                failures.append(f"attempt {attempt}/3: {type(exc).__name__}: {exc}")
                if attempt < 3:
                    log(f"Artifact collection failed ({attempt}/3); retrying.")
                    time.sleep(min(30, 5 * attempt))
        raise RecoverableControllerIOError(
            "Artifact collection failed after 3 attempts: " + failures[-1][-2000:]
        )

    def start_aligned_benchmark(self, run_id: str, task_id: str | None) -> None:
        if self.benchmark_mode != "aligned_l1":
            return
        if self.execution_mode == "executor_adapter":
            assert self.executor_adapter is not None
            if not self.executor_adapter.supports("start_benchmark"):
                raise RuntimeError(
                    "aligned_l1 requires executor_adapter capability start_benchmark"
                )
            self.executor_adapter.invoke(
                "start_benchmark",
                context=self.executor_context(),
                payload={"run_id": run_id, "task_id": task_id},
            )
            log(f"Started aligned L1 benchmark through executor adapter for run={run_id}")
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
        if self.execution_mode == "executor_adapter":
            assert self.executor_adapter is not None
            result = self.executor_adapter.invoke(
                "snapshot",
                context=self.executor_context(),
                payload={"task_id": str(task_id)},
            )
            return validate_executor_snapshot(result)
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
            processes_terminal = not processes_running and bool(
                re.search(
                    r"\b(?:idle|failed|error|exited|stopped|completed|succeeded)"
                    r"(?:=\d+)?\b",
                    status_summary,
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
        if self.execution_mode == "executor_adapter":
            if not task_id:
                return
            assert self.executor_adapter is not None
            if self.executor_adapter.supports("stop_partial"):
                self.executor_adapter.invoke(
                    "stop_partial",
                    context=self.executor_context(),
                    payload={"task_id": str(task_id)},
                )
            else:
                self.executor_adapter.invoke(
                    "stop",
                    context=self.executor_context(),
                    payload={"task_id": str(task_id), "reason": "partial_failure"},
                )
            return
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

    def stop_active_task(self, state: dict[str, Any]) -> str:
        """Stop only the task recorded by the frozen Session state."""
        task_id = state.get("active_task_id")
        if not task_id:
            return "No active task is recorded; only the local stop request was saved."
        execution_mode = str(state.get("execution_mode", self.execution_mode))
        if execution_mode == "executor_adapter":
            if self.execution_mode != "executor_adapter" or self.executor_adapter is None:
                raise RuntimeError(
                    "Frozen Session requires its executor adapter configuration"
                )
            result = self.executor_adapter.invoke(
                "stop",
                context=self.executor_context(),
                payload={"task_id": str(task_id), "reason": "operator_stop"},
            )
            return str(result.get("message", "External executor task stopped."))
        if execution_mode == "ktp_lab":
            lease_name = str(state.get("lease_name") or self.lab.get("lease_name"))
            if not lease_name:
                raise RuntimeError("Frozen Session has no Lease identity to stop")
            return self.ssh(
                f"cd {shlex.quote(self.remote_project)} && "
                f"ktp-lab stop --lease {shlex.quote(lease_name)}",
                timeout=180,
            )
        if not str(task_id).isdigit():
            raise RuntimeError(f"Invalid legacy ktp task id: {task_id!r}")
        return self.ssh(f"ktp stop {int(task_id)}", timeout=180)

    @staticmethod
    def benchmark_regime(
        metrics: dict[str, Any],
        round_dir: Path | None = None,
    ) -> str:
        """Identify stable benchmark inputs without per-run target details."""
        mode = str(metrics.get("benchmark_mode", "legacy_or_unknown"))
        identity = metrics.get("benchmark_identity", {})
        if isinstance(identity, dict) and identity.get("sha256"):
            return f"identity:{identity['sha256']}"
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
                params_path = (
                    metrics_path.parents[1] / "02_parameters" / "candidate_params.yaml"
                )
                params = load_yaml(params_path) if params_path.exists() else {}
                item = {
                        "round": metrics_path.parents[1].name,
                        "params": params,
                        "metrics": metrics,
                        "benchmark_regime": self.benchmark_regime(
                            metrics,
                            metrics_path.parents[1],
                        ),
                    }
                decision_path = (
                    metrics_path.parents[1] / "06_agent_analysis" / "decision.json"
                )
                if decision_path.exists():
                    decision = json.loads(decision_path.read_text(encoding="utf-8"))
                    if isinstance(decision, dict):
                        item["decision"] = decision
                history.append(item)
            except (OSError, ValueError, yaml.YAMLError):
                continue
        if not history:
            return history
        latest_regime = history[-1]["benchmark_regime"]
        return [item for item in history if item["benchmark_regime"] == latest_regime]

    def attempted_history_summary(self, session_dir: Path) -> list[dict[str, Any]]:
        """Return terminal candidates without conflating infra and experiments.

        A candidate is experimentally covered only after it produced metrics or a
        structured failure decision attributed startup failure to that candidate.
        Lease/address/network/process failures remain visible for diagnosis, but
        must not consume search coverage or allow the Agent to skip the value.
        """
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
                    "counts_as_parameter_experiment": metrics_path.exists(),
                    "experiment_evidence_status": (
                        "benchmarked" if metrics_path.exists() else "unattributed_failure"
                    ),
                }
                decision_path = round_dir / "06_agent_analysis" / "decision.json"
                if decision_path.exists():
                    decision = json.loads(decision_path.read_text(encoding="utf-8"))
                    if isinstance(decision, dict):
                        item["decision"] = decision
                if metrics_path.exists():
                    item["metrics"] = json.loads(
                        metrics_path.read_text(encoding="utf-8")
                    )
                if failure_path.exists():
                    item["failure"] = load_yaml(failure_path)
                    failure_decision_path = (
                        round_dir / "06_agent_analysis" / "failure_decision.json"
                    )
                    reclassification_path = (
                        round_dir
                        / "06_agent_analysis"
                        / "failure_reclassification.json"
                    )
                    if failure_decision_path.exists():
                        decision = json.loads(
                            failure_decision_path.read_text(encoding="utf-8")
                        )
                        item["failure_decision"] = decision
                        classification = str(decision.get("classification", ""))
                        action = str(decision.get("action", ""))
                        parameter_attributed = (
                            action == "adjust_parameters"
                            and classification in FAILURE_ADJUSTABLE_CLASSIFICATIONS
                        )
                        if parameter_attributed:
                            item["counts_as_parameter_experiment"] = True
                            item["experiment_evidence_status"] = (
                                "parameter_attributed_startup_failure"
                            )
                        else:
                            item["experiment_evidence_status"] = (
                                "infrastructure_or_unattributed_failure"
                            )
                    if reclassification_path.exists():
                        reclassification = json.loads(
                            reclassification_path.read_text(encoding="utf-8")
                        )
                        item["failure_reclassification"] = reclassification
                        if reclassification.get("counts_as_parameter_experiment") is False:
                            item["counts_as_parameter_experiment"] = False
                            item["experiment_evidence_status"] = (
                                "superseded_as_infrastructure_or_unattributed_failure"
                            )
                attempts.append(item)
            except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
                continue
        return attempts

    def measured_exploration_budget_state(
        self,
        history: list[dict[str, Any]],
        attempted_history: list[dict[str, Any]],
        budget: dict[str, Any],
    ) -> dict[str, Any]:
        """Measure strategy mix without choosing the Agent's next intent."""
        targets = {
            "exploitation": float(budget.get("exploitation_fraction", 0.65)),
            "cross_layer_interaction": float(
                budget.get("cross_layer_interaction_fraction", 0.25)
            ),
            "frontier_novelty": float(
                budget.get("frontier_novelty_fraction", 0.10)
            ),
        }
        if any(value < 0 for value in targets.values()) or not math.isclose(
            sum(targets.values()), 1.0, abs_tol=1e-6
        ):
            raise ValueError("Cross-layer exploration budget fractions must sum to 1")

        records: dict[str, dict[str, Any]] = {}
        for item in [*history, *attempted_history]:
            decision = item.get("decision")
            if isinstance(decision, dict):
                records[str(item.get("round", len(records)))] = decision
        counts = {intent: 0 for intent in targets}
        diagnostic_count = 0
        for decision in records.values():
            intent = str(decision.get("exploration_intent", ""))
            if intent in counts:
                counts[intent] += 1
            elif intent == "diagnostic_ablation":
                diagnostic_count += 1
        measured_rounds = sum(counts.values())
        actual = {
            intent: (
                round(count / measured_rounds, 4) if measured_rounds else 0.0
            )
            for intent, count in counts.items()
        }
        deficits = {
            intent: round(targets[intent] - actual[intent], 4)
            for intent in targets
        }
        underrepresented = [
            intent
            for intent, deficit in sorted(
                deficits.items(), key=lambda pair: pair[1], reverse=True
            )
            if deficit > 0
        ]
        diagnostic_maximum = 1
        hierarchy = self.strategy_profile.get("hierarchy", {})
        probes = hierarchy.get("ordered_probes", []) if isinstance(hierarchy, dict) else []
        for probe in probes:
            diagnostic = probe.get("diagnostic_ablation", {}) if isinstance(probe, dict) else {}
            if isinstance(diagnostic, dict) and "maximum_per_session" in diagnostic:
                diagnostic_maximum = int(diagnostic["maximum_per_session"])
                break
        return {
            "programmatic_tracking": True,
            "agent_final_choice": bool(budget.get("agent_final_choice", True)),
            "target_fractions": targets,
            "measured_round_count": measured_rounds,
            "counts": counts,
            "actual_fractions": actual,
            "fraction_deficits": deficits,
            "underrepresented_intents": underrepresented,
            "controller_preselected_intent": None,
            "diagnostic_ablation_count": diagnostic_count,
            "diagnostic_ablation_maximum_per_session": diagnostic_maximum,
            "diagnostic_ablation_remaining": max(
                0, diagnostic_maximum - diagnostic_count
            ),
        }

    @staticmethod
    def primary_performance_score(item: dict[str, Any]) -> float | None:
        """Return the comparable primary score for a successful history item."""
        metrics = item.get("metrics", {})
        if metrics.get("benchmark_mode") == "aligned_l1":
            value = metrics.get("l1", {}).get("primary_aggregate_output_tps_geomean")
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
            if item.get("metrics", {}).get("benchmark_mode") == "aligned_l1":
                assessment = self.assess_aligned_l1([baseline, item])
            else:
                assessment = self.assess_measurement(
                    [baseline, item], self.pairwise_metric_comparison(baseline, item)
                )
            if assessment.get("eligible_as_improvement"):
                accepted.append(item)
        scored = [(self.primary_performance_score(item), item) for item in accepted]
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

    @staticmethod
    def pairwise_metric_comparison(
        baseline: dict[str, Any], current: dict[str, Any]
    ) -> dict[str, Any]:
        """Build the baseline deltas used by non-L1 deterministic gates."""
        baseline_metrics = baseline.get("metrics", {}).get("metrics", {})
        current_metrics = current.get("metrics", {}).get("metrics", {})
        deltas: dict[str, Any] = {}
        for key, current_value in current_metrics.items():
            baseline_value = baseline_metrics.get(key)
            if (
                isinstance(current_value, (int, float))
                and not isinstance(current_value, bool)
                and isinstance(baseline_value, (int, float))
                and not isinstance(baseline_value, bool)
            ):
                deltas[key] = {
                    "current": current_value,
                    "reference": baseline_value,
                    "absolute": current_value - baseline_value,
                    "percent": (
                        (current_value - baseline_value) / baseline_value * 100
                        if baseline_value != 0
                        else None
                    ),
                }
        return {"numeric_metric_deltas_vs_baseline": deltas}

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
                if not item.get("counts_as_parameter_experiment", False):
                    continue
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
                    "gain_vs_baseline_percent": (score / baseline_score - 1.0) * 100.0,
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
            and item.get("counts_as_parameter_experiment", False)
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
        previous_metrics = (
            history[-2]["metrics"].get("metrics", {})
            if len(history) > 1
            else baseline_metrics
        )
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
        """Replace source-default baseline estimates with values resolved by vLLM.

        B0 adds only the audited ``max_model_len=64000`` compatibility override.
        The Agent must not receive other typed pre-launch estimates as if they
        were observed values, so the first successful round is reconciled from
        the engine log before analysis or candidate generation can begin.
        """
        initial = self.config.get("initial_baseline", {})
        round_launch_profile = self.round_launch_profile(round_dir)
        if round_launch_profile is None:
            # Compatibility with archived/tests created before candidate.env
            # recorded the launch identity next to every round.
            round_launch_profile = (
                str(initial.get("launch_profile", "explicit_candidate"))
                if state.get("round_label") == str(initial.get("label", "a0"))
                else "explicit_candidate"
            )
        if (
            round_launch_profile != B0_LAUNCH_PROFILE
            or initial.get("launch_profile") != B0_LAUNCH_PROFILE
            or state.get("official_source_defaults_reconciled")
        ):
            return

        log_path = round_dir / "04_runtime" / "master.log"
        if not log_path.is_file():
            raise RuntimeError(
                "B0 completed without master.log; defaults cannot be audited"
            )
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        resolved = dict(state["current_candidate"])
        launch_profile = str(initial["launch_profile"])
        evidence: dict[str, Any] = {
            "launch_profile": launch_profile,
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
                "enable_balance_scheduling": False,
                "enable_reduce_sample": False,
                "speculative_config__enforce_eager": None,
            },
            "log_resolved": {},
        }
        evidence["explicit_deployment_overrides"] = {
            "max_model_len": int(state["current_candidate"]["max_model_len"]),
            "rationale": (
                "The pinned model resolves max_seq_len=1048576, which requires "
                "107.25 GiB KV cache while the 2x16-NPU deployment exposes "
                "28.82 GiB. Fixing 64000 is the sole deployability override."
            ),
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
        if "enable_balance_scheduling" in resolved:
            resolved["enable_balance_scheduling"] = False
        if "enable_reduce_sample" in resolved:
            resolved["enable_reduce_sample"] = False
        if "speculative_config__enforce_eager" in resolved:
            resolved["speculative_config__enforce_eager"] = None
        if "speculative_config__method" in resolved:
            resolved["speculative_config__method"] = None

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
            # Once B0 has resolved a source default, a second ``None`` choice
            # would merely re-run that same effective value while weakening
            # numeric and combination validation. Preserve None only when the
            # authoritative effective value itself is None.
            if value is not None and None in limits:
                limits = [item for item in limits if item is not None]
                self.config["search_limits"][name] = limits
            if value not in limits:
                self.config["search_limits"][name] = [value, *limits]

        # Automatic Search Limits are initially compiled before B0 has run.
        # Re-anchor numeric axes using the authoritative effective B0 values,
        # rather than the older portrait/scenario snapshot.
        if self.automatic_compatibility:
            for name, value in resolved.items():
                rebuilt = self.automatic_compatibility.numeric_domain(
                    name,
                    value,
                    self.config["search_limits"][name],
                    include_source_values=False,
                )
                if rebuilt is not None:
                    self.config["search_limits"][name] = rebuilt

        # The B0 log is authoritative for source-default anchors. Keep the
        # frozen automatic validator on exactly the same post-reconciliation
        # domains as the Session whitelist, otherwise a newly observed source
        # default could pass the outer whitelist but fail the inner compiler
        # snapshot on the first Agent handoff.
        if self.automatic_registry_validation:
            active_parameters = self.automatic_registry_validation["compiled"][
                "active_parameters"
            ]
            for parameter in active_parameters:
                name = str(parameter["canonical_name"])
                parameter["values"] = copy.deepcopy(self.config["search_limits"][name])
            self.config["automatic_registry_validation"] = copy.deepcopy(
                self.automatic_registry_validation
            )

        save_yaml(round_dir / "02_parameters" / "candidate_params.yaml", resolved)
        save_yaml(
            round_dir / "02_parameters" / "b0_effective_resolution.yaml", evidence
        )
        effective_path = round_dir / "02_parameters" / "effective_config.yaml"
        effective = load_yaml(effective_path) if effective_path.is_file() else {}
        effective["launch_profile"] = launch_profile
        effective["authoritative_effective_service"] = resolved
        effective["authoritative_source"] = (
            "master.log plus pinned-source defaults and max_model_len deployability override"
        )
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
        if history and history[-1]["metrics"].get("benchmark_mode") == "aligned_l1":
            return self.assess_aligned_l1(history)
        policy = self.measurement_policy
        if not history:
            return {"classification": "missing_metrics", "passes_guardrails": False}
        current = history[-1]["metrics"].get("metrics", {})
        successful = int(current.get("successful_requests", 0) or 0)
        failed = int(current.get("failed_requests", 0) or 0)
        selected_definition = self.benchmark.get(self.benchmark_mode, {})
        required_successes = int(
            selected_definition.get(
                "minimum_successful_requests",
                policy.get(
                    "minimum_successful_requests",
                    selected_definition.get(
                        "num_prompts", self.config.get("fixed_scenario", {}).get("num_prompts", 1)
                    ),
                ),
            )
        )
        passes_guardrails = successful >= required_successes
        require_zero_failed = bool(
            selected_definition.get(
                "require_zero_failed_requests",
                policy.get("require_zero_failed_requests", True),
            )
        )
        if require_zero_failed:
            passes_guardrails = passes_guardrails and failed == 0
        assessment: dict[str, Any] = {
            "classification": "baseline_only" if len(history) == 1 else "candidate",
            "passes_guardrails": passes_guardrails,
            "successful_requests": successful,
            "failed_requests": failed,
            "policy": policy,
            "eligible_as_improvement": False,
            "advisories": [],
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
        latency_guardrail_mode = str(
            selected_definition.get(
                "latency_guardrail_mode",
                policy.get("latency_guardrail_mode", "hard"),
            )
        )
        if latency_guardrail_mode not in {"hard", "advisory"}:
            raise ValueError(
                "measurement latency_guardrail_mode must be hard or advisory"
            )
        assessment["latency_guardrail_mode"] = latency_guardrail_mode
        if throughput_gain is None or (
            latency_guardrail_mode == "hard"
            and (ttft_change is None or tpot_change is None)
        ):
            assessment["classification"] = "insufficient_comparison"
            return assessment
        minimum_gain = float(
            selected_definition.get(
                "minimum_throughput_gain_percent",
                policy.get("minimum_throughput_gain_percent", 3.0),
            )
        )
        maximum_ttft = float(
            selected_definition.get(
                "maximum_ttft_regression_percent",
                policy.get("maximum_ttft_regression_percent", 10.0),
            )
        )
        maximum_tpot = float(
            selected_definition.get(
                "maximum_tpot_regression_percent",
                policy.get("maximum_tpot_regression_percent", 10.0),
            )
        )
        latency_passes = bool(
            ttft_change is not None
            and tpot_change is not None
            and ttft_change <= maximum_ttft
            and tpot_change <= maximum_tpot
        )
        if latency_guardrail_mode == "advisory" and not latency_passes:
            assessment["advisories"].append(
                "latency regression exceeded the configured reference threshold"
            )
        assessment["eligible_as_improvement"] = bool(
            passes_guardrails
            and throughput_gain >= minimum_gain
            and (latency_guardrail_mode == "advisory" or latency_passes)
        )
        return assessment

    def assess_aligned_l1(
        self,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        policy = self.measurement_policy.get("aligned_l1", {})
        latency_guardrail_mode = str(policy.get("latency_guardrail_mode", "hard"))
        if latency_guardrail_mode not in {"hard", "advisory"}:
            raise ValueError(
                "aligned_l1 latency_guardrail_mode must be hard or advisory"
            )
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
            "advisories": [],
            "latency_guardrail_mode": latency_guardrail_mode,
        }
        if not absolute_gate:
            assessment["classification"] = "absolute_gate_failed"
            assessment["violations"].append(
                "not all required L1 repetitions passed the frozen case evidence gate"
            )
            return assessment
        if len(history) == 1:
            return assessment

        baseline_payload = history[0]["metrics"]
        baseline_l1 = baseline_payload.get("l1", {})
        if not baseline_l1.get("all_repetitions_gate_passed"):
            assessment["classification"] = "invalid_baseline"
            assessment["violations"].append(
                "baseline did not pass the L1 evidence gate"
            )
            return assessment

        baseline_score = baseline_l1.get("primary_aggregate_output_tps_geomean")
        current_score = current_l1.get("primary_aggregate_output_tps_geomean")
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
        throughput_floor = (
            1.0
            - float(
                policy.get("maximum_single_workload_throughput_regression_percent", 5.0)
            )
            / 100.0
        )
        workload_throughput_ratios: dict[str, float] = {}
        for key, current_case in current_cases.items():
            baseline_case = baseline_cases[key]
            case_name = f"{key[0]}@c{key[1]}"
            latency_ratios[case_name] = {}
            for metric, limit in latency_limits.items():
                current_value = float(current_case[metric])
                baseline_value = float(baseline_case[metric])
                ratio = (
                    current_value / baseline_value if baseline_value > 0 else math.inf
                )
                latency_ratios[case_name][metric] = ratio
                if ratio > limit:
                    destination = (
                        assessment["violations"]
                        if latency_guardrail_mode == "hard"
                        else assessment["advisories"]
                    )
                    destination.append(
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
        if self.execution_mode == "executor_adapter":
            assert self.executor_adapter is not None
            result = self.executor_adapter.invoke(
                "wait_for_release",
                context=self.executor_context(),
                payload={
                    "task_id": str(task_id),
                    "timeout_seconds": min(self.round_timeout_minutes * 60, 3600),
                },
            )
            if result.get("released") is not True:
                raise RuntimeError(
                    f"Executor task {task_id!r} did not release; refusing overlap"
                )
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

    @staticmethod
    def failure_signature_evidence(round_dir: Path, *, max_lines: int = 120) -> list[str]:
        """Extract decisive error lines from complete logs before prompt truncation."""
        patterns = re.compile(
            r"Address already in use|EADDRINUSE|ZMQError|Traceback|"
            r"out of memory|OutOfMemory|HCCL.*(?:error|failed)|"
            r"ERR\d+|EI\d+|Vector core execution timed out|aclnn\w+|"
            r"fused_mlapo|Connection closed by peer|"
            r"Process ApiServer_\d+.*died|benchmark.*(?:error|failed)",
            re.IGNORECASE,
        )
        evidence: list[str] = []
        runtime_dir = round_dir / "04_runtime"
        for name in (
            "master.log",
            "worker.log",
            "benchmark_runner.log",
            "warmup.log",
            "formal.log",
        ):
            path = runtime_dir / name
            if not path.is_file():
                continue
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(),
                start=1,
            ):
                if patterns.search(line):
                    evidence.append(f"{name}:{line_number}: {line[-4000:]}")
                    if len(evidence) >= max_lines:
                        return evidence
        return evidence

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
            "topology_first_layer": copy.deepcopy(self.topology_plan),
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
                round_dir.parent / "00_search_space" / "parameter_portraits.agent.yaml",
                max_chars=120000,
            ),
            "runtime_rules": (
                {
                    "rules": self.runtime_rule_store.data.get("rules", []),
                    "quarantines": self.runtime_rule_store.data.get("quarantines", []),
                    "proposals": self.runtime_rule_store.data.get("proposals", []),
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
            protocol_failures: list[str] = []
            result = None
            protocol_attempts = self.max_agent_protocol_retries + 1
            for protocol_attempt in range(1, protocol_attempts + 1):
                protocol_prompt = prompt
                if protocol_failures:
                    protocol_prompt += f"""

The previous provider call failed its transport or structured-output contract.
Return a fresh, complete JSON decision conforming exactly to the supplied schema.
Failure detail: {protocol_failures[-1][-2000:]}
"""
                try:
                    result = run_structured_agent(
                        self.agent_config,
                        prompt=protocol_prompt,
                        schema_path=self.agent_schema_path(session_dir),
                        output_path=decision_path,
                        cwd=KB_ROOT,
                        allowed_dir=round_dir,
                    )
                except Exception as exc:
                    protocol_failures.append(
                        f"attempt {protocol_attempt}/{protocol_attempts}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    if protocol_attempt < protocol_attempts:
                        log(
                            "Agent provider raised a transport exception "
                            f"({protocol_attempt}/{protocol_attempts}); retrying."
                        )
                        continue
                    raise AgentProtocolError(
                        "Agent provider raised exceptions after all protocol "
                        f"attempts: {protocol_failures[-1][-2000:]}"
                    ) from exc
                suffix = f"attempt_{attempt:02d}.protocol_{protocol_attempt:02d}"
                (analysis_dir / f"agent_events.{suffix}.jsonl").write_text(
                    result.stdout, encoding="utf-8"
                )
                (analysis_dir / f"agent_stderr.{suffix}.log").write_text(
                    result.stderr, encoding="utf-8"
                )
                if result.returncode == 0 and decision_path.exists():
                    break
                protocol_failures.append(
                    f"attempt {protocol_attempt}/{protocol_attempts}: "
                    f"{result.provider}: {result.stderr[-2000:]}"
                )
                log(
                    "Agent protocol attempt failed "
                    f"({protocol_attempt}/{protocol_attempts}); retrying within "
                    "the same experiment round."
                )
            assert result is not None
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
                save_json(
                    analysis_dir / "agent_protocol_recovery_audit.json",
                    {
                        "failed_at": now(),
                        "max_protocol_retries": self.max_agent_protocol_retries,
                        "failures": protocol_failures,
                        "final_status": "controller_restart_required",
                    },
                )
                raise AgentProtocolError(
                    f"{result.provider} Agent analysis failed after "
                    f"{protocol_attempts} protocol attempts: {result.stderr[-2000:]}"
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
        selection_policy = self.effective_change_policy(history, attempted_history)
        evidence["selection_policy"] = selection_policy
        save_json(
            round_dir / "06_agent_analysis" / "evidence_bundle.json",
            evidence,
        )
        if self.benchmark_mode == "aligned_l1":
            repetition_count = int(self.benchmark["aligned_l1"]["repetitions"])
            repetition_aggregation = (
                "the single complete repetition"
                if repetition_count == 1
                else f"the median across {repetition_count} complete repetitions"
            )
            latency_is_advisory = (
                str(
                    self.measurement_policy.get("aligned_l1", {}).get(
                        "latency_guardrail_mode", "hard"
                    )
                )
                == "advisory"
            )
            latency_role = (
                "TTFT/TPOT P50/P90 are advisory diagnostics: report and reason "
                "about them, but they cannot veto a valid output-throughput gain. "
                "Zero errors/incomplete requests, exact token shapes, per-workload "
                "output throughput, and run-to-run CV remain deterministic guardrails."
                if latency_is_advisory
                else "TTFT/TPOT P50/P90, zero errors/incomplete requests, exact "
                "token shapes, per-workload throughput, and run-to-run CV are "
                "deterministic guardrails."
            )
            aligned_definition = self.benchmark["aligned_l1"]
            formal_case_count = int(aligned_definition.get("expected_formal_cases", 12))
            suite_id = str(aligned_definition.get("suite_id", "tuning-fixed"))
            concurrency_description = (
                "fixed C32 concurrency"
                if formal_case_count == 4
                else "fixed C1/C16/C32 concurrency"
            )
            benchmark_goal = f"""Goal: improve the strict aggregate output-token throughput score for the
frozen ServeBench {suite_id} L1 matrix. The matrix has four workloads
(1024/256, 8192/512, 1024/1024, 256/2048), {concurrency_description},
{formal_case_count} formal cases, fixed JSONL prompts, temperature=0, and
{repetition_count} complete repetition(s). The primary score is
{repetition_aggregation} of the C32 workload geometric mean. The final report
must explicitly surface output throughput, TTFT P50/P90 and TPOT P50/P90.
{latency_role}"""
        else:
            definition = self.benchmark[self.benchmark_mode]
            latency_is_advisory = (
                str(
                    definition.get(
                        "latency_guardrail_mode",
                        self.measurement_policy.get(
                            "latency_guardrail_mode", "hard"
                        ),
                    )
                )
                == "advisory"
            )
            latency_role = (
                "Mean TTFT and mean TPOT are advisory diagnostics and cannot "
                "veto a valid output-throughput gain. Request completeness and "
                "error requirements remain deterministic guardrails."
                if latency_is_advisory
                else "Successful/failed request counts, mean TTFT and mean TPOT "
                "are deterministic guardrails."
            )
            benchmark_goal = f"""Goal: improve measured output-token throughput under the frozen
{self.benchmark_profile_name} benchmark profile ({self.benchmark_mode}). Its complete definition
is present in the evidence bundle. {latency_role} Never compare this profile with a different
benchmark identity."""
        hierarchical_probe = selection_policy.get("hierarchical_probe")
        hierarchy_instruction = ""
        if isinstance(hierarchical_probe, dict):
            if hierarchical_probe.get("parameter_selection_owner") == "agent":
                hierarchy_instruction = f"""
This strategy is in an ordered high-impact layer. The Controller selected only
the current semantic layer because its prerequisites precede later layers; it
did not select a parameter, parameter count, value, or coupling for you. You own
those choices. The coupling_hints are non-binding hypotheses, not an experiment
queue. Compare plausible alternatives inside the layer and choose one to four
legal parameters according to evidence and information value. Prefer a coherent
multi-parameter experiment when known coupling makes it faster and more valid
than isolated sweeps; choose a single parameter when isolation is genuinely the
highest-information test. Constraint-required companion parameters may come
from another layer, but cite why they are mechanical or causally necessary:
{yaml.safe_dump(hierarchical_probe, allow_unicode=True, sort_keys=False)}
"""
            else:
                hierarchy_instruction = f"""
This strategy is in an ordered high-impact probe stage. Center the next experiment
on the active probe below; choose a valid untested value from this parameter family
before substituting a lower-impact local tweak. You may skip the probe only when
the frozen constraints or attempted history make it invalid, unsafe, or exhausted,
and then you must cite the exact evidence for skipping it. The probe is a general
strategy prior, not a hidden historical candidate:
{yaml.safe_dump(hierarchical_probe, allow_unicode=True, sort_keys=False)}
"""
        if selection_policy.get("cross_layer_selection_owner") == "agent":
            hierarchy_instruction = f"""
The ordered coverage stage is complete. You—not the Controller—own the next
layer and cross-layer choice. The Controller has supplied measurement summaries
for every covered family, the full active whitelist, attempted combinations and
the best accepted anchor. Select the most informative untried layer, or a
defensible one-to-four parameter cross-layer interaction. Balance exploitation, novel
high-upside interactions, and frontier exploration according to the frozen
autonomous_cross_layer policy. Cite the evidence for your choice; no candidate
family has been preselected for you:
{yaml.safe_dump(selection_policy.get('cross_layer_evidence', []), allow_unicode=True, sort_keys=False)}
"""
        prompt = f"""You are the tuning analyst for a measured vLLM-Ascend experiment.

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
{hierarchy_instruction}
In exploration, use the configured preferred active-parameter range when evidence
supports a coherent faster probe; in refinement, prefer a small local change. A derived
parameter changed together with one of its declared drivers does not consume an
active-parameter slot or grid-step budget, but it must still be declared in changes
and satisfy every hard invariant. Treat listed high-risk parameters conservatively:
use a single high-risk parameter for an isolated probe, but use an approved high-risk
interaction group up to its stated maximum when the hypothesis and constraint checks
require the coupled experiment. Other high-risk combinations require equally explicit
evidence and must remain inside every active-parameter and grid-step budget.
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
empty changes array, change_strategy=none, exploration_intent=none, and empty
interaction_analysis and constraint_checks when the useful whitelist search space is exhausted, no safe
untested change remains, or further testing is not justified by the measurements.
For a continuing guided decision, set exploration_intent=ordered_probe during ordered
coverage, or select exploitation, cross_layer_interaction, or frontier_novelty in
cross-layer refinement using measured_exploration_budget_state as a measured deficit
signal. The Agent makes the final choice; the Controller does not preselect an intent.
Use diagnostic_ablation only for an explicit diagnostic hypothesis. Keep MTP enabled
in normal experiments. Disabling a currently enabled MTP path is allowed only with
exploration_intent=diagnostic_ablation and at most once per Session.
Use only values allowed by this whitelist:
{yaml.safe_dump(self.config['search_limits'], allow_unicode=True, sort_keys=False)}

The frozen runtime contract is: {self.runtime_guardrail}.
Topology was resolved in the topology-first outer stage and frozen into this
Session. Do not change model, DP/TP, Pod/NPU topology, network, ports, image,
benchmark, quantization, or fixed Ascend environment inside this Session.
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
            if self.strategy_profile_name in {
                "hierarchical_agentic_frontier_v3",
                "hierarchical_agentic_guided_v4",
            }:
                intent = decision.get("exploration_intent")
                stage = selection_policy.get("hierarchical_stage")
                allowed_intents = (
                    {"ordered_probe", "diagnostic_ablation"}
                    if stage == "ordered_probe"
                    else {
                        "exploitation",
                        "cross_layer_interaction",
                        "frontier_novelty",
                        "diagnostic_ablation",
                    }
                )
                if intent not in allowed_intents:
                    raise ValueError(
                        f"exploration_intent={intent!r} is invalid for stage={stage!r}"
                    )
            self.validate_candidate(
                previous,
                decision["candidate"],
                decision["changes"],
                decision,
                selection_policy,
            )
            self.validate_selected_portrait_evidence(decision["changes"])
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
        historical_failure_signatures = []
        for prior_round in sorted(session_dir.glob("round_*")):
            if prior_round == round_dir:
                continue
            failure_path = prior_round / "05_results" / "failure.yaml"
            if not failure_path.is_file():
                continue
            historical_failure_signatures.append(
                {
                    "round": prior_round.name,
                    "signatures": self.failure_signature_evidence(
                        prior_round, max_lines=80
                    ),
                    "prior_decision": self.evidence_text(
                        prior_round
                        / "06_agent_analysis"
                        / "failure_decision.json",
                        max_chars=20000,
                    ),
                }
            )
        failure_evidence = {
            "current_candidate": current,
            "current_runtime_recovery_parameters": copy.deepcopy(
                self.runtime_recovery_values
            ),
            "recovery_parameter_registry": copy.deepcopy(
                self.recovery_parameter_registry
            ),
            "history": history,
            "historical_failure_signatures": historical_failure_signatures,
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
            "error_signatures_from_complete_logs": self.failure_signature_evidence(
                round_dir
            ),
            "parameter_portraits": self.evidence_text(
                session_dir / "00_search_space" / "parameter_portraits.agent.yaml",
                max_chars=120000,
            ),
            "runtime_rules": (
                {
                    "rules": self.runtime_rule_store.data.get("rules", []),
                    "quarantines": self.runtime_rule_store.data.get("quarantines", []),
                    "proposals": self.runtime_rule_store.data.get("proposals", []),
                }
                if self.runtime_rule_store
                else None
            ),
        }
        save_json(
            round_dir / "06_agent_analysis" / "failure_evidence_bundle.json",
            failure_evidence,
        )
        prompt = f"""You are the failure analyst for a vLLM-Ascend experiment
that ended without metrics.json.

All required evidence is embedded below. You may execute local read-only commands for
additional inspection. Never write or edit files, run remote commands, access the
network, submit jobs, stop jobs, or change external state.

You own the recovery decision. The Controller does not choose a layer, parameter, or
candidate for you. Correlate the earliest actionable exception in the current round with
historical_failure_signatures; later Gloo/HCCL peer failures may be cascades and must not
erase an unresolved earlier root cause.

Choose the highest-information legal next action:
- adjust_parameters may be used for any adjustable classification below. You may choose
  ordinary Active Search parameters, evidence-gated Recovery Registry controls, or a
  coupled cross-layer combination. A hypothesis-driven high-risk diagnostic is allowed
  when it can bypass the failing path or decisively distinguish competing causes; state
  the hypothesis and constraints explicitly.
- retry_same is appropriate for a genuinely transient infrastructure/network/HCCL or
  Benchmark failure when an unchanged rerun provides value.
- diagnostic_retry_same is appropriate when evidence is incomplete and an unchanged run
  is the best information-gain experiment.
- benchmark_failure: when SERVICE_READY exists, the failure is confined to the benchmark
  harness, the candidate is unchanged, and serving logs contain no dangerous OOM, HCCL,
  EngineCore, API-startup, or identity-drift signature, choose retry_same. Uncertainty about
  the exact benchmark-harness trigger is not by itself a reason to pause. Deterministic
  benchmark recovery is evaluated before this Agent call.
- pause_for_human is a hard-terminal action only. Use it only when evidence explicitly
  proves an unavailable/unsupported image or model capability, immutable identity
  mismatch, missing permission/credential/model artifact, unsatisfied resource/topology
  contract, or corrupt/inconsistent controller state. Repeated failure, uncertainty,
  HCCL/network errors, runtime bugs, or lack of an obvious parameter fix are not by
  themselves sufficient reasons to pause. If no hard terminal condition is proven,
  choose an adjustment or retry and continue autonomy.

The controller enforces this exact action contract:
- adjust_parameters classifications: {sorted(FAILURE_ADJUSTABLE_CLASSIFICATIONS)}
- retry_same classifications: {sorted(FAILURE_RETRYABLE_CLASSIFICATIONS)}
- diagnostic_retry_same classifications: {sorted(FAILURE_DIAGNOSTIC_RETRY_CLASSIFICATIONS)}
- pause_for_human: any classification.
Set safe_to_automate=true for adjust_parameters or retry_same. Set it to false for
pause_for_human.

When choosing adjust_parameters, choose the most informative justified change set. Use
changes for ordinary tuning parameters and recovery_changes for Recovery Registry
controls. A recovery-only repair must preserve candidate and use changes=[]. Up to
{self.max_parameters_per_round} tuning parameters and
{self.max_recovery_parameter_changes} recovery parameters are allowed. Multiple changes are allowed when
they form a reasoned coupled diagnostic or recovery experiment. Explain the interaction and give an
explicit constraint check for every changed parameter. Grid-step budgets are:
{yaml.safe_dump(self.change_policy, allow_unicode=True, sort_keys=False)}
For retry_same, diagnostic_retry_same or pause_for_human, return the current candidate unchanged, empty
changes and recovery_changes arrays, change_strategy=none, and empty interaction_analysis and
constraint_checks.

Allowed values:
{yaml.safe_dump(self.config['search_limits'], allow_unicode=True, sort_keys=False)}

Recovery Registry (failure repair only; not part of Active Search):
{yaml.safe_dump(self.recovery_parameter_registry, allow_unicode=True, sort_keys=False)}

Never modify topology, model, image, network, paths, benchmark, or system state. Do not
edit files. Return only schema-valid JSON. You may make an auditable, hypothesis-driven
parameter experiment inside the allowed values even when evidence is not conclusive.
A healthy-service benchmark-harness failure must use retry_same as specified above.

Embedded evidence:
{json.dumps(failure_evidence, ensure_ascii=False)}
"""
        analysis_dir = round_dir / "06_agent_analysis"
        prompt_path = analysis_dir / "failure_agent_prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        decision_path = analysis_dir / "failure_decision.json"
        result = None
        protocol_failures: list[str] = []
        protocol_attempts = self.max_agent_protocol_retries + 1
        for protocol_attempt in range(1, protocol_attempts + 1):
            retry_prompt = prompt
            if protocol_failures:
                retry_prompt += (
                    "\n\nThe prior failure-analysis provider call did not satisfy "
                    "the structured-output contract. Return one fresh complete JSON "
                    "decision. Failure detail: " + protocol_failures[-1][-2000:]
                )
            try:
                result = run_structured_agent(
                    self.agent_config,
                    prompt=retry_prompt,
                    schema_path=self.failure_schema_path(session_dir),
                    output_path=decision_path,
                    cwd=KB_ROOT,
                    allowed_dir=round_dir,
                )
            except Exception as exc:
                protocol_failures.append(
                    f"attempt {protocol_attempt}/{protocol_attempts}: "
                    f"{type(exc).__name__}: {exc}"
                )
                if protocol_attempt < protocol_attempts:
                    continue
                raise AgentProtocolError(
                    "Failure-analysis Agent raised exceptions after all attempts: "
                    + protocol_failures[-1][-2000:]
                ) from exc
            suffix = f"failure_agent.protocol_{protocol_attempt:02d}"
            (analysis_dir / f"{suffix}.events.jsonl").write_text(
                result.stdout, encoding="utf-8"
            )
            (analysis_dir / f"{suffix}.stderr.log").write_text(
                result.stderr, encoding="utf-8"
            )
            if result.returncode == 0 and decision_path.exists():
                try:
                    decision = json.loads(decision_path.read_text(encoding="utf-8"))
                    self.validate_failure_decision(session_dir, decision, current)
                except (KeyError, ValueError, json.JSONDecodeError) as exc:
                    invalid_copy = decision_path.with_name(
                        f"failure_decision.invalid_semantic_{protocol_attempt:02d}.json"
                    )
                    shutil.copyfile(decision_path, invalid_copy)
                    protocol_failures.append(
                        f"attempt {protocol_attempt}/{protocol_attempts}: "
                        f"semantic validation rejected the decision: {exc}"
                    )
                    if protocol_attempt < protocol_attempts:
                        continue
                    raise AgentProtocolError(
                        "Failure Agent exhausted semantic reselection attempts: "
                        + protocol_failures[-1][-2000:]
                    ) from exc
                break
            protocol_failures.append(
                f"attempt {protocol_attempt}/{protocol_attempts}: "
                f"{result.provider}: {result.stderr[-2000:]}"
            )
        assert result is not None
        (analysis_dir / "failure_agent_events.jsonl").write_text(
            result.stdout, encoding="utf-8"
        )
        (analysis_dir / "failure_agent_stderr.log").write_text(
            result.stderr, encoding="utf-8"
        )
        if result.returncode != 0 or not decision_path.exists():
            raise AgentProtocolError(
                f"{result.provider} failure analysis failed after "
                f"{protocol_attempts} protocol attempts: {result.stderr[-2000:]}"
            )
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        self.write_selected_portrait_evidence(
            round_dir,
            decision["changes"],
            prefix="failure_selected",
        )
        if decision.get("recovery_changes"):
            save_yaml(
                analysis_dir / "failure_selected_recovery_parameters.yaml",
                {
                    "registry": self.recovery_parameter_registry,
                    "current": self.runtime_recovery_values,
                    "changes": decision["recovery_changes"],
                },
            )
        self.refresh_runtime_rules(session_dir, round_dir)
        return decision

    @staticmethod
    def round_launch_profile(round_dir: Path) -> str | None:
        env_path = round_dir / "02_parameters" / "candidate.env"
        if not env_path.is_file():
            return None
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("LAUNCH_PROFILE="):
                continue
            values = shlex.split(line.split("=", 1)[1])
            if len(values) != 1 or values[0] not in {
                B0_LAUNCH_PROFILE,
                "explicit_candidate",
            }:
                raise ValueError(f"Invalid archived LAUNCH_PROFILE line: {line!r}")
            return values[0]
        return None

    def deterministic_startup_port_retry(
        self,
        round_dir: Path,
        current: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Retry an unchanged launch when vLLM's dynamic ZMQ port race is proven."""
        runtime_dir = round_dir / "04_runtime"
        if (runtime_dir / "SERVICE_READY").exists():
            return None
        master_path = runtime_dir / "master.log"
        if not master_path.is_file():
            return None
        text = master_path.read_text(encoding="utf-8", errors="replace")
        address_match = re.search(
            r"(?:ZMQError|zmq\.error\.ZMQError).*?Address already in use.*?"
            r"(?:addr=)?['\"]?(tcp://[^'\"\s)]+)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        api_server_died = re.search(
            r"Process ApiServer_\d+.*?(?:died with exit code|exited)",
            text,
            re.IGNORECASE,
        )
        if not address_match or not api_server_died:
            return None
        address = address_match.group(1)
        return {
            "summary": (
                "vLLM API startup hit a proven transient ZMQ port collision at "
                f"{address}; retrying the identical launch profile and candidate."
            ),
            "classification": "transient_infrastructure",
            "root_cause": (
                "The multi-API-server launcher selected a TCP address that was no "
                "longer free when the child process bound its ZMQ ROUTER socket."
            ),
            "evidence": [
                f"master.log reports Address already in use at {address}.",
                "An ApiServer child exited before SERVICE_READY was created.",
                "No serving parameter change is required or permitted for this retry.",
            ],
            "action": "retry_same",
            "safe_to_automate": True,
            "change_strategy": "none",
            "interaction_analysis": [],
            "constraint_checks": [],
            "changes": [],
            "candidate": current,
        }

    def deterministic_engine_frontend_handshake_retry(
        self,
        round_dir: Path,
        current: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Retry a proven pre-service DP front-end coordination timeout."""
        runtime_dir = round_dir / "04_runtime"
        if (runtime_dir / "SERVICE_READY").exists():
            return None
        texts: list[str] = []
        for path in (
            runtime_dir / "master.log",
            runtime_dir / "worker.log",
            runtime_dir / "run_status.json",
            round_dir / "05_results" / "failure.yaml",
        ):
            if path.is_file():
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
        evidence = "\n".join(texts)
        handshake_timeout = (
            "Did not receive response from front-end process within 5 minutes"
            in evidence
        )
        partial_process_set = bool(
            re.search(
                r"LeaseProcessesPartialFailure|active.?[=:] ?1.*inactive.?[=:] ?1",
                evidence,
                re.IGNORECASE | re.DOTALL,
            )
        )
        unsafe_signature = re.search(
            r"out of memory|(?:NPU|device) OOM|invalid (?:argument|parameter)|"
            r"HCCL.*(?:error|failed)|traceback.*candidate",
            evidence,
            re.IGNORECASE | re.DOTALL,
        )
        if not handshake_timeout or not partial_process_set or unsafe_signature:
            return None
        return {
            "summary": (
                "EngineCore timed out before SERVICE_READY while waiting for the "
                "DP front-end and the Lease reported a partial process set; retrying "
                "the identical candidate within the infrastructure budget."
            ),
            "classification": "transient_infrastructure",
            "root_cause": (
                "The DP engine/front-end startup coordination did not complete "
                "within five minutes; no parameter, OOM, or HCCL signature proves "
                "that changing the serving candidate is corrective."
            ),
            "evidence": [
                "SERVICE_READY was never created.",
                "Logs contain the exact five-minute front-end response timeout.",
                "The Lease reported one active and one inactive process.",
            ],
            "action": "retry_same",
            "safe_to_automate": True,
            "change_strategy": "none",
            "interaction_analysis": [],
            "constraint_checks": [],
            "changes": [],
            "candidate": current,
        }

    def deterministic_fused_moe_shared_expert_recovery(
        self,
        round_dir: Path,
        current: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Disable only the split shared-expert path after its exact validator fails."""
        parameter = "additional_config__multistream_overlap_shared_expert"
        if parameter not in self.recovery_parameter_registry:
            return None
        if self.runtime_recovery_values.get(parameter) is not True:
            return None
        texts: list[str] = []
        for path in (
            round_dir / "04_runtime" / "master.log",
            round_dir / "04_runtime" / "worker.log",
            round_dir / "05_results" / "failure.yaml",
        ):
            if path.is_file():
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
        evidence = "\n".join(texts)
        signature = (
            "FusedMoE shared experts split computation does not match "
            "the integrated computation"
        )
        if signature not in evidence:
            return None
        return {
            "summary": (
                "The exact FusedMoE split/shared-expert consistency validator "
                "failed; disabling only the multistream split shared-expert path."
            ),
            "classification": "model_or_runtime_bug",
            "root_cause": (
                "multistream_overlap_shared_expert registers the failing split-versus-"
                "integrated consistency check during weight processing. The exact "
                "validator signature proves that this recovery control gates the path."
            ),
            "evidence": [
                f"Archived runtime logs contain the exact validator signature: {signature}.",
                "The pinned fused_moe.py registers this validator only when "
                "multistream_overlap_shared_expert is enabled.",
                "The service failed before SERVICE_READY, so no Benchmark parameter "
                "change is implicated.",
            ],
            "action": "adjust_parameters",
            "safe_to_automate": True,
            "change_strategy": "single_parameter",
            "interaction_analysis": [
                "The recovery disables the split overlap path while preserving the "
                "integrated shared-expert computation and every tuning parameter."
            ],
            "constraint_checks": [
                "false is an allowed value in the frozen Recovery Registry and is "
                "injected through additional_config.multistream_overlap_shared_expert."
            ],
            "changes": [],
            "recovery_changes": [
                {
                    "parameter": parameter,
                    "before": True,
                    "after": False,
                    "rationale": (
                        "The exact failing validator is conditional on this split "
                        "shared-expert overlap control."
                    ),
                }
            ],
            "candidate": current,
        }

    def deterministic_mlapo_vector_timeout_recovery(
        self,
        round_dir: Path,
        current: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Disable MLAPO when its fused weight-processing path times out exactly.

        A worker that dies in this path commonly makes the remaining DP ranks emit
        Gloo ``Connection closed by peer`` errors.  Those later errors are a
        consequence, not the root cause, so this signature must be evaluated before
        generic communication recovery or Agent classification.
        """
        if current.get("mlapo") is not True:
            return None
        if False not in self.config.get("search_limits", {}).get("mlapo", []):
            return None
        texts: list[str] = []
        for path in (
            round_dir / "04_runtime" / "master.log",
            round_dir / "04_runtime" / "worker.log",
            round_dir / "05_results" / "failure.yaml",
        ):
            if path.is_file():
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
        evidence = "\n".join(texts)
        mlapo_path = (
            "_process_weights_for_fused_mlapo" in evidence
            or "attention/sfa_v1.py" in evidence
        )
        vector_timeout = (
            "Vector core execution timed out" in evidence
            and "aclnnMuls" in evidence
        )
        if not mlapo_path or not vector_timeout:
            return None

        candidate = copy.deepcopy(current)
        candidate["mlapo"] = False
        return {
            "summary": (
                "An NPU vector-core timeout occurred inside the fused MLAPO weight "
                "processing path; disabling MLAPO while preserving all other "
                "candidate and recovery parameters."
            ),
            "classification": "model_or_runtime_bug",
            "root_cause": (
                "The first actionable worker exception is aclnnMuls vector-core "
                "timeout in _process_weights_for_fused_mlapo. Subsequent Gloo peer "
                "closures are cascading failures after that worker exits."
            ),
            "evidence": [
                "The archived traceback enters attention/sfa_v1.py "
                "_process_weights_for_fused_mlapo during process_weights_after_loading.",
                "That traceback reports aclnnMuls and 'Vector core execution timed out'.",
                "Gloo connection-closed messages occur only after the MLAPO worker "
                "exception and therefore do not justify an unchanged retry.",
            ],
            "action": "adjust_parameters",
            "safe_to_automate": True,
            "change_strategy": "single_parameter",
            "interaction_analysis": [],
            "constraint_checks": [
                "mlapo=false is in the frozen Active Search grid and bypasses only "
                "the directly implicated fused MLAPO path."
            ],
            "changes": [
                {
                    "parameter": "mlapo",
                    "before": True,
                    "after": False,
                    "rationale": (
                        "Bypass the exact fused MLAPO weight-processing path that "
                        "raised the NPU vector-core timeout."
                    ),
                }
            ],
            "recovery_changes": [],
            "candidate": candidate,
        }

    def pending_historical_deterministic_recovery(
        self,
        session_dir: Path,
        current: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return an evidenced repair from an earlier failed round still pending.

        Infrastructure failures in a later retry must not erase an actionable
        parameter/runtime root cause from an earlier attempt.  A repair is pending
        only while the current candidate/recovery state still has the exact value
        implicated by that earlier signature.
        """
        round_dirs = sorted(
            (path for path in session_dir.glob("round_*") if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
        for source_round in round_dirs:
            decision = self.deterministic_fused_moe_shared_expert_recovery(
                source_round, current
            ) or self.deterministic_mlapo_vector_timeout_recovery(
                source_round, current
            )
            if decision is None:
                continue
            decision = copy.deepcopy(decision)
            decision["summary"] = (
                f"Pending evidenced recovery from {source_round.name}: "
                + str(decision["summary"])
            )
            decision["evidence"] = [
                f"Cross-round recovery source: {source_round.name}.",
                *list(decision.get("evidence", [])),
                "A later independent failure does not mark this earlier corrective "
                "action as applied or disproven.",
            ]
            return decision
        return None

    def deterministic_hccl_communicator_retry(
        self,
        round_dir: Path,
        current: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Retry a startup failure proven to be HCCL communicator establishment."""
        runtime_dir = round_dir / "04_runtime"
        if (runtime_dir / "SERVICE_READY").exists():
            return None
        texts: list[str] = []
        for path in (
            runtime_dir / "master.log",
            runtime_dir / "worker.log",
            round_dir / "05_results" / "failure.yaml",
        ):
            if path.is_file():
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
        evidence = "\n".join(texts)
        exact_signature = (
            "hcclCommInitRootInfoConfig" in evidence
            and "ERR02200" in evidence
            and "HcclAllGather" in evidence
            and "EI0006" in evidence
        )
        hard_block = MANUAL_INTERVENTION_SIGNATURES.search(evidence)
        parameter_failure = re.search(
            r"(?:NPU|device).*out of memory|\bOOM\b|invalid (?:argument|parameter)",
            evidence,
            re.IGNORECASE,
        )
        if not exact_signature or hard_block or parameter_failure:
            return None
        return {
            "summary": (
                "Startup reached model profile_run but HCCL communicator creation "
                "and its DP all-gather socket establishment timed out; retrying under "
                "the infrastructure budget instead of terminating autonomous work."
            ),
            "classification": "network_or_hccl",
            "root_cause": (
                "Cross-node HCCL communicator establishment failed with ERR02200, "
                "followed by EI0006 socket timeout in HcclAllGather. No image, model, "
                "permission, OOM, or invalid-parameter hard block is proven."
            ),
            "evidence": [
                "Logs contain hcclCommInitRootInfoConfig and ERR02200.",
                "The failed collective is HcclAllGather and reports EI0006 socket timeout.",
                "The service failed before SERVICE_READY without a hard terminal signature.",
            ],
            "action": "retry_same",
            "safe_to_automate": True,
            "change_strategy": "none",
            "interaction_analysis": [],
            "constraint_checks": [],
            "changes": [],
            "recovery_changes": [],
            "candidate": current,
        }

    def validate_runtime_recovery_values(
        self, values: dict[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(values, dict):
            raise ValueError("runtime_recovery_parameters must be a mapping")
        if set(values) != set(self.recovery_parameter_registry):
            missing = sorted(set(self.recovery_parameter_registry) - set(values))
            extra = sorted(set(values) - set(self.recovery_parameter_registry))
            raise ValueError(
                "Recovery parameter schema mismatch; "
                f"missing={missing}, extra={extra}"
            )
        resolved = copy.deepcopy(values)
        for name, value in resolved.items():
            allowed = self.recovery_parameter_registry[name]["allowed_values"]
            if value not in allowed:
                raise ValueError(
                    f"Recovery parameter {name}={value!r} is outside {allowed!r}"
                )
        return resolved

    def validate_recovery_changes(
        self,
        changes: list[dict[str, Any]],
        current_values: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if len(changes) > self.max_recovery_parameter_changes:
            raise ValueError(
                "Recovery decision exceeds max_recovery_parameter_changes="
                f"{self.max_recovery_parameter_changes}"
            )
        current = self.validate_runtime_recovery_values(
            current_values
            if current_values is not None
            else self.runtime_recovery_values
        )
        updated = copy.deepcopy(current)
        seen: set[str] = set()
        for item in changes:
            name = str(item.get("parameter", "")).removeprefix("--").replace(
                "-", "_"
            )
            if name in seen:
                raise ValueError(f"Duplicate recovery change for {name}")
            seen.add(name)
            if name not in self.recovery_parameter_registry:
                raise ValueError(
                    f"{name!r} is not in the evidence-gated Recovery Registry"
                )
            before = item.get("before")
            after = item.get("after")
            if before != current[name]:
                raise ValueError(
                    f"Recovery change before={before!r} does not match {name}="
                    f"{current[name]!r}"
                )
            if after == before:
                raise ValueError(f"Recovery change for {name} does not change value")
            allowed = self.recovery_parameter_registry[name]["allowed_values"]
            if after not in allowed:
                raise ValueError(
                    f"Recovery change {name}={after!r} is outside {allowed!r}"
                )
            if len(str(item.get("rationale", "")).strip()) < 12:
                raise ValueError(f"Recovery rationale for {name} is too short")
            updated[name] = copy.deepcopy(after)
        return updated

    def validate_failure_decision(
        self,
        session_dir: Path,
        decision: dict[str, Any],
        current: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Validate a failure decision and return its known-success rollback, if any."""
        action = decision["action"]
        classification = decision["classification"]
        candidate = decision["candidate"]
        changes = decision["changes"]
        recovery_changes = decision.get("recovery_changes", [])
        allowed_classifications = FAILURE_ACTION_CLASSIFICATIONS.get(action)
        if allowed_classifications is None:
            raise ValueError(f"Unsupported failure recovery action: {action!r}")
        if classification not in allowed_classifications:
            allowed = ", ".join(sorted(allowed_classifications))
            raise ValueError(
                f"Failure classification {classification!r} cannot use action "
                f"{action!r}; allowed classifications: {allowed}"
            )
        if action == "pause_for_human":
            if decision["safe_to_automate"]:
                raise ValueError(
                    "pause_for_human requires safe_to_automate=false"
                )
        elif not decision["safe_to_automate"]:
            raise ValueError(
                f"{action} requires safe_to_automate=true"
            )
        if action == "adjust_parameters":
            if not changes and not recovery_changes:
                raise ValueError(
                    "adjust_parameters requires a tuning or Recovery Registry change"
                )
            if changes:
                failure_policy = self.effective_change_policy()
                # Failure recovery is Agent-owned and evidence-driven. Normal
                # exploration curricula may prefer or require 2+ coordinated
                # parameters, but must not reject a precise one-parameter repair.
                failure_policy["minimum_parameters_per_round"] = 1
                failure_policy["max_parameters_per_round"] = (
                    self.max_parameters_per_round
                )
                self.validate_candidate(
                    current,
                    candidate,
                    changes,
                    decision,
                    failure_policy,
                )
            elif candidate != current:
                raise ValueError(
                    "A recovery-only adjustment must preserve the tuning candidate"
                )
            self.validate_recovery_changes(recovery_changes)
            if recovery_changes:
                return None
            known_success = self.successful_candidate(session_dir, candidate)
            if self.candidate_was_attempted(session_dir, candidate) and not known_success:
                raise ValueError(
                    "Failure recovery proposed a previously failed candidate"
                )
            return known_success
        else:
            if candidate != current or changes or recovery_changes:
                raise ValueError(
                    f"{action} must preserve the candidate and have no changes"
                )
            self.validate_no_change_metadata(decision)
        return None

    def deterministic_benchmark_retry(
        self,
        round_dir: Path,
        current: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return a bounded same-candidate retry for a clean harness failure."""
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
            zero_measurement_signature = (
                "Cannot compile GenerativeMetrics: No measurement start or end "
                "times available" in text
            )
            serving_failure_signature = re.search(
                r"(?:NPU|device).*out of memory|HCCL.*(?:error|failed)|"
                r"EngineCore.*(?:error|failed)|Connection refused|"
                r"Application startup failed",
                "\n".join(
                    path.read_text(encoding="utf-8", errors="replace")
                    for path in (runtime_dir / "master.log", runtime_dir / "worker.log")
                    if path.is_file()
                ),
                re.IGNORECASE,
            )
            if zero_measurement_signature and not serving_failure_signature:
                return {
                    "summary": (
                        "GuideLLM failed to compile one case because its measurement "
                        "window was empty after SERVICE_READY; retrying the identical "
                        "candidate within the benchmark/infrastructure budget."
                    ),
                    "classification": "benchmark_failure",
                    "root_cause": (
                        "The benchmark harness emitted the exact zero-measurement "
                        "metrics-compilation signature while the serving logs contain "
                        "no OOM, HCCL, EngineCore, or API-startup failure."
                    ),
                    "evidence": [
                        "SERVICE_READY was present before Benchmark execution.",
                        (
                            "benchmark_runner.log contains: Cannot compile "
                            "GenerativeMetrics: No measurement start or end times "
                            "available."
                        ),
                        "No serving-side dangerous signature was found.",
                    ],
                    "action": "retry_same",
                    "safe_to_automate": True,
                    "change_strategy": "none",
                    "interaction_analysis": [],
                    "constraint_checks": [],
                    "changes": [],
                    "candidate": current,
                }
            case_exits = re.findall(r"CASE COMPLETED .*? runner_exit=(\d+)", text)
            repetitions = int(self.benchmark["aligned_l1"]["repetitions"])
            expected_case_exits = 24 * repetitions
            if len(case_exits) != expected_case_exits or any(
                code != "0" for code in case_exits
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

    def deterministic_healthy_service_benchmark_retry(
        self,
        round_dir: Path,
        current: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Retry a harness-only failure after a proven healthy serving phase."""
        runtime_dir = round_dir / "04_runtime"
        if not (runtime_dir / "SERVICE_READY").exists():
            return None
        if (round_dir / "05_results" / "metrics.json").exists():
            return None
        evidence = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in (
                runtime_dir / "master.log",
                runtime_dir / "worker.log",
                runtime_dir / "benchmark_runner.log",
                runtime_dir / "run_status.json",
                round_dir / "05_results" / "failure.yaml",
            )
            if path.is_file()
        )
        static_asset_failure = re.search(
            r"dataset (?:slice|manifest).*(?:missing|mismatch|drift)|"
            r"数据集缺少切片|suite.*(?:fingerprint|contract).*(?:mismatch|drift)|"
            r"benchmark.*(?:schema|asset).*(?:missing|mismatch|drift)",
            evidence,
            re.IGNORECASE,
        )
        if static_asset_failure:
            # An identical candidate cannot repair a frozen suite/dataset
            # contract.  Let the Agent/hard-terminal policy produce an
            # actionable versioned-asset diagnosis instead of wasting NPU time.
            return None
        dangerous = re.search(
            r"(?:NPU|device).*out of memory|\bOOM\b|HCCL.*(?:error|failed)|"
            r"EngineCore.*(?:error|failed|died)|Application startup failed|"
            r"image.*(?:digest|identity|mismatch)|permission denied|"
            r"no such file or directory|invalid (?:argument|parameter)",
            evidence,
            re.IGNORECASE,
        )
        harness_failure = (runtime_dir / "BENCHMARK_FAILED").exists() or re.search(
            r"BENCHMARK_FAILED|CASE FAILED|benchmark.*(?:failed|timeout)|"
            r"GuideLLM|GenerativeMetrics|metrics(?:\.json)?.*(?:absent|missing)|"
            r"MASTER_DONE exists but metrics\.json is absent",
            evidence,
            re.IGNORECASE,
        )
        if dangerous or not harness_failure:
            return None
        return {
            "summary": (
                "The serving phase reached SERVICE_READY and the archived evidence "
                "confines the failure to the Benchmark harness; retrying the identical "
                "candidate within the bounded recovery budget."
            ),
            "classification": "benchmark_failure",
            "root_cause": (
                "Benchmark execution or metrics compilation ended without a valid "
                "metrics artifact while serving logs contain no dangerous signature."
            ),
            "evidence": [
                "SERVICE_READY exists.",
                "A Benchmark failure/missing-metrics signature exists.",
                "No OOM, HCCL, EngineCore, identity, permission, path, or parameter-invalid signature exists.",
            ],
            "action": "retry_same",
            "safe_to_automate": True,
            "change_strategy": "none",
            "interaction_analysis": [],
            "constraint_checks": [],
            "changes": [],
            "candidate": current,
        }

    def prefer_bounded_diagnostic_over_pause(
        self,
        failure: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Use one bounded unchanged diagnostic run before an inconclusive pause."""
        if failure.get("action") != "pause_for_human":
            return failure
        classification = str(failure.get("classification", "unknown"))
        text = "\n".join(
            str(failure.get(key, "")) for key in ("summary", "root_cause", "evidence")
        )
        if self.hard_terminal_intervention(failure) is not None:
            return failure
        if classification in FAILURE_RETRYABLE_CLASSIFICATIONS:
            if (
                not self.hard_terminal_only
                and int(state.get("failure_retries", 0))
                >= self.max_same_candidate_retries
            ):
                return failure
            return {
                **failure,
                "summary": (
                    "The Agent requested a pause, but no hard terminal dependency is "
                    "proven and infrastructure retry budget remains; continuing the "
                    "autonomous retry. " + str(failure.get("summary", ""))
                ),
                "action": "retry_same",
                "safe_to_automate": True,
                "change_strategy": "none",
                "interaction_analysis": [],
                "constraint_checks": [],
                "changes": [],
                "recovery_changes": [],
                "candidate": state["current_candidate"],
            }
        if classification not in FAILURE_DIAGNOSTIC_RETRY_CLASSIFICATIONS:
            return failure
        if (
            not self.hard_terminal_only
            and int(state.get("failure_diagnostic_retries", 0))
            >= self.max_agent_diagnostic_retries
        ):
            return failure
        return {
            **failure,
            "summary": (
                "The Agent could not prove an automatic repair, but no explicit "
                "manual-only dependency is present; running one unchanged diagnostic "
                "attempt before pausing. " + str(failure.get("summary", ""))
            ),
            "action": "diagnostic_retry_same",
            "safe_to_automate": True,
            "change_strategy": "none",
            "interaction_analysis": [],
            "constraint_checks": [],
            "changes": [],
            "recovery_changes": [],
            "candidate": state["current_candidate"],
        }

    @staticmethod
    def hard_terminal_intervention(failure: dict[str, Any]) -> dict[str, Any] | None:
        """Return actionable operator guidance only for a proven immutable block."""
        if str(failure.get("classification", "")) not in {
            "image_or_dependency",
            "parameter_invalid",
            "model_or_runtime_bug",
            "unknown",
        }:
            return None
        text = "\n".join(
            str(failure.get(key, ""))
            for key in ("summary", "root_cause", "evidence")
        )
        if not MANUAL_INTERVENTION_SIGNATURES.search(text):
            return None
        if re.search(
            r"\b(?:no|not|without)\b[^\n]{0,80}"
            r"(?:image|digest|identity|permission|credential|model|topology|state)"
            r"[^\n]{0,80}(?:mismatch|missing|invalid|corrupt|denied)",
            text,
            re.IGNORECASE,
        ):
            return None
        categories = (
            (
                "image_or_version_identity",
                r"image.*(?:digest|identity|mismatch|not approved)|(?:digest|commit).*mismatch",
                [
                    "Provide or approve an image whose digest and pinned vLLM/vLLM-Ascend commits match the Session contract.",
                    "Start a new Session after changing image identity; do not rewrite the frozen Session.",
                ],
            ),
            (
                "permission_or_credential",
                r"permission denied|access denied|credential|api key|secret|authentication failed",
                [
                    "Restore the missing permission or credential in the server-owned secret/environment file.",
                    "Run the read-only preflight, then authorize resume of the preserved Session.",
                ],
            ),
            (
                "missing_model_or_artifact",
                r"no such file or directory|missing (?:model|image)|model path",
                [
                    "Mount or provide the exact missing model/image artifact at the frozen path.",
                    "Verify its identity, then resume; use a new Session if the path or model identity must change.",
                ],
            ),
            (
                "immutable_contract_mismatch",
                r"configuration.*(?:invalid|mismatch)|state.*(?:inconsistent|corrupt)|topology.*mismatch",
                [
                    "Inspect the recorded mismatch/corruption evidence and restore the frozen contract from its audited source.",
                    "If model, topology, search space, or benchmark identity must change, start a new Session.",
                ],
            ),
        )
        category = "proven_manual_dependency"
        guidance = [
            "Resolve the explicit immutable dependency recorded in the evidence bundle.",
            "Run preflight and resume the preserved Session after the external fix.",
        ]
        for name, pattern, instructions in categories:
            if re.search(pattern, text, re.IGNORECASE):
                category = name
                guidance = instructions
                break
        return {
            "required": True,
            "category": category,
            "summary": str(failure.get("summary", "")),
            "evidence": list(failure.get("evidence", [])),
            "operator_steps": guidance,
            "resume_condition": (
                "The external immutable dependency is fixed and the frozen-Session "
                "read-only preflight passes."
            ),
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
                session_dir / "00_search_space" / "parameter_portraits.agent.yaml",
                max_chars=120000,
            ),
            "runtime_rules": (
                {
                    "rules": self.runtime_rule_store.data.get("rules", []),
                    "quarantines": self.runtime_rule_store.data.get("quarantines", []),
                    "proposals": self.runtime_rule_store.data.get("proposals", []),
                }
                if self.runtime_rule_store
                else None
            ),
        }
        analysis_dir = failed_round_dir / "06_agent_analysis"
        save_json(analysis_dir / "recovery_evidence_bundle.json", evidence)
        prompt = f"""You are the tuning analyst recovering from a proven invalid
vLLM-Ascend candidate. The frozen runtime contract is: {self.runtime_guardrail}.

The failed candidate has already been rejected and must never be resubmitted. The
failure analyst recommended a rollback to a previously measured, known-good candidate.
Do not resubmit that known-good candidate either. Starting from the known-good
candidate, choose the smallest safe, evidence-based, untested change set of 1 to
{self.max_parameters_per_round} parameters, or return stop_complete if no useful safe
untested change remains.

All successful and failed configurations are listed in attempted_history. Never repeat
any of them. Treat parameter_invalid and parameter_oom configurations as excluded.
Preserve the fixed model, image, DP/TP, Pod/NPU topology, network, paths,
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
        result = None
        protocol_failures: list[str] = []
        protocol_attempts = self.max_agent_protocol_retries + 1
        for protocol_attempt in range(1, protocol_attempts + 1):
            try:
                result = run_structured_agent(
                    self.agent_config,
                    prompt=prompt,
                    schema_path=self.agent_schema_path(session_dir),
                    output_path=decision_path,
                    cwd=KB_ROOT,
                    allowed_dir=failed_round_dir,
                )
            except Exception as exc:
                protocol_failures.append(
                    f"attempt {protocol_attempt}/{protocol_attempts}: "
                    f"{type(exc).__name__}: {exc}"
                )
                if protocol_attempt < protocol_attempts:
                    continue
                raise AgentProtocolError(
                    "Recovery Agent provider raised exceptions after all attempts: "
                    + protocol_failures[-1][-2000:]
                ) from exc
            if result.returncode == 0 and decision_path.exists():
                break
            protocol_failures.append(
                f"attempt {protocol_attempt}/{protocol_attempts}: "
                f"{result.provider}: {result.stderr[-2000:]}"
            )
        assert result is not None
        (analysis_dir / "recovery_agent_events.jsonl").write_text(
            result.stdout, encoding="utf-8"
        )
        (analysis_dir / "recovery_agent_stderr.log").write_text(
            result.stderr, encoding="utf-8"
        )
        if result.returncode != 0 or not decision_path.exists():
            raise AgentProtocolError(
                f"{result.provider} recovery analysis failed after "
                f"{protocol_attempts} attempts: {result.stderr[-2000:]}"
            )
        try:
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            if decision["action"] == "stop_complete":
                if decision["candidate"] != rollback or decision["changes"]:
                    raise ValueError("stop_complete must preserve the known-good candidate")
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
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            raise AgentProtocolError(
                "Rollback recovery Agent returned a semantically invalid decision: "
                f"{exc}"
            ) from exc
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
        launch_profile: str | None = None,
    ) -> tuple[Path, str | None, str]:
        next_dir = self.round_dir(session_dir, index, label)
        state["pending_submission"] = {
            "round_index": index,
            "round_label": label,
            "candidate": candidate,
            "runtime_recovery_parameters": copy.deepcopy(
                self.runtime_recovery_values
            ),
            "prepared_at": now(),
        }
        self.save_state(state)
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
        save_yaml(
            next_dir / "02_parameters" / "runtime_recovery_parameters.yaml",
            self.runtime_recovery_values,
        )
        task_id, run_id = self.submit(
            next_dir,
            label,
            candidate,
            dry_run=False,
            launch_profile=launch_profile,
        )
        # Commit the external identity immediately after submit returns. This
        # closes the larger crash window before each caller's bookkeeping and
        # lets Supervisor resume the exact task/run instead of resubmitting.
        state.update(
            round_index=index,
            round_label=label,
            active_task_id=task_id,
            active_run_id=run_id,
            current_candidate=candidate,
            round_submitted_at=now(),
            pending_submission=None,
        )
        self.save_state(state)
        return next_dir, task_id, run_id

    def create_session(self) -> tuple[Path, dict[str, Any]]:
        if STATE_FILE.exists():
            try:
                previous_state = load_controller_state()
                previous_dir = Path(previous_state.get("session_dir", ""))
                if previous_dir.is_dir():
                    save_json(
                        previous_dir
                        / f"controller_state_archived_{dt.datetime.now():%Y%m%d_%H%M%S}.json",
                        previous_state,
                    )
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        runtime_profile = self.config.get("runtime", {}).get("resolved_profile", {})
        session_prefix = str(
            runtime_profile.get("session_prefix", "glm52_continuous")
        )
        session_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_prefix).strip("_.-")
        if not session_prefix:
            raise ValueError("Runtime adapter session_prefix is empty after sanitization")
        session_id = session_prefix + "_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
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
            "round_label": str(
                self.config.get("initial_baseline", {}).get("label", "a0")
            ),
            "active_task_id": None,
            "active_run_id": None,
            "execution_mode": self.execution_mode,
            "executor_identity": self.executor_identity,
            "image_identity": self.image_identity,
            "runtime_identity": self.runtime_identity,
            "topology_profile": self.config.get("topology", {}).get("profile"),
            "topology_identity": {
                name: self.topology.get(name)
                for name in (
                    "executor",
                    "nodes",
                    "npu_per_node",
                    "data_parallel_size",
                    "data_parallel_size_local",
                    "tensor_parallel_size",
                    "worker_replicas",
                    "worker_data_parallel_start_rank",
                )
            },
            "search_limits_mode": self.config.get("resolved_search_space", {}).get(
                "mode", "manual"
            ),
            "search_space_profile": self.config.get("resolved_search_space", {}).get(
                "profile", "legacy_manual"
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
                self.lab.get("lease_name") if self.execution_mode == "ktp_lab" else None
            ),
            "current_candidate": self.config["baseline"],
            "runtime_recovery_parameters": copy.deepcopy(
                self.runtime_recovery_values
            ),
            "failure_retries": 0,
            "failure_adjustments": 0,
            "failure_diagnostic_retries": 0,
            "total_failure_recovery_rounds": 0,
            "round_submitted_at": None,
            "created_at": now(),
            "updated_at": now(),
        }
        save_json(STATE_FILE, state)
        save_yaml(session_dir / "session_config.yaml", self.config)
        save_yaml(session_dir / "image_version_manifest.yaml", self.image_manifest)
        return session_dir, state

    def import_completed_baseline(
        self,
        session_dir: Path,
        state: dict[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        """Seed a new Session from one identity-matched completed B0 round."""
        setting = self.config.get("baseline_reuse", {})
        source_value = setting.get("source_session") if isinstance(setting, dict) else None
        if not source_value:
            raise ValueError("baseline_reuse.source_session is required")
        source_session = Path(str(source_value)).expanduser().resolve()
        archive_root = ARCHIVE_ROOT.resolve()
        if not source_session.is_relative_to(archive_root) or source_session == session_dir:
            raise ValueError(
                "Reusable baseline Session must be a different directory below the "
                "current runtime experiments root"
            )
        source_config_path = source_session / "session_config.yaml"
        if not source_config_path.is_file():
            raise ValueError(f"Reusable baseline Session config is missing: {source_config_path}")
        source_config = load_yaml(source_config_path)

        source_profile = source_config.get("benchmark", {}).get("resolved_profile", {})
        source_mode = str(source_profile.get("mode", ""))
        source_definition_key = str(source_profile.get("definition_key", source_mode))
        source_definition = source_config.get("benchmark", {}).get(source_definition_key)
        source_identity = {
            "schema_version": "vllmtkb-benchmark-identity/v1",
            "profile": source_config.get("benchmark", {}).get("profile"),
            "mode": source_mode,
            "definition": source_definition,
        }
        identity_bytes = json.dumps(
            source_identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        source_identity["sha256"] = hashlib.sha256(identity_bytes).hexdigest()
        checks = {
            "benchmark_identity": source_identity == self.benchmark_identity,
            "image_manifest": source_config.get("image_identity", {}).get(
                "resolved_manifest"
            )
            == self.image_manifest,
            "topology_profile": source_config.get("topology", {}).get("profile")
            == self.config.get("topology", {}).get("profile"),
            "deployment": source_config.get("deployment")
            == self.config.get("deployment"),
            "baseline_definition": source_config.get("initial_baseline", {})
            .get("resolved_definition", {})
            .get("sha256")
            == self.config.get("initial_baseline", {})
            .get("resolved_definition", {})
            .get("sha256"),
        }
        failed_checks = [name for name, passed in checks.items() if not passed]
        if failed_checks:
            raise ValueError(
                "Reusable baseline identity mismatch: " + ", ".join(failed_checks)
            )

        source_rounds = sorted(source_session.glob("round_000_*"))
        if len(source_rounds) != 1:
            raise ValueError("Reusable baseline Session must contain exactly one round_000")
        source_round = source_rounds[0]
        source_metrics = source_round / "05_results" / "metrics.json"
        source_candidate = source_round / "02_parameters" / "candidate_params.yaml"
        if not source_metrics.is_file() or not source_candidate.is_file():
            raise ValueError("Reusable baseline round must contain metrics and candidate parameters")
        metrics = json.loads(source_metrics.read_text(encoding="utf-8"))
        candidate = load_yaml(source_candidate)
        if metrics.get("benchmark_mode") != self.benchmark_mode:
            raise ValueError("Reusable baseline metrics use a different benchmark mode")
        if set(candidate) != self.candidate_schema:
            raise ValueError("Reusable baseline candidate schema differs from this Session")

        label = str(self.config.get("initial_baseline", {}).get("label", "b0"))
        target_round = self.round_dir(session_dir, 0, label)
        self.write_context(target_round, state)
        self.run_query(target_round)
        # Import measured evidence only. Old Agent analysis/context is excluded
        # so the new strategy sees B0, never the old Session's decisions.
        for child in ("02_parameters", "03_submission", "04_runtime", "05_results"):
            shutil.copytree(
                source_round / child,
                target_round / child,
                dirs_exist_ok=True,
                copy_function=shutil.copy2,
            )
        state["current_candidate"] = candidate
        state["baseline_reuse"] = {
            "source_session": str(source_session),
            "source_round": source_round.name,
            "source_metrics_sha256": hashlib.sha256(
                source_metrics.read_bytes()
            ).hexdigest(),
            "benchmark_identity_sha256": self.benchmark_identity["sha256"],
            "identity_checks": checks,
            "imported_at": now(),
        }
        save_yaml(session_dir / "baseline_reuse.yaml", state["baseline_reuse"])
        self.reconcile_official_source_default_baseline(
            session_dir, target_round, state
        )
        self.save_state(state)
        return target_round, state

    def save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = now()
        save_json(STATE_FILE, state)
        # Keep a second complete copy of the same committed snapshot. A backup
        # that lagged by one transition could resurrect an already-submitted
        # round and cause duplicate work after storage corruption.
        save_json(STATE_FILE.with_name(STATE_FILE.name + ".previous"), state)

    def recover_pending_submission(self, state: dict[str, Any]) -> bool:
        pending = state.get("pending_submission")
        if not isinstance(pending, dict):
            return False
        session_dir = Path(str(state.get("session_dir", "")))
        round_dir = self.round_dir(
            session_dir,
            int(pending["round_index"]),
            str(pending["round_label"]),
        )
        submission_path = round_dir / "03_submission" / "submission.json"
        submission: dict[str, Any] = {}
        if submission_path.is_file():
            submission = json.loads(submission_path.read_text(encoding="utf-8"))
        run_id = str(submission.get("run_id", "")).strip()
        task_id = submission.get("task_id") or submission.get("lease_name")
        if not run_id and self.execution_mode == "ktp_lab":
            pointer = round_dir / "02_parameters" / "lab_active_run.env"
            if pointer.is_file():
                match = re.search(
                    r"^EXPERIMENT_RUN_ID=(?:'([^']+)'|\"([^\"]+)\"|(\S+))$",
                    pointer.read_text(encoding="utf-8"),
                    re.MULTILINE,
                )
                if match:
                    run_id = next(value for value in match.groups() if value)
                    task_id = self.lab.get("lease_name")
        if not run_id or not task_id:
            return False
        state.update(
            round_index=int(pending["round_index"]),
            round_label=str(pending["round_label"]),
            active_task_id=str(task_id),
            active_run_id=run_id,
            current_candidate=pending["candidate"],
            round_submitted_at=submission.get("submitted_at") or now(),
            pending_submission=None,
            status="running",
        )
        self.save_state(state)
        save_json(
            round_dir / "03_submission" / "submission_recovery.audit.json",
            {
                "recovered_at": now(),
                "run_id": run_id,
                "task_id": str(task_id),
                "source": (
                    str(submission_path)
                    if submission
                    else "lab_active_run.env submission intent"
                ),
            },
        )
        return True

    def reanalyze_current(self) -> dict[str, Any]:
        if not STATE_FILE.exists():
            raise RuntimeError("No controller state exists to reanalyze")
        state = load_controller_state()
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
        if not state.get("active_task_id") and not state.get("active_run_id"):
            # A paused Controller may be repaired and reanalyzed from already
            # archived metrics. Mark that round resumable so the validated
            # decision can enter the normal submit loop without rerunning it.
            state["status"] = "stopped_after_current_round"
        self.save_state(state)
        log(
            f"Reanalyzed saved round {state['round_label']}; "
            f"action={decision['action']}. No experiment was submitted."
        )
        return decision

    def upgrade_frozen_recovery_policy(self, source_path: Path) -> dict[str, Any]:
        """Auditably widen only unattended-recovery settings for an idle Session."""
        state = load_controller_state()
        retained_audit_run = (
            state.get("status") == "stopped_after_failed_round"
            and state.get("last_failure_classification") == "operator_stop_before_metrics"
            and not state.get("active_task_id")
        )
        paused_idle_run = False
        if state.get("status") == "paused_for_human" and state.get("active_task_id"):
            snapshot = self.task_snapshot(state.get("active_task_id"))
            process_counts = snapshot.get("process_counts", {})
            process_set_idle = bool(process_counts) and not any(
                int(count or 0) > 0
                for name, count in process_counts.items()
                if str(name).lower() not in {"idle", "inactive", "succeeded", "failed"}
            )
            paused_idle_run = bool(
                snapshot.get("terminal")
                or int(snapshot.get("active_pods") or 0) == 0
                or process_set_idle
            )
        if (state.get("active_task_id") and not paused_idle_run) or (
            state.get("active_run_id") and not retained_audit_run
            and not paused_idle_run
        ):
            raise RuntimeError("Recovery-policy upgrade requires an archived idle round")
        session_dir = Path(state["session_dir"])
        frozen_path = session_dir / "session_config.yaml"
        frozen = load_yaml(frozen_path)
        source = load_config(source_path)
        before = copy.deepcopy(frozen)

        def copy_if_higher(target: dict[str, Any], origin: dict[str, Any], key: str) -> None:
            if key not in origin:
                return
            old = int(target.get(key, 0))
            new = int(origin[key])
            if new < old:
                raise RuntimeError(f"Recovery upgrade cannot lower {key}: {old} -> {new}")
            target[key] = new

        frozen.setdefault("failure_recovery", {})
        source_recovery = source.get("failure_recovery", {})
        if bool(source_recovery.get("hard_terminal_only", False)):
            frozen["failure_recovery"]["hard_terminal_only"] = True
        for key in (
            "same_candidate_retries",
            "agent_diagnostic_retries",
            "parameter_adjustments",
            "total_recovery_rounds",
        ):
            copy_if_higher(
                frozen["failure_recovery"], source.get("failure_recovery", {}), key
            )
        copy_if_higher(frozen, source, "max_controller_recovery_attempts")
        frozen.setdefault("agent", {}).setdefault("settings", {})
        copy_if_higher(
            frozen["agent"]["settings"],
            source.get("agent", {}).get("settings", {}),
            "max_protocol_retries",
        )
        frozen.setdefault("lab", {})
        for key in (
            "readiness_wait_seconds",
            "submission_readiness_retry_limit",
        ):
            copy_if_higher(frozen["lab"], source.get("lab", {}), key)
        frozen.setdefault("benchmark", {}).setdefault("aligned_l1", {})
        for key in (
            "case_retry_limit",
            "runtime_retry_limit",
            "metrics_retry_limit",
            "total_full_retry_limit",
        ):
            copy_if_higher(
                frozen["benchmark"]["aligned_l1"],
                source.get("benchmark", {}).get("aligned_l1", {}),
                key,
            )
        frozen.setdefault("change_policy", {})
        copy_if_higher(
            frozen["change_policy"],
            source.get("change_policy", {}),
            "max_candidate_reselections",
        )
        changed = {
            "failure_recovery": frozen.get("failure_recovery"),
            "max_controller_recovery_attempts": frozen.get(
                "max_controller_recovery_attempts"
            ),
            "agent_protocol_retries": frozen.get("agent", {})
            .get("settings", {})
            .get("max_protocol_retries"),
            "lab_recovery": {
                key: frozen.get("lab", {}).get(key)
                for key in ("readiness_wait_seconds", "submission_readiness_retry_limit")
            },
            "benchmark_retries": {
                key: frozen.get("benchmark", {}).get("aligned_l1", {}).get(key)
                for key in (
                    "case_retry_limit",
                    "runtime_retry_limit",
                    "metrics_retry_limit",
                    "total_full_retry_limit",
                )
            },
            "max_candidate_reselections": frozen.get("change_policy", {}).get(
                "max_candidate_reselections"
            ),
        }
        audit = {
            "schema_version": "vllmtkb-recovery-policy-upgrade/v2",
            "upgraded_at": now(),
            "source": str(source_path.resolve()),
            "session": state.get("session_id"),
            "round": state.get("round_label"),
            "immutable_scope": (
                "model/image/topology/executor/search_limits/strategy/agent provider/"
                "benchmark identity and candidate remain frozen"
            ),
            "settings": changed,
        }
        snapshot = session_dir / f"session_config.before_recovery_upgrade.{dt.datetime.now():%Y%m%d_%H%M%S}.yaml"
        save_yaml(snapshot, before)
        save_yaml(frozen_path, frozen)
        save_json(session_dir / "recovery_policy_upgrade.audit.json", audit)
        return audit

    def retry_paused_current(self) -> dict[str, Any]:
        """Operator-authorized same-candidate diagnostic retry."""
        if not STATE_FILE.exists():
            raise RuntimeError("No controller state exists to retry")
        state = load_controller_state()
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
        self.runtime_recovery_values = self.validate_runtime_recovery_values(
            state.get(
                "runtime_recovery_parameters", self.runtime_recovery_values
            )
        )
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
            launch_profile=self.round_launch_profile(failed_round),
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

    def replay_unmeasured_candidate(self, source_round_name: str) -> dict[str, Any]:
        """Replay a prior candidate whose failure was not parameter-attributed.

        This is the crash-safe/operator repair path for infrastructure failures
        discovered only after later recovery work.  It never rewrites history:
        the interrupted current round is archived with an explicit audit record
        and the source candidate is submitted as a new retry round.
        """
        if not STATE_FILE.exists():
            raise RuntimeError("No controller state exists to repair")
        state = load_controller_state()
        self.assert_state_image_identity(state)
        session_dir = Path(state["session_dir"])
        self.load_session_sidecars(session_dir)
        source_round = session_dir / source_round_name
        if source_round.parent != session_dir or not source_round.is_dir():
            raise RuntimeError(f"Unknown Session round: {source_round_name!r}")
        source_failure = source_round / "05_results" / "failure.yaml"
        source_metrics = source_round / "05_results" / "metrics.json"
        source_params = source_round / "02_parameters" / "candidate_params.yaml"
        if not source_failure.is_file() or source_metrics.exists() or not source_params.is_file():
            raise RuntimeError(
                "Replay source must have candidate parameters and failure evidence, "
                "but no Benchmark metrics"
            )
        source_attempt = next(
            (
                item
                for item in self.attempted_history_summary(session_dir)
                if item.get("round") == source_round_name
            ),
            None,
        )
        if not source_attempt:
            raise RuntimeError("Replay source is not a terminal recorded attempt")
        if source_attempt.get("counts_as_parameter_experiment", False):
            raise RuntimeError(
                "Refusing to replay a candidate already covered by Benchmark or "
                "a parameter-attributed startup failure"
            )
        snapshot = self.task_snapshot(state.get("active_task_id"))
        if not snapshot.get("terminal") and int(snapshot.get("active_pods") or 0) > 0:
            raise RuntimeError(
                "Current task still has active processes; stop it before replay"
            )
        current_round = self.round_dir(
            session_dir, int(state["round_index"]), str(state["round_label"])
        )
        current_failure = current_round / "05_results" / "failure.yaml"
        current_metrics = current_round / "05_results" / "metrics.json"
        if current_metrics.exists():
            raise RuntimeError("Current round already has Benchmark metrics")
        if not current_failure.exists():
            save_yaml(
                current_failure,
                {
                    "detected_at": now(),
                    "reason": (
                        "Operator superseded this unmeasured recovery round to replay "
                        f"the earlier infrastructure-blocked candidate {source_round_name}."
                    ),
                    "classification": "operator_superseded_unmeasured_recovery",
                },
            )
        source_candidate = load_yaml(source_params)
        next_index = int(state["round_index"]) + 1
        candidate_index_match = re.search(r"_a(\d+)", source_round_name)
        candidate_index = (
            int(candidate_index_match.group(1))
            if candidate_index_match
            else int(state["candidate_index"])
        )
        retry_number = max(1, int(state.get("failure_retries", 0)) + 1)
        next_label = f"a{candidate_index}r{retry_number}"
        audit = {
            "authorized_at": now(),
            "source_round": source_round_name,
            "superseded_round": current_round.name,
            "retry_label": next_label,
            "candidate": source_candidate,
            "reason": (
                "The source candidate never produced Benchmark metrics and its "
                "failure was infrastructure/unattributed, so search coverage must "
                "replay it after the external fix."
            ),
        }
        save_json(
            current_round / "06_agent_analysis" / "unmeasured_candidate_replay.json",
            audit,
        )
        _, task_id, run_id = self.prepare_and_submit_round(
            session_dir,
            state,
            index=next_index,
            label=next_label,
            candidate=source_candidate,
            launch_profile=self.round_launch_profile(source_round),
        )
        state.update(
            round_index=next_index,
            candidate_index=candidate_index,
            round_label=next_label,
            active_task_id=task_id,
            active_run_id=run_id,
            current_candidate=source_candidate,
            failure_retries=retry_number,
            round_submitted_at=now(),
            status="retrying_infrastructure_failure",
            recovery_source_round=source_round_name,
            recovery_reason="replay_unmeasured_candidate_after_external_fix",
            last_failure_classification="transient_infrastructure",
            last_failure_summary=audit["reason"],
        )
        self.save_state(state)
        log(
            f"Replayed unmeasured {source_round_name} as {next_label} "
            f"task={task_id} run={run_id}"
        )
        return state

    def auto_retry_paused_current(self) -> dict[str, Any]:
        """Reopen a paused round for a fresh Agent-owned autonomous decision."""
        state = load_controller_state()
        if state.get("status") != "paused_for_human":
            raise RuntimeError("Automatic paused recovery requires paused_for_human")
        self.assert_state_image_identity(state)
        session_dir = Path(state["session_dir"])
        self.load_session_sidecars(session_dir)
        self.runtime_recovery_values = self.validate_runtime_recovery_values(
            state.get(
                "runtime_recovery_parameters", self.runtime_recovery_values
            )
        )
        failed_round = self.round_dir(
            session_dir, int(state["round_index"]), str(state["round_label"])
        )
        prior_decision = failed_round / "06_agent_analysis" / "failure_decision.json"
        if prior_decision.is_file():
            archive = prior_decision.with_name(
                "failure_decision.pre-agent-autonomy-v3.json"
            )
            if not archive.exists():
                shutil.copyfile(prior_decision, archive)
        decision = self.analyze_failure(
            session_dir, failed_round, state["current_candidate"]
        )
        decision = self.prefer_bounded_diagnostic_over_pause(decision, state)
        if decision["action"] == "pause_for_human":
            intervention = self.hard_terminal_intervention(decision)
            if intervention is None:
                raise RuntimeError(
                    "Non-terminal pause escaped autonomous recovery policy"
                )
            save_json(
                failed_round / "06_agent_analysis" / "human_intervention_required.json",
                intervention,
            )
            state["human_intervention"] = intervention
            self.save_state(state)
            raise RuntimeError(
                "Fresh Agent analysis proved a hard terminal dependency: "
                + str(decision["summary"])
            )
        self.validate_failure_decision(
            session_dir, decision, state["current_candidate"]
        )
        total_recovery_rounds = int(state.get("total_failure_recovery_rounds", 0)) + 1
        if (
            not self.hard_terminal_only
            and total_recovery_rounds > self.max_total_failure_recovery_rounds
        ):
            raise RuntimeError("Paused round exhausted its total failure-recovery budget")
        save_json(
            failed_round / "06_agent_analysis" / "post_pause_agent_decision.json",
            decision,
        )
        next_index = int(state["round_index"]) + 1
        if decision["action"] == "adjust_parameters":
            failure_adjustments = int(state.get("failure_adjustments", 0)) + 1
            if (
                not self.hard_terminal_only
                and failure_adjustments > self.max_parameter_failure_adjustments
            ):
                raise RuntimeError(
                    "Paused round exhausted its parameter failure-adjustment budget"
                )
            retry_number = 0
            next_candidate = decision["candidate"]
            next_recovery_values = self.validate_recovery_changes(
                decision.get("recovery_changes", []),
                state.get(
                    "runtime_recovery_parameters", self.runtime_recovery_values
                ),
            )
            self.runtime_recovery_values = copy.deepcopy(next_recovery_values)
            next_label = f"a{int(state['candidate_index'])}f{failure_adjustments}"
            next_status = "recovering_parameter_failure"
        elif decision["action"] == "diagnostic_retry_same":
            diagnostic_retries = int(state.get("failure_diagnostic_retries", 0)) + 1
            if (
                not self.hard_terminal_only
                and diagnostic_retries > self.max_agent_diagnostic_retries
            ):
                raise RuntimeError(
                    "Paused round exhausted its diagnostic retry budget"
                )
            retry_number = int(state.get("failure_retries", 0))
            failure_adjustments = int(state.get("failure_adjustments", 0))
            next_candidate = state["current_candidate"]
            next_recovery_values = state.get(
                "runtime_recovery_parameters", self.runtime_recovery_values
            )
            next_label = f"a{int(state['candidate_index'])}d{diagnostic_retries}"
            next_status = "retrying_agent_diagnosed_failure"
        else:
            diagnostic_retries = int(state.get("failure_diagnostic_retries", 0))
            retry_number = int(state.get("failure_retries", 0)) + 1
            if (
                not self.hard_terminal_only
                and retry_number > self.max_same_candidate_retries
            ):
                raise RuntimeError(
                    "Paused round exhausted its same-candidate retry budget"
                )
            failure_adjustments = int(state.get("failure_adjustments", 0))
            next_candidate = state["current_candidate"]
            next_recovery_values = state.get(
                "runtime_recovery_parameters", self.runtime_recovery_values
            )
            next_label = f"a{int(state['candidate_index'])}r{retry_number}"
            next_status = "retrying_infrastructure_failure"
        if decision["action"] == "adjust_parameters":
            diagnostic_retries = int(state.get("failure_diagnostic_retries", 0))
        _, task_id, run_id = self.prepare_and_submit_round(
            session_dir,
            state,
            index=next_index,
            label=next_label,
            candidate=next_candidate,
            launch_profile=(
                self.round_launch_profile(failed_round)
                if decision["action"] == "retry_same"
                else None
            ),
        )
        state.update(
            round_index=next_index,
            round_label=next_label,
            active_task_id=task_id,
            active_run_id=run_id,
            round_submitted_at=now(),
            failure_retries=retry_number,
            failure_adjustments=failure_adjustments,
            failure_diagnostic_retries=diagnostic_retries,
            total_failure_recovery_rounds=total_recovery_rounds,
            current_candidate=next_candidate,
            runtime_recovery_parameters=next_recovery_values,
            status=next_status,
            recovery_source_round=failed_round.name,
            recovery_reason=(
                f"agent_owned_{decision['classification']}_"
                f"{decision['action']}"
            ),
            last_failure_classification=decision["classification"],
            last_failure_summary=decision["summary"],
        )
        self.save_state(state)
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
                    self.history_summary(session_dir),
                    self.attempted_history_summary(session_dir),
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
        # Preserve the simulated identity as audit evidence without presenting
        # it as a resumable live run to the service recovery wrapper.
        state["last_dry_run_id"] = run_id
        state["active_run_id"] = None
        state["status"] = "dry_run_complete"
        self.save_state(state)
        log(f"Dry-run complete: {session_dir}")

    def start(self, *, resume: bool = False) -> None:
        def stop_was_requested_before_submission() -> bool:
            if not STOP_FILE.exists():
                return False
            state.update(status="stop_requested")
            self.save_state(state)
            log(
                "STOP_REQUESTED rechecked after analysis/recovery; refusing to "
                "submit another experiment."
            )
            return True

        if resume:
            if not STATE_FILE.exists():
                raise RuntimeError("No controller state exists to resume")
            state = load_controller_state()
            self.runtime_recovery_values = self.validate_runtime_recovery_values(
                state.get("runtime_recovery_parameters", self.runtime_recovery_values)
            )
            self.assert_state_image_identity(state)
            session_dir = Path(state["session_dir"])
            if not session_dir.is_dir():
                raise RuntimeError(f"Session directory does not exist: {session_dir}")
            self.load_session_sidecars(session_dir)
            self.write_decision_schemas(session_dir)
            pending = state.get("pending_submission")
            if self.recover_pending_submission(state):
                log("Recovered an interrupted submission transaction from its ledger.")
            elif isinstance(pending, dict):
                log(
                    "Interrupted submission has no external identity ledger; "
                    "replaying the same frozen submission intent."
                )
                _, task_id, run_id = self.prepare_and_submit_round(
                    session_dir,
                    state,
                    index=int(pending["round_index"]),
                    label=str(pending["round_label"]),
                    candidate=dict(pending["candidate"]),
                )
                state.update(active_task_id=task_id, active_run_id=run_id)
                self.save_state(state)
            if not state.get("active_task_id") or not state.get("active_run_id"):
                archived_round = self.round_dir(
                    session_dir,
                    int(state["round_index"]),
                    str(state["round_label"]),
                )
                resumable_statuses = {
                    "stopped_after_current_round",
                    "stopped_after_failed_round",
                    "budget_paused",
                    "topology_feasibility_passed",
                }
                has_terminal_artifact = (
                    archived_round / "05_results" / "metrics.json"
                ).exists() or (archived_round / "05_results" / "failure.yaml").exists()
                status_is_resumable = (
                    state.get("status") in resumable_statuses
                    or (
                        state.get("status") == "running"
                        and state.get("analysis_status") == "ready"
                    )
                )
                if not status_is_resumable or not has_terminal_artifact:
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
                    previous_state = load_controller_state()
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
            if self.config.get("baseline_reuse", {}).get("source_session"):
                round_dir, state = self.import_completed_baseline(session_dir, state)
                log(
                    f"Imported identity-matched {initial_label.upper()} evidence; "
                    "invoking Agent analysis without a baseline submission."
                )
                decision = self.analyze(
                    session_dir, round_dir, state["current_candidate"]
                )
                if decision["action"] == "stop_complete":
                    state.update(
                        status="completed_by_agent",
                        completion_summary=decision["summary"],
                    )
                    self.save_state(state)
                    return
                next_label = "a1"
                next_candidate = decision["candidate"]
                _, task_id, run_id = self.prepare_and_submit_round(
                    session_dir,
                    state,
                    index=1,
                    label=next_label,
                    candidate=next_candidate,
                )
                state.update(
                    round_index=1,
                    candidate_index=1,
                    round_label=next_label,
                    active_task_id=task_id,
                    active_run_id=run_id,
                    current_candidate=next_candidate,
                    status="running",
                    round_submitted_at=now(),
                )
                self.save_state(state)
                log(f"Submitted {next_label} task={task_id} run={run_id}")
            else:
                round_dir = self.round_dir(session_dir, 0, initial_label)
                self.write_context(round_dir, state)
                self.run_query(round_dir)
                save_yaml(
                    round_dir / "02_parameters" / "candidate_params.yaml",
                    state["current_candidate"],
                )
                save_yaml(
                    round_dir
                    / "02_parameters"
                    / "runtime_recovery_parameters.yaml",
                    self.runtime_recovery_values,
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
            local_metrics = round_dir / "05_results" / "metrics.json"
            if not state.get("active_run_id") and local_metrics.is_file():
                found = {name: False for name in REMOTE_ARTIFACTS}
                found["metrics.json"] = True
            else:
                found = self.collect_with_retry(state["active_run_id"], round_dir)
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
                pause_status = session_budget_pause_status(
                    int(state["candidate_index"]),
                    self.pause_after_candidate_index,
                    topology_feasibility_only=self.topology_feasibility_only,
                )
                if pause_status is not None:
                    self.write_comparison(session_dir, round_dir)
                    state.update(
                        status=pause_status,
                        active_task_id=None,
                        active_run_id=None,
                        budget_pause_after_candidate_index=(
                            self.pause_after_candidate_index
                        ),
                        topology_feasibility=(
                            "passed" if self.topology_feasibility_only else None
                        ),
                    )
                    self.save_state(state)
                    log(
                        f"Session paused at candidate index {state['candidate_index']} "
                        f"with status={pause_status}; no next experiment was submitted."
                    )
                    return
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
                        active_run_id=None,
                        completion_summary=decision["summary"],
                    )
                    self.save_state(state)
                    log(
                        "Codex determined that tuning is complete; no new round submitted."
                    )
                    return
                if stop_was_requested_before_submission():
                    state.update(
                        status="stopped_after_current_round",
                        active_task_id=None,
                    )
                    self.save_state(state)
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
                    failure_diagnostic_retries=0,
                    total_failure_recovery_rounds=0,
                    round_submitted_at=now(),
                    status="running",
                    controller_recovery_attempts=0,
                    controller_error=None,
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
                if self.topology_feasibility_only:
                    state.update(
                        status="topology_feasibility_failed",
                        active_task_id=None,
                        active_run_id=None,
                        topology_feasibility="failed",
                        last_failure_classification="topology_feasibility_probe_failed",
                        last_failure_summary="; ".join(reasons),
                    )
                    self.save_state(state)
                    log(
                        "Experimental topology baseline failed its one-shot "
                        "startup/Fast-C32 feasibility probe; no parameter recovery "
                        "or further topology budget will be consumed."
                    )
                    return
                log(
                    f"Round {state['round_label']} failed; invoking Codex failure analysis."
                )
                try:
                    failure = self.saved_failure_decision(
                        session_dir,
                        round_dir,
                        state["current_candidate"],
                    ) or self.analyze_failure(
                        session_dir,
                        round_dir,
                        state["current_candidate"],
                    )
                except AgentProtocolError:
                    # Deterministic decisions are availability fallbacks only. They
                    # never preselect the Agent's layer or candidate when the Agent
                    # provider is healthy.
                    failure = self.deterministic_startup_port_retry(
                        round_dir, state["current_candidate"]
                    ) or self.deterministic_engine_frontend_handshake_retry(
                        round_dir, state["current_candidate"]
                    ) or self.deterministic_fused_moe_shared_expert_recovery(
                        round_dir, state["current_candidate"]
                    ) or self.deterministic_mlapo_vector_timeout_recovery(
                        round_dir, state["current_candidate"]
                    ) or self.deterministic_hccl_communicator_retry(
                        round_dir, state["current_candidate"]
                    ) or self.deterministic_benchmark_retry(
                        round_dir, state["current_candidate"]
                    ) or self.deterministic_healthy_service_benchmark_retry(
                        round_dir, state["current_candidate"]
                    )
                    if failure is None:
                        raise
                    self.validate_failure_decision(
                        session_dir, failure, state["current_candidate"]
                    )
                    save_json(
                        round_dir
                        / "06_agent_analysis"
                        / "deterministic_provider_fallback_decision.json",
                        failure,
                    )
                    log(
                        "Agent provider failed its protocol; deterministic fallback "
                        f"selected action={failure['action']}: {failure['summary']}"
                    )
                failure = self.prefer_bounded_diagnostic_over_pause(failure, state)
                action = failure["action"]
                rollback_success = None
                if action == "pause_for_human" and failure["classification"] in {
                    "parameter_invalid",
                    "parameter_oom",
                }:
                    anchor = self.best_accepted_anchor(self.history_summary(session_dir))
                    if anchor and isinstance(anchor.get("params"), dict):
                        rollback_success = self.successful_candidate(
                            session_dir, anchor["params"]
                        )
                        if rollback_success:
                            failure = {**failure, "candidate": anchor["params"]}
                            log(
                                "The failed candidate is parameter-invalid/OOM and "
                                "the Agent has no direct correction; rejecting it and "
                                "continuing selection from the known-good anchor."
                            )
                if action == "pause_for_human" and rollback_success is None:
                    intervention = self.hard_terminal_intervention(failure)
                    if self.hard_terminal_only and intervention is None:
                        raise RuntimeError(
                            "Non-terminal pause escaped autonomous recovery policy"
                        )
                    if intervention is not None:
                        save_json(
                            round_dir
                            / "06_agent_analysis"
                            / "human_intervention_required.json",
                            intervention,
                        )
                    state.update(
                        status="paused_for_human",
                        last_failure_classification=failure["classification"],
                        last_failure_summary=failure["summary"],
                        human_intervention=intervention,
                    )
                    self.save_state(state)
                    log(
                        "Paused because the failure is not safe to repair automatically: "
                        + failure["summary"]
                    )
                    return

                if action == "adjust_parameters" and not failure.get(
                    "recovery_changes", []
                ):
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
                        round_dir / "06_agent_analysis" / "recovery_resolution.json",
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

                    if stop_was_requested_before_submission():
                        state.update(
                            status="stopped_after_failed_round",
                            active_task_id=None,
                        )
                        self.save_state(state)
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
                total_recovery_rounds = int(
                    state.get("total_failure_recovery_rounds", 0)
                ) + 1
                if (
                    not self.hard_terminal_only
                    and total_recovery_rounds > self.max_total_failure_recovery_rounds
                ):
                    state.update(
                        status="paused_after_total_recovery_budget",
                        last_failure_classification=failure["classification"],
                        last_failure_summary=failure["summary"],
                    )
                    self.save_state(state)
                    log("Paused after exhausting the total failure-recovery budget.")
                    return
                state["total_failure_recovery_rounds"] = total_recovery_rounds
                if action == "adjust_parameters":
                    state["failure_retries"] = 0
                    state["failure_adjustments"] += 1
                    if (
                        not self.hard_terminal_only
                        and
                        state["failure_adjustments"]
                        > self.max_parameter_failure_adjustments
                    ):
                        state.update(
                            status="paused_after_parameter_recovery_budget",
                            last_failure_classification=failure["classification"],
                            last_failure_summary=failure["summary"],
                        )
                        self.save_state(state)
                        log("Paused after exhausting the parameter-recovery budget.")
                        return
                    next_candidate = failure["candidate"]
                    next_recovery_values = self.validate_recovery_changes(
                        failure.get("recovery_changes", []),
                        state.get(
                            "runtime_recovery_parameters",
                            self.runtime_recovery_values,
                        ),
                    )
                    state["runtime_recovery_parameters"] = next_recovery_values
                    self.runtime_recovery_values = copy.deepcopy(
                        next_recovery_values
                    )
                    next_label = (
                        f"a{state['candidate_index']}f{state['failure_adjustments']}"
                    )
                    next_status = "recovering_parameter_failure"
                elif action == "diagnostic_retry_same":
                    state["failure_diagnostic_retries"] = int(
                        state.get("failure_diagnostic_retries", 0)
                    ) + 1
                    if (
                        not self.hard_terminal_only
                        and
                        state["failure_diagnostic_retries"]
                        > self.max_agent_diagnostic_retries
                    ):
                        state.update(
                            status="paused_after_diagnostic_recovery_budget",
                            last_failure_classification=failure["classification"],
                            last_failure_summary=failure["summary"],
                        )
                        self.save_state(state)
                        log("Paused after exhausting the Agent diagnostic-retry budget.")
                        return
                    next_candidate = state["current_candidate"]
                    next_label = (
                        f"a{state['candidate_index']}d"
                        f"{state['failure_diagnostic_retries']}"
                    )
                    next_status = "retrying_agent_diagnosed_failure"
                else:
                    state["failure_retries"] += 1
                    if (
                        not self.hard_terminal_only
                        and state["failure_retries"] > self.max_same_candidate_retries
                    ):
                        state.update(
                            status="paused_after_repeated_infrastructure_failure",
                            last_failure_classification=failure["classification"],
                            last_failure_summary=failure["summary"],
                        )
                        self.save_state(state)
                        log("Paused after exhausting identical infrastructure retries.")
                        return
                    next_candidate = state["current_candidate"]
                    next_label = (
                        f"a{state['candidate_index']}r{state['failure_retries']}"
                    )
                    next_status = "retrying_infrastructure_failure"

                if stop_was_requested_before_submission():
                    state.update(
                        status="stopped_after_failed_round",
                        active_task_id=None,
                    )
                    self.save_state(state)
                    return

                _, task_id, run_id = self.prepare_and_submit_round(
                    session_dir,
                    state,
                    index=next_index,
                    label=next_label,
                    candidate=next_candidate,
                    launch_profile=(
                        self.round_launch_profile(round_dir)
                        if action == "retry_same"
                        else None
                    ),
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
                log(
                    f"Failure recovery submitted {next_label} task={task_id} run={run_id}"
                )
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
    group.add_argument(
        "--start", action="store_true", help="start a new continuous session"
    )
    group.add_argument(
        "--resume",
        action="store_true",
        help="resume the active session from state.json",
    )
    group.add_argument(
        "--dry-run", action="store_true", help="validate without submitting"
    )
    group.add_argument(
        "--offline-dry-run",
        action="store_true",
        help="validate local knowledge/config generation without querying or writing a Lease",
    )
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
    group.add_argument(
        "--auto-retry-paused-current",
        action="store_true",
        help="reanalyze a paused round with the Agent and resume unless a hard terminal block is proven",
    )
    group.add_argument(
        "--stop-active-task",
        action="store_true",
        help="stop only the task recorded by state.json using frozen Session config",
    )
    group.add_argument(
        "--replay-unmeasured-candidate",
        metavar="ROUND_DIR",
        help=(
            "replay a prior no-metrics candidate only when its failure was not "
            "attributed to the candidate parameters"
        ),
    )
    group.add_argument(
        "--upgrade-recovery-policy-from",
        metavar="CONFIG",
        help="idle Session only: auditably raise recovery budgets from this config",
    )
    group.add_argument("--status", action="store_true", help="print controller state")
    parser.add_argument(
        "--strategy-profile",
        help="strategy profile for a new Session; resume uses the frozen Session value",
    )
    parser.add_argument(
        "--agent-provider",
        choices=["codex", "anthropic", "openai_compatible", "deepseek", "command"],
        help="Agent provider for a new Session; credentials stay in environment variables",
    )
    parser.add_argument(
        "--config",
        help="configuration file for a new Session; defaults to continuous/config.yaml",
    )
    parser.add_argument(
        "--runtime-root",
        help="isolated mutable state/log/Session root; legacy paths remain the default",
    )
    parser.add_argument(
        "--benchmark-profile",
        help="benchmark profile for a new Session; resume uses the frozen Session value",
    )
    parser.add_argument(
        "--topology-profile",
        help=(
            "topology profile for a new Session; use topology_advisor.py for "
            "the novice-safe recommendation"
        ),
    )
    parser.add_argument(
        "--search-space-profile",
        help="search-space profile for a new Session; resume uses the frozen Session value",
    )
    parser.add_argument(
        "--reuse-baseline-session",
        help=(
            "new Session only: import an identity-matched completed round_000 "
            "and start Agent analysis at A1 without rerunning B0"
        ),
    )
    parser.add_argument(
        "--use-frozen-session",
        action="store_true",
        help="with --check-only, validate the Session config referenced by state.json",
    )
    parser.add_argument(
        "--allow-active-lease",
        action="store_true",
        help="with --check-only, allow a running Lease for an already-active task",
    )
    parser.add_argument(
        "--pause-after-candidate-index",
        type=int,
        help=(
            "pause after archiving metrics for this candidate index; used by the "
            "outer topology Campaign to time-slice one frozen Session"
        ),
    )
    parser.add_argument(
        "--topology-feasibility-only",
        action="store_true",
        help=(
            "new experimental topology only: run candidate index 0 once and stop "
            "without Agent recovery if startup or Benchmark fails"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_runtime_root(Path(args.runtime_root) if args.runtime_root else None)
    if args.status:
        if not STATE_FILE.exists():
            print("No controller state exists.")
            return 0
        print(STATE_FILE.read_text(encoding="utf-8"))
        return 0
    search_space_result = None
    use_frozen_config = bool(
        args.resume
        or args.reanalyze_current
        or args.retry_paused_current
        or args.auto_retry_paused_current
        or args.stop_active_task
        or args.replay_unmeasured_candidate
        or args.upgrade_recovery_policy_from
        or (args.check_only and args.use_frozen_session)
    )
    if (args.use_frozen_session or args.allow_active_lease) and not args.check_only:
        raise RuntimeError("check-only flags require --check-only")
    if args.pause_after_candidate_index is not None and not (args.start or args.resume):
        raise RuntimeError("--pause-after-candidate-index requires --start or --resume")
    if args.topology_feasibility_only and not (args.start or args.resume):
        raise RuntimeError("--topology-feasibility-only requires --start or --resume")
    if use_frozen_config and not STATE_FILE.exists():
        raise RuntimeError("No controller state exists for frozen-Session preflight")
    if use_frozen_config and STATE_FILE.exists():
        if (
            args.strategy_profile
            or args.agent_provider
            or args.benchmark_profile
            or args.topology_profile
            or args.search_space_profile
            or args.reuse_baseline_session
        ):
            raise RuntimeError(
                "Search-space, Strategy, Agent provider, and Benchmark profiles are frozen in "
                "session_config.yaml; "
                "start a new Session to change them"
            )
        frozen_state = load_controller_state()
        frozen_config = Path(frozen_state["session_dir"]) / "session_config.yaml"
        if not frozen_config.is_file():
            raise RuntimeError(
                f"Frozen Session configuration is missing: {frozen_config}"
            )
        config = load_yaml(frozen_config)
    else:
        config_path = Path(args.config).expanduser() if args.config else HERE / "config.yaml"
        raw_config = load_config(config_path)
        external_runtime_adapter = bool(
            isinstance(raw_config.get("runtime"), dict)
            and raw_config["runtime"].get("adapter_file")
        )
        if external_runtime_adapter and (
            args.strategy_profile
            or args.benchmark_profile
            or args.search_space_profile
        ):
            raise RuntimeError(
                "An external runtime adapter owns Strategy, Benchmark and "
                "Search-Space bindings; update and revalidate the adapter instead "
                "of overriding those profiles on the command line"
            )
        raw_config, _ = resolve_runtime_profile(raw_config, KB_ROOT)
        if args.strategy_profile:
            raw_config.setdefault("strategy", {})["profile"] = args.strategy_profile
        if args.topology_profile:
            raw_config.setdefault("topology", {})["profile"] = args.topology_profile
        if args.agent_provider:
            raw_config.setdefault("agent", {})["provider"] = args.agent_provider
        if args.benchmark_profile:
            raw_config.setdefault("benchmark", {})["profile"] = args.benchmark_profile
        if args.search_space_profile:
            raw_config.setdefault("search_space", {})[
                "profile"
            ] = args.search_space_profile
        if args.reuse_baseline_session:
            raw_config["baseline_reuse"] = {
                "source_session": args.reuse_baseline_session
            }
        raw_config = apply_topology_baseline_binding(raw_config)
        raw_config = resolve_initial_baseline_definition(raw_config, KB_ROOT)
        validate_runtime_selections(raw_config)
        config, search_space_result = resolve_search_limits(
            raw_config,
            project_root=KB_ROOT,
            archive_root=ARCHIVE_ROOT,
        )
    controller = Controller(
        config,
        dry_run=bool(args.dry_run or args.offline_dry_run),
        offline_dry_run=args.offline_dry_run,
        search_space_result=search_space_result,
        pause_after_candidate_index=args.pause_after_candidate_index,
        topology_feasibility_only=args.topology_feasibility_only,
    )
    if args.stop_active_task:
        state = load_controller_state()
        print(controller.stop_active_task(state))
        return 0
    validate_agent_credentials(controller.agent_config)
    if args.check_only:
        try:
            controller.check_ready(require_idle_lease=not args.allow_active_lease)
        except Exception as exc:
            print(f"End-to-end preflight failed: {exc}", file=sys.stderr)
            return 2
        transport = "local server" if controller.remote_transport == "local" else "SSH"
        print(f"Controller, {transport}, and persistent Lease preflight: OK")
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
        elif args.upgrade_recovery_policy_from:
            print(
                json.dumps(
                    controller.upgrade_frozen_recovery_policy(
                        Path(args.upgrade_recovery_policy_from).expanduser()
                    ),
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
        elif args.auto_retry_paused_current:
            print(
                json.dumps(
                    controller.auto_retry_paused_current(),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            controller.start(resume=True)
        elif args.replay_unmeasured_candidate:
            print(
                json.dumps(
                    controller.replay_unmeasured_candidate(
                        args.replay_unmeasured_candidate
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            controller.start(resume=True)
        elif args.dry_run or args.offline_dry_run:
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
                        recoverable_controller_error = (
                            controller_exception_is_recoverable(exc)
                        )
                        explicit_manual_block = bool(
                            MANUAL_INTERVENTION_SIGNATURES.search(str(exc))
                        ) and not recoverable_controller_error
                        if recoverable_controller_error or (
                            controller.hard_terminal_only
                            and not explicit_manual_block
                        ):
                            recovery_attempts = int(
                                failed_state.get("controller_recovery_attempts", 0)
                            ) + 1
                            failed_state["controller_recovery_attempts"] = recovery_attempts
                            paused_status = (
                                "recovering_controller_error"
                                if recovery_attempts
                                <= controller.max_controller_recovery_attempts
                                else "paused_after_repeated_controller_error"
                            )
                        else:
                            paused_status = "paused_controller_error"
                        intervention = None
                        if explicit_manual_block:
                            intervention = controller.hard_terminal_intervention(
                                {
                                    "summary": (
                                        "Controller encountered an explicit immutable "
                                        f"dependency: {type(exc).__name__}: {exc}"
                                    ),
                                    "root_cause": str(exc),
                                    "evidence": [f"{type(exc).__name__}: {exc}"],
                                    "classification": "unknown",
                                }
                            )
                        failed_state.update(
                            status=paused_status,
                            controller_error=f"{type(exc).__name__}: {exc}",
                            human_intervention=intervention,
                            updated_at=now(),
                        )
                        save_json(STATE_FILE, failed_state)
                        log(
                            "Controller recorded a bounded restart decision after an "
                            f"internal/control-plane error: {type(exc).__name__}: {exc}"
                        )
                    except (OSError, ValueError, json.JSONDecodeError):
                        pass
                raise
    finally:
        release_controller_lock(lock_descriptor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
