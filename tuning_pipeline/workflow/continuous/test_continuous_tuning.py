from __future__ import annotations

import datetime as dt
import json
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from . import continuous_tuning as tuning
from .search_space_adapter import resolve_search_limits


def config() -> dict:
    value = {
        "remote_host": "example",
        "remote_project": "/allowed/project",
        "poll_seconds": 1,
        "round_timeout_minutes": 10,
        "fixed_scenario": {
            "input_tokens": 32000,
            "output_tokens": 1000,
            "num_prompts": 8,
        },
        "change_policy": {
            "max_parameters_per_round": 3,
            "max_grid_steps_per_parameter": 2,
            "max_total_grid_steps": 4,
        },
        "search_limits": {
            "max_num_seqs": [48, 64],
            "max_model_len": [64000],
            "max_num_batched_tokens": [4096],
            "gpu_memory_utilization": [0.9, 0.93],
            "enable_prefix_caching": [True],
            "async_scheduling": [True],
            "enable_expert_parallel": [True],
            "compilation_mode": ["FULL_DECODE_ONLY"],
            "num_speculative_tokens": [3],
            "long_prefill_token_threshold": [0],
        },
    }
    value["baseline"] = {
        "max_num_seqs": 48,
        "max_model_len": 64000,
        "max_num_batched_tokens": 4096,
        "gpu_memory_utilization": 0.93,
        "enable_prefix_caching": True,
        "async_scheduling": True,
        "enable_expert_parallel": True,
        "compilation_mode": "FULL_DECODE_ONLY",
        "num_speculative_tokens": 3,
        "long_prefill_token_threshold": 0,
    }
    return value


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._log_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._log_directory.cleanup)
        self._log_patch = patch.object(
            tuning,
            "LOG_FILE",
            Path(self._log_directory.name) / "controller.test.log",
        )
        self._log_patch.start()
        self.addCleanup(self._log_patch.stop)
        self.controller = tuning.Controller(config())
        self.baseline = {
            "max_num_seqs": 48,
            "max_model_len": 64000,
            "max_num_batched_tokens": 4096,
            "gpu_memory_utilization": 0.93,
            "enable_prefix_caching": True,
            "async_scheduling": True,
            "enable_expert_parallel": True,
            "compilation_mode": "FULL_DECODE_ONLY",
            "num_speculative_tokens": 3,
            "long_prefill_token_threshold": 0,
        }

    def test_numeric_grid_distance_uses_sorted_value_order(self) -> None:
        self.assertEqual(
            1,
            self.controller.grid_step_distance([48, 8, 16, 32, 64], 48, 64),
        )
        self.assertEqual(
            4,
            self.controller.grid_step_distance([48, 8, 16, 32, 64], 8, 64),
        )

    def test_adaptive_policy_enters_refinement_after_valid_improvement(self) -> None:
        configured = config()
        configured["change_policy"]["adaptive"] = {
            "enabled": True,
            "exploration": {
                "max_parameters_per_round": 4,
                "max_total_grid_steps": 6,
            },
            "refinement": {
                "max_parameters_per_round": 2,
                "max_total_grid_steps": 3,
            },
        }
        controller = tuning.Controller(configured)
        history = [
            {"metrics": {"benchmark_mode": "aligned_l1"}},
            {"metrics": {"benchmark_mode": "aligned_l1"}},
        ]
        with patch.object(
            controller,
            "assess_aligned_l1",
            return_value={"eligible_as_improvement": True},
        ):
            policy = controller.effective_change_policy(history)
        self.assertEqual("refinement", policy["phase"])
        self.assertEqual(2, policy["max_parameters_per_round"])
        self.assertEqual(3, policy["max_total_grid_steps"])

    def test_strategy_profile_can_switch_for_a_new_session(self) -> None:
        configured = config()
        configured["strategy"] = {
            "profile": "best_anchor_coverage_v3",
            "profiles_file": "workflow/continuous/strategy_profiles.yaml",
        }
        controller = tuning.Controller(configured)
        self.assertEqual("best_anchor_coverage_v3", controller.strategy_profile_name)
        self.assertEqual(
            [2, 3],
            controller.change_policy["adaptive"]["exploration"][
                "preferred_parameters_per_round"
            ],
        )
        self.assertEqual(
            2,
            controller.effective_change_policy()["minimum_parameters_per_round"],
        )
        with self.assertRaisesRegex(ValueError, "between 2 and 3"):
            controller.validate_candidate(
                self.baseline,
                dict(self.baseline, max_num_seqs=64),
                [{
                    "parameter": "max_num_seqs",
                    "before": 48,
                    "after": 64,
                    "rationale": "A single change is invalid under V3 exploration.",
                }],
            )
        configured["strategy"]["profile"] = "missing"
        with self.assertRaisesRegex(ValueError, "Unknown strategy profile"):
            tuning.Controller(configured)

    def test_agent_provider_selects_its_named_settings_profile(self) -> None:
        configured = config()
        configured["agent"] = {
            "provider": "anthropic",
            "providers": {
                "codex": {"command": "auto"},
                "anthropic": {
                    "model": "test-claude",
                    "api_key_env": "TEST_ANTHROPIC_KEY",
                },
            },
        }
        controller = tuning.Controller(configured)
        self.assertEqual("anthropic", controller.agent_config["provider"])
        self.assertEqual(
            "TEST_ANTHROPIC_KEY",
            controller.agent_config["settings"]["api_key_env"],
        )

    def test_candidate_rejection_is_reselected_and_audited(self) -> None:
        configured = config()
        configured["change_policy"]["max_candidate_reselections"] = 2
        controller = tuning.Controller(configured)
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "session"
            round_dir = session / "round_000_a0"
            schema_dir = session / "00_search_space"
            analysis_dir = round_dir / "06_agent_analysis"
            schema_dir.mkdir(parents=True)
            analysis_dir.mkdir(parents=True)
            (schema_dir / "agent_decision.schema.json").write_text(
                "{}", encoding="utf-8"
            )
            decisions = [
                {"candidate": {"id": "bad"}, "changes": []},
                {"candidate": {"id": "good"}, "changes": []},
            ]
            prompts: list[str] = []

            def fake_run(agent: dict, **kwargs: object) -> SimpleNamespace:
                prompts.append(str(kwargs["prompt"]))
                output = Path(kwargs["output_path"])
                output.write_text(
                    json.dumps(decisions[len(prompts) - 1]), encoding="utf-8"
                )
                return SimpleNamespace(
                    provider="codex", returncode=0, stdout="events", stderr=""
                )

            def validate(decision: dict) -> None:
                if decision["candidate"]["id"] == "bad":
                    raise ValueError("synthetic rejection")

            with patch.object(tuning, "run_structured_agent", side_effect=fake_run):
                accepted = controller.run_agent_decision_with_reselection(
                    session, round_dir, "base prompt", validate
                )

            self.assertEqual("good", accepted["candidate"]["id"])
            self.assertIn("synthetic rejection", prompts[1])
            audit = json.loads(
                (analysis_dir / "candidate_reselection_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("accepted", audit["final_status"])

    def test_codex_command_is_portable_and_can_be_overridden(self) -> None:
        with patch.dict(
            tuning.os.environ,
            {"VLLMTKB_CODEX_COMMAND": "portable-codex"},
            clear=False,
        ):
            with patch.object(tuning.shutil, "which", return_value="C:/tools/codex.cmd"):
                self.assertEqual(
                    tuning.resolve_codex_command("auto"),
                    "C:/tools/codex.cmd",
                )

    def test_missing_codex_command_fails_with_actionable_error(self) -> None:
        with patch.dict(tuning.os.environ, {}, clear=True):
            with patch.object(tuning.shutil, "which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "VLLMTKB_CODEX_COMMAND"):
                    tuning.resolve_codex_command("auto")

    def test_candidate_requires_a_bounded_whitelisted_change_set(self) -> None:
        candidate = dict(self.baseline, max_num_seqs=64)
        self.controller.validate_candidate(
            self.baseline,
            candidate,
            [
                {
                    "parameter": "--max-num-seqs",
                    "before": 48,
                    "after": 64,
                    "rationale": "Adjacent scheduler-capacity experiment.",
                }
            ],
        )
        with self.assertRaises(ValueError):
            self.controller.validate_candidate(self.baseline, self.baseline, [])
        with self.assertRaises(ValueError):
            self.controller.validate_candidate(
                self.baseline,
                dict(self.baseline, max_num_seqs=96),
                [
                    {
                        "parameter": "--max-num-seqs",
                        "before": 48,
                        "after": 96,
                        "rationale": "Value deliberately falls outside the whitelist.",
                    }
                ],
            )

    def test_coupled_changes_require_auditable_evidence(self) -> None:
        candidate = dict(
            self.baseline,
            max_num_seqs=64,
            gpu_memory_utilization=0.9,
        )
        changes = [
            {
                "parameter": "max_num_seqs",
                "before": 48,
                "after": 64,
                "rationale": "Increase scheduler concurrency within the adjacent grid.",
            },
            {
                "parameter": "gpu_memory_utilization",
                "before": 0.93,
                "after": 0.9,
                "rationale": "Reserve memory headroom required by the concurrency increase.",
            },
        ]
        decision = {
            "change_strategy": "coupled_parameters",
            "knowledge_evidence": [
                "Runtime scheduling evidence supports testing the next sequence-count grid.",
                "Prior OOM evidence requires memory headroom when capacity is increased.",
            ],
            "interaction_analysis": [
                "Higher sequence concurrency and memory headroom jointly control startup safety."
            ],
            "constraint_checks": [
                "max_num_batched_tokens remains above max_num_seqs times speculative factor.",
                "gpu_memory_utilization remains inside the configured and legal interval.",
            ],
        }
        self.controller.validate_candidate(
            self.baseline, candidate, changes, decision
        )
        with self.assertRaises(ValueError):
            self.controller.validate_candidate(
                self.baseline,
                candidate,
                changes,
                {**decision, "interaction_analysis": []},
            )
        with self.assertRaises(ValueError):
            self.controller.validate_candidate(
                self.baseline,
                dict(self.baseline, max_num_seqs=64),
                [
                    {
                        "parameter": "max_num_seqs",
                        "before": 48,
                        "after": 48,
                        "rationale": "Metadata intentionally contradicts the candidate.",
                    }
                ],
            )

    def test_state_image_identity_must_match_verified_manifest(self) -> None:
        with self.assertRaises(RuntimeError):
            self.controller.assert_state_image_identity({})
        self.controller.assert_state_image_identity(
            {"image_identity": self.controller.image_identity}
        )

    def test_repository_templates_match_verified_image(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        controller = tuning.Controller(configured)
        controller.validate_deployment_configuration()
        self.assertEqual(controller.benchmark_mode, "aligned_l1")
        env_text = controller.candidate_env("b0", configured["baseline"])
        self.assertIn("LAUNCH_PROFILE=official_source_defaults", env_text)
        self.assertIn("BENCHMARK_MODE=aligned_l1", env_text)
        self.assertIn("BENCHMARK_REPETITIONS=3", env_text)
        self.assertIn(
            "BENCHMARK_SUITE='01_调优_结构化定长-v4.yaml'",
            env_text,
        )
        self.assertIn('"schema_files_sha256"', env_text)
        self.assertIn("SAFETENSORS_LOAD_STRATEGY=prefetch", env_text)
        self.assertIn("SAFETENSORS_PREFETCH_NUM_THREADS=8", env_text)
        self.assertIn("SAFETENSORS_PREFETCH_BLOCK_SIZE=16777216", env_text)

        runtime = (tuning.HERE / "remote" / "common_runtime_loop.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('--safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}"', runtime)
        self.assertIn(
            '--safetensors-prefetch-num-threads "${SAFETENSORS_PREFETCH_NUM_THREADS}"',
            runtime,
        )
        self.assertIn(
            'if [[ "${LAUNCH_PROFILE}" == "explicit_candidate" ]]', runtime
        )

    def test_model_loading_contract_rejects_unsafe_prefetch_settings(self) -> None:
        configured = config()
        configured["model_loading"] = {
            "safetensors_load_strategy": "unknown",
            "safetensors_prefetch_num_threads": 8,
            "safetensors_prefetch_block_size": 16 * 1024 * 1024,
        }
        with self.assertRaisesRegex(ValueError, "safetensors_load_strategy"):
            tuning.Controller(configured)

        configured["model_loading"]["safetensors_load_strategy"] = "prefetch"
        configured["model_loading"]["safetensors_prefetch_num_threads"] = 64
        with self.assertRaisesRegex(ValueError, "prefetch_num_threads"):
            tuning.Controller(configured)

    def test_b0_reconciles_source_resolved_values_before_agent_handoff(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        controller = tuning.Controller(configured)
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "session"
            round_dir = session / "round_000_b0"
            (round_dir / "02_parameters").mkdir(parents=True)
            (round_dir / "04_runtime").mkdir(parents=True)
            (round_dir / "04_runtime" / "master.log").write_text(
                "Chunked prefill is enabled with max_num_batched_tokens=2048.\n"
                "Initializing a V1 LLM engine with config: max_seq_len=131072, "
                "enable_prefix_caching=True, enable_chunked_prefill=True, "
                "compilation_config={'cudagraph_mode': "
                "<CUDAGraphMode.FULL_AND_PIECEWISE: (1, 1)>, "
                "'cudagraph_capture_sizes': [16, 32, 64, 128, 256], "
                "'max_cudagraph_capture_size': 256}\n",
                encoding="utf-8",
            )
            state = {
                "round_label": "b0",
                "current_candidate": dict(configured["baseline"]),
            }
            controller.reconcile_official_source_default_baseline(
                session, round_dir, state
            )
            self.assertTrue(state["official_source_defaults_reconciled"])
            self.assertEqual(131072, state["current_candidate"]["max_model_len"])
            self.assertEqual(
                [16, 32, 64, 128, 256],
                state["current_candidate"]["cudagraph_capture_sizes"],
            )
            self.assertTrue(
                (round_dir / "02_parameters" / "b0_effective_resolution.yaml").is_file()
            )

    def test_default_automated_search_space_resolves_and_manual_fallback_remains(self) -> None:
        project_root = tuning.KB_ROOT
        raw = tuning.load_yaml(tuning.HERE / "config.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            resolved, result = resolve_search_limits(
                raw,
                project_root=project_root,
                archive_root=Path(temporary),
            )
            self.assertIsNotNone(result)
            self.assertEqual(
                "automated", resolved["resolved_search_space"]["mode"]
            )
            self.assertEqual(12, len(result["active_search_limits"]))
            self.assertNotIn("enable_eplb", result["active_search_limits"])
            self.assertEqual([False], resolved["search_limits"]["enable_eplb"])
            self.assertEqual(
                [0], resolved["search_limits"]["eplb_num_redundant_experts"]
            )
            self.assertNotIn(
                "decode_context_parallel_size", result["active_search_limits"]
            )
            # Automated axes are merged with singleton runtime-contract axes
            # from the manual fallback.  The exact count can grow when a new
            # approved current-version parameter becomes active.
            expected_names = set(raw["search_limits"]) | set(
                result["active_search_limits"]
            )
            self.assertEqual(expected_names, set(resolved["search_limits"]))
            self.assertEqual(
                raw["search_limits"], resolved["manual_search_limits"]
            )
            self.assertEqual(
                [64000], resolved["search_limits"]["max_model_len"]
            )
            self.assertEqual(
                [raw["baseline"]["async_scheduling"]],
                resolved["search_limits"]["async_scheduling"],
            )
            self.assertEqual(
                [raw["baseline"]["enable_expert_parallel"]],
                resolved["search_limits"]["enable_expert_parallel"],
            )
            self.assertEqual(
                [
                    [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192],
                    None,
                ],
                resolved["search_limits"]["cudagraph_capture_sizes"],
            )
            if "cudagraph_capture_sizes" not in result["active_search_limits"]:
                self.assertIn(
                    "cudagraph_capture_sizes",
                    resolved["resolved_search_space"]["derived_runtime_parameters"],
                )
            controller = tuning.Controller(resolved)
            mismatched_graph = dict(resolved["baseline"])
            mismatched_graph["max_cudagraph_capture_size"] = 256
            mismatched_graph["cudagraph_capture_sizes"] = [
                16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192
            ]
            with self.assertRaisesRegex(
                ValueError, "must equal the largest explicit"
            ):
                controller.validate_runtime_configuration(mismatched_graph)
            mismatched_graph["cudagraph_capture_sizes"] = None
            controller.validate_runtime_configuration(mismatched_graph)

            manual_raw = dict(raw)
            manual_raw["search_limits_mode"] = "manual"
            manual, manual_result = resolve_search_limits(
                manual_raw,
                project_root=project_root,
                archive_root=Path(temporary),
            )
            self.assertIsNone(manual_result)
            self.assertEqual(17, len(manual["search_limits"]))

    def test_reserved_decode_context_parameter_keeps_an_injection_contract(self) -> None:
        self.assertEqual(
            "DECODE_CONTEXT_PARALLEL_SIZE",
            tuning.ALL_PARAM_TO_ENV["decode_context_parallel_size"],
        )
        runtime = (tuning.HERE / "remote" / "common_runtime_loop.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--decode-context-parallel-size", runtime)

    def test_measurement_policy_rejects_small_or_guardrail_violating_gain(self) -> None:
        configured = config()
        configured["measurement_policy"] = {
            "minimum_successful_requests": 8,
            "require_zero_failed_requests": True,
            "minimum_throughput_gain_percent": 3.0,
            "maximum_ttft_regression_percent": 10.0,
            "maximum_tpot_regression_percent": 10.0,
        }
        controller = tuning.Controller(configured)
        history = [
            {"metrics": {"metrics": {"successful_requests": 8, "failed_requests": 0}}},
            {"metrics": {"metrics": {"successful_requests": 8, "failed_requests": 0}}},
        ]
        accepted = controller.assess_measurement(
            history,
            {
                "numeric_metric_deltas_vs_baseline": {
                    "output_token_throughput": {"percent": 5.0},
                    "mean_ttft": {"percent": 4.0},
                    "mean_tpot": {"percent": 2.0},
                }
            },
        )
        self.assertTrue(accepted["eligible_as_improvement"])
        rejected = controller.assess_measurement(
            history,
            {
                "numeric_metric_deltas_vs_baseline": {
                    "output_token_throughput": {"percent": 2.0},
                    "mean_ttft": {"percent": 1.0},
                    "mean_tpot": {"percent": 1.0},
                }
            },
        )
        self.assertFalse(rejected["eligible_as_improvement"])

    def test_aligned_l1_accepts_noise_adjusted_guardrailed_gain(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        controller = tuning.Controller(configured)

        def payload(score: float, cv: float, ratio: float = 1.0) -> dict:
            cases = []
            for workload in configured["benchmark"]["aligned_l1"]["workloads"]:
                for concurrency in (1, 16, 32):
                    cases.append(
                        {
                            "workload": workload,
                            "concurrency": concurrency,
                            "aggregate_output_tps": 100.0 * ratio,
                            "ttft_p50_ms": 100.0,
                            "ttft_p90_ms": 120.0,
                            "tpot_p50_ms": 10.0,
                            "tpot_p90_ms": 12.0,
                        }
                    )
            return {
                "benchmark_mode": "aligned_l1",
                "metrics": {
                    "successful_requests": 1176,
                    "failed_requests": 0,
                    "output_token_throughput": score,
                },
                "l1": {
                    "all_repetitions_gate_passed": True,
                    "repetition_count": configured["benchmark"]["aligned_l1"]["repetitions"],
                    "primary_concurrency": 32,
                    "primary_aggregate_output_tps_geomean": score,
                    "primary_score_cv_percent": cv,
                    "cases": cases,
                },
            }

        accepted = controller.assess_measurement(
            [
                {"metrics": payload(100.0, 0.5)},
                {"metrics": payload(104.0, 0.5, 1.04)},
            ],
            {},
        )
        self.assertTrue(accepted["eligible_as_improvement"])
        self.assertEqual(accepted["noise_adjusted_required_gain_percent"], 3.0)

        noisy = controller.assess_measurement(
            [
                {"metrics": payload(100.0, 3.0)},
                {"metrics": payload(104.0, 2.0, 1.04)},
            ],
            {},
        )
        self.assertFalse(noisy["eligible_as_improvement"])
        self.assertEqual(noisy["noise_adjusted_required_gain_percent"], 6.0)

    def test_aligned_l1_rejects_one_workload_regression(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        controller = tuning.Controller(configured)
        baseline_cases = []
        candidate_cases = []
        workloads = list(configured["benchmark"]["aligned_l1"]["workloads"])
        for workload in workloads:
            for concurrency in (1, 16, 32):
                base = {
                    "workload": workload,
                    "concurrency": concurrency,
                    "aggregate_output_tps": 100.0,
                    "ttft_p50_ms": 100.0,
                    "ttft_p90_ms": 120.0,
                    "tpot_p50_ms": 10.0,
                    "tpot_p90_ms": 12.0,
                }
                baseline_cases.append(base)
                candidate = dict(base)
                candidate["aggregate_output_tps"] = (
                    90.0
                    if workload == workloads[0] and concurrency == 32
                    else 110.0
                )
                candidate_cases.append(candidate)
        base_l1 = {
            "all_repetitions_gate_passed": True,
            "repetition_count": configured["benchmark"]["aligned_l1"]["repetitions"],
            "primary_concurrency": 32,
            "primary_aggregate_output_tps_geomean": 100.0,
            "primary_score_cv_percent": 0.5,
            "cases": baseline_cases,
        }
        candidate_l1 = {
            **base_l1,
            "primary_aggregate_output_tps_geomean": 105.0,
            "cases": candidate_cases,
        }
        assessment = controller.assess_measurement(
            [
                {"metrics": {"benchmark_mode": "aligned_l1", "l1": base_l1}},
                {"metrics": {"benchmark_mode": "aligned_l1", "l1": candidate_l1}},
            ],
            {},
        )
        self.assertFalse(assessment["eligible_as_improvement"])
        self.assertTrue(
            any("aggregate output TPS ratio" in item for item in assessment["violations"])
        )

    def test_ssh_uses_explicit_remote_status_marker(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="payload\n__CONTINUOUS_REMOTE_RC__=0\n",
            stderr="Server failed to confirm ownership of private host keys\n",
        )
        with patch.object(tuning, "run_process", return_value=completed):
            self.assertEqual(self.controller.ssh("printf payload"), "payload")

        failed = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="bad\n__CONTINUOUS_REMOTE_RC__=2\n",
            stderr="remote error\n",
        )
        with patch.object(tuning, "run_process", return_value=failed):
            with self.assertRaises(RuntimeError):
                self.controller.ssh("false")

    def test_aligned_runner_is_started_in_a_detached_session(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        controller = tuning.Controller(configured)
        with patch.object(controller, "ssh", return_value="") as remote:
            controller.start_aligned_benchmark("a0_example", "example-lease")
        command = remote.call_args.args[0]
        self.assertIn("setsid -f bash -c", command)
        self.assertNotIn("nohup bash", command)
        self.assertIn("benchmark_runner.pid", command)
        self.assertIn("BENCHMARK_START_LOCK", command)

    def test_remote_watchdog_is_started_in_a_detached_session(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        controller = tuning.Controller(configured)
        with patch.object(controller, "ssh", return_value="") as remote:
            controller.launch_benchmark_watchdog("a0_example", "example-lease")
        command = remote.call_args.args[0]
        self.assertIn("setsid -f bash -c", command)
        self.assertIn("benchmark_watchdog.sh", command)
        self.assertIn("benchmark_watchdog.pid", command)

    def test_aligned_benchmark_requires_live_remote_processes(self) -> None:
        self.controller.benchmark_mode = "aligned_l1"
        ready = {"SERVICE_READY": True, "BENCHMARK_STARTED": False}
        self.assertTrue(
            self.controller.should_start_aligned_benchmark(
                ready, {"terminal": False, "active_pods": 2}
            )
        )
        self.assertFalse(
            self.controller.should_start_aligned_benchmark(
                ready, {"terminal": True, "active_pods": 0}
            )
        )
        self.assertFalse(
            self.controller.should_start_aligned_benchmark(
                ready, {"terminal": False, "active_pods": None}
            )
        )

    def test_idle_lease_is_not_misread_from_running_note(self) -> None:
        output = """\
LEASE  example
RESOURCE  active-nodes=2/2  npu=32

SLOT  service
  status   idle=2
  nodes    master=idle worker=idle

NOTE  running means a managed process is alive.
"""
        with patch.object(self.controller, "ssh", return_value=output):
            snapshot = self.controller.task_snapshot("example-lease")
        self.assertFalse(snapshot["active_pods"])
        self.assertTrue(snapshot["terminal"])

    def test_active_idle_lease_is_available(self) -> None:
        output = """\
LEASE  example
RESOURCE  status=active  nodes=2/2 Ready  npu=32/32

        SLOT  service
  status   idle=2
"""
        self.controller.lab = {"lease_name": "example"}
        with patch.object(self.controller, "ssh", return_value=output):
            self.assertEqual(output, self.controller.ensure_lab_available())

    def test_fresh_active_lease_without_slots_is_available(self) -> None:
        output = """\
LEASE  example
RESOURCE  status=active  nodes=2/2 Ready  npu=32/32
NODES
  rank-000  master[0]  Ready  npu=16
  rank-001  worker[0]  Ready  npu=16
SLOTS  none
"""
        self.controller.lab = {"lease_name": "example"}
        with patch.object(self.controller, "ssh", return_value=output):
            self.assertEqual(output, self.controller.ensure_lab_available())

    def test_lab_dry_run_reuses_idle_persistent_lease(self) -> None:
        self.controller.lab = {"lease_name": "example"}
        with tempfile.TemporaryDirectory() as temporary:
            round_dir = Path(temporary)
            (round_dir / "03_submission").mkdir()
            (round_dir / "02_parameters").mkdir()
            with (
                patch.object(
                    self.controller,
                    "validate_runtime_configuration",
                ),
                patch.object(
                    self.controller,
                    "ensure_lab_available",
                    return_value="idle lease",
                ) as available,
                patch.object(
                    self.controller,
                    "prepare_lab",
                    side_effect=AssertionError("must not recreate the lease"),
                ),
            ):
                task_id, run_id = self.controller.submit_lab(
                    round_dir,
                    "a0",
                    self.baseline,
                    dry_run=True,
                )
            available.assert_called_once_with()
            self.assertIsNone(task_id)
            self.assertTrue(run_id.startswith("a0_"))
            rendered = (round_dir / "02_parameters" / "candidate.env").read_text(
                encoding="utf-8"
            )
            self.assertIn("MAX_NUM_SEQS=48", rendered)

    def test_partial_lease_exit_is_detected_after_grace_period(self) -> None:
        output = """\
LEASE  example
RESOURCE  active-nodes=2/2  npu=32

SLOT  service
  status   idle=1, running=1
  nodes    master=running worker=idle
"""
        with patch.object(self.controller, "ssh", return_value=output):
            snapshot = self.controller.task_snapshot("example-lease")
        self.assertEqual(snapshot["status"], "LeaseProcessesPartialFailure")
        self.assertEqual(snapshot["active_pods"], 1)
        self.assertTrue(snapshot["partial_failure"])
        self.assertFalse(snapshot["terminal"])

        recent = dt.datetime.now().astimezone()
        self.assertFalse(
            self.controller.partial_exit_is_failure(
                {"round_submitted_at": recent.isoformat(timespec="seconds")},
                snapshot,
            )
        )
        old = recent - dt.timedelta(seconds=121)
        self.assertTrue(
            self.controller.partial_exit_is_failure(
                {"round_submitted_at": old.isoformat(timespec="seconds")},
                snapshot,
            )
        )

    def test_attempted_history_includes_successes_and_parameter_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir)
            success = session / "round_000_a0"
            failed = session / "round_001_a1"
            tuning.save_yaml(
                success / "02_parameters" / "candidate_params.yaml",
                self.baseline,
            )
            tuning.save_json(
                success / "05_results" / "metrics.json",
                {"metrics": {"successful_requests": 8, "failed_requests": 0}},
            )
            failed_candidate = dict(self.baseline, max_num_seqs=64)
            tuning.save_yaml(
                failed / "02_parameters" / "candidate_params.yaml",
                failed_candidate,
            )
            tuning.save_yaml(
                failed / "05_results" / "failure.yaml",
                {"reason": "invalid parameter"},
            )
            tuning.save_json(
                failed / "06_agent_analysis" / "failure_decision.json",
                {
                    "classification": "parameter_invalid",
                    "candidate": self.baseline,
                },
            )

            attempts = self.controller.attempted_history_summary(session)
            self.assertEqual([item["outcome"] for item in attempts], ["success", "failed"])
            self.assertTrue(
                self.controller.candidate_was_attempted(session, failed_candidate)
            )

    def test_exploration_memory_uses_accepted_anchor_and_preserves_causality(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        controller = tuning.Controller(configured)

        def payload(score: float) -> dict:
            cases = []
            for workload in configured["benchmark"]["aligned_l1"]["workloads"]:
                for concurrency in (1, 16, 32):
                    cases.append(
                        {
                            "workload": workload,
                            "concurrency": concurrency,
                            "aggregate_output_tps": score,
                            "ttft_p50_ms": 100.0,
                            "ttft_p90_ms": 120.0,
                            "tpot_p50_ms": 10.0,
                            "tpot_p90_ms": 12.0,
                        }
                    )
            return {
                "benchmark_mode": "aligned_l1",
                "l1": {
                    "all_repetitions_gate_passed": True,
                    "repetition_count": configured["benchmark"]["aligned_l1"]["repetitions"],
                    "primary_concurrency": 32,
                    "primary_aggregate_output_tps_geomean": score,
                    "primary_score_cv_percent": 0.0,
                    "cases": cases,
                },
            }

        baseline = dict(configured["baseline"])
        accepted_params = dict(baseline, max_num_batched_tokens=8192)
        rejected_combo = dict(
            baseline,
            max_num_batched_tokens=16384,
            max_num_seqs=64,
        )
        history = [
            {"round": "round_A0", "params": baseline, "metrics": payload(100.0)},
            {
                "round": "round_A1",
                "params": accepted_params,
                "metrics": payload(104.0),
            },
            {
                "round": "round_A2",
                "params": rejected_combo,
                "metrics": payload(90.0),
            },
        ]
        memory = controller.exploration_memory(history, history)

        self.assertEqual(memory["best_accepted_anchor"]["round"], "round_A1")
        self.assertEqual(len(memory["direct_single_parameter_effects"]), 1)
        self.assertEqual(
            set(
                memory["direct_single_parameter_effects"][0][
                    "independent_changes_vs_baseline"
                ]
            ),
            {"max_num_batched_tokens"},
        )
        self.assertEqual(len(memory["measured_multi_parameter_combinations"]), 1)
        self.assertIn(
            32768,
            memory["coverage_by_parameter"]["max_num_batched_tokens"][
                "untested_values"
            ],
        )

    def test_best_anchor_coverage_strategy_is_exposed_to_agent(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        controller = tuning.Controller(configured)
        policy = controller.effective_change_policy([])
        self.assertEqual(policy["strategy_version"], "best_anchor_coverage_v2")
        self.assertEqual(policy["preferred_parameters_per_round"], [2, 3])
        self.assertEqual(policy["max_parameters_per_round"], 3)

    def test_failure_recovery_allows_known_good_rollback_but_not_failed_candidate(
        self,
    ) -> None:
        configured = config()
        configured["search_limits"]["num_speculative_tokens"] = [1, 2, 3]
        controller = tuning.Controller(configured)
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir)
            success = session / "round_000_a0"
            tuning.save_yaml(
                success / "02_parameters" / "candidate_params.yaml",
                self.baseline,
            )
            tuning.save_json(
                success / "05_results" / "metrics.json",
                {"metrics": {"successful_requests": 8, "failed_requests": 0}},
            )
            invalid = dict(self.baseline, num_speculative_tokens=2)
            failed = session / "round_001_a1"
            tuning.save_yaml(
                failed / "02_parameters" / "candidate_params.yaml",
                invalid,
            )
            tuning.save_yaml(
                failed / "05_results" / "failure.yaml",
                {"reason": "invalid parameter"},
            )

            rollback = {
                "action": "adjust_parameters",
                "classification": "parameter_invalid",
                "safe_to_automate": True,
                "change_strategy": "single_parameter",
                "evidence": [
                    "The previous successful round proves speculative token count three is valid."
                ],
                "interaction_analysis": [],
                "constraint_checks": [
                    "A speculative factor of four divides the fixed tensor parallel size sixteen."
                ],
                "candidate": self.baseline,
                "changes": [
                    {
                        "parameter": "num_speculative_tokens",
                        "before": 2,
                        "after": 3,
                        "rationale": "known-good rollback",
                    }
                ],
            }
            known_success = controller.validate_failure_decision(
                session,
                rollback,
                invalid,
            )
            self.assertEqual(known_success["round"], "round_000_a0")

            failed_retry = {
                "action": "adjust_parameters",
                "classification": "parameter_invalid",
                "safe_to_automate": True,
                "change_strategy": "single_parameter",
                "evidence": [
                    "The failed history already identifies speculative token count two as invalid."
                ],
                "interaction_analysis": [],
                "constraint_checks": [
                    "The proposed speculative factor three does not divide tensor parallel size sixteen."
                ],
                "candidate": invalid,
                "changes": [
                    {
                        "parameter": "num_speculative_tokens",
                        "before": 1,
                        "after": 2,
                        "rationale": "repeat invalid candidate",
                    }
                ],
            }
            with self.assertRaisesRegex(ValueError, "previously failed"):
                controller.validate_failure_decision(
                    session,
                    failed_retry,
                    dict(self.baseline, num_speculative_tokens=1),
                )

    def test_structured_one_request_shortfall_gets_bounded_retry(self) -> None:
        self.controller.benchmark_mode = "aligned_l1"
        with tempfile.TemporaryDirectory() as temporary:
            round_dir = Path(temporary)
            runtime_dir = round_dir / "04_runtime"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "SERVICE_READY").touch()
            (runtime_dir / "benchmark_case_retry_plan.json").write_text(
                json.dumps(
                    {
                        "retryable": True,
                        "successful": 31,
                        "planned": 32,
                        "config": "decode/c16/guidellm.yaml",
                    }
                ),
                encoding="utf-8",
            )

            decision = self.controller.deterministic_benchmark_retry(
                round_dir, self.baseline
            )

            self.assertIsNotNone(decision)
            self.assertEqual(decision["action"], "retry_same")
            self.assertEqual(decision["candidate"], self.baseline)

    def test_log_fallback_scales_case_count_with_repetitions(self) -> None:
        self.controller.benchmark_mode = "aligned_l1"
        self.controller.benchmark = {"aligned_l1": {"repetitions": 2}}
        with tempfile.TemporaryDirectory() as temporary:
            round_dir = Path(temporary)
            runtime_dir = round_dir / "04_runtime"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "SERVICE_READY").touch()
            cases = "\n".join(
                f"CASE COMPLETED case-{index}/guidellm.yaml runner_exit=0"
                for index in range(48)
            )
            gate = (
                "zero-error request gate failed: {'successful': 31, "
                "'incomplete': 0, 'errored': 0, 'total': 31}, "
                "planned=32, minimum_successful=32"
            )
            (runtime_dir / "benchmark_runner.log").write_text(
                cases + "\n" + gate, encoding="utf-8"
            )

            decision = self.controller.deterministic_benchmark_retry(
                round_dir, self.baseline
            )

            self.assertIsNotNone(decision)

    def test_task_terminal_and_round_timeout_detection(self) -> None:
        self.controller.ssh = lambda *_args, **_kwargs: (
            "Status: Failed\nActive Pods: 0"
        )
        snapshot = self.controller.task_snapshot("123")
        self.assertTrue(snapshot["terminal"])
        self.assertEqual(snapshot["status"], "Failed")

        self.controller.ssh = lambda *_args, **_kwargs: (
            "Status: Pending\nActive Pods: 0"
        )
        pending = self.controller.task_snapshot("123")
        self.assertFalse(pending["terminal"])

        old = dt.datetime.now().astimezone() - dt.timedelta(minutes=11)
        self.assertTrue(
            self.controller.round_timed_out(
                {"round_submitted_at": old.isoformat(timespec="seconds")}
            )
        )

    def test_controller_lock_rejects_second_instance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "controller.lock"
            with patch.object(tuning, "LOCK_FILE", lock_path):
                descriptor = tuning.acquire_controller_lock()
                try:
                    with self.assertRaises(RuntimeError):
                        tuning.acquire_controller_lock()
                finally:
                    tuning.release_controller_lock(descriptor)
                self.assertFalse(lock_path.exists())

    def test_sliced_mtp_path_is_required_and_exported(self) -> None:
        with self.assertRaises(RuntimeError):
            self.controller.validate_runtime_configuration(self.baseline)

        configured = config()
        configured["mtp_draft_model"] = "/models/share/GLM-5.2-mtp-sliced"
        controller = tuning.Controller(configured)
        controller.validate_runtime_configuration(self.baseline)
        env_text = controller.candidate_env("a1", self.baseline)
        self.assertIn(
            "MTP_DRAFT_MODEL_PATH=/models/share/GLM-5.2-mtp-sliced",
            env_text,
        )

    def test_lab_task_snapshot_never_treats_persistent_lease_as_terminal(self) -> None:
        configured = config()
        configured["execution_mode"] = "ktp_lab"
        configured["lab"] = {"lease_name": "cjx-lab"}
        controller = tuning.Controller(configured)
        controller.ssh = lambda *_args, **_kwargs: "master running\nworker running"
        snapshot = controller.task_snapshot("cjx-lab")
        self.assertEqual(snapshot["status"], "LeaseActive")
        self.assertFalse(snapshot["terminal"])
        controller.ssh = lambda *_args, **_kwargs: "master failed\nworker stopped"
        failed = controller.task_snapshot("cjx-lab")
        self.assertEqual(failed["status"], "LeaseProcessesTerminal")
        self.assertTrue(failed["terminal"])


if __name__ == "__main__":
    unittest.main()
