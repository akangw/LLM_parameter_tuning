from __future__ import annotations

import datetime as dt
import base64
import json
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from . import continuous_tuning as tuning
from .search_space_adapter import (
    _latest_history,
    _latest_previous_selection,
    _selected_benchmark_identity,
    resolve_search_limits,
    write_session_search_space,
)
from .model_loading_profile import resolve_model_loading_profile


def config() -> dict:
    production_benchmark = tuning.load_yaml(tuning.HERE / "config.yaml")["benchmark"]
    value = {
        "remote_host": "example",
        "remote_project": "/allowed/project",
        "poll_seconds": 1,
        "round_timeout_minutes": 10,
        "benchmark": production_benchmark,
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
        # Most Controller unit tests exercise the legacy generic policy in
        # isolation. Production config.yaml explicitly selects the unified
        # hierarchical_throughput_v1 default.
        "strategy": {
            "profile": "best_anchor_coverage_v2",
            "profiles_file": "workflow/continuous/strategy_profiles.yaml",
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
    def test_config_overlay_preserves_base_and_replaces_nested_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base.yaml").write_text(
                "remote_transport: paramiko\nagent:\n  provider: codex\n  keep: true\n",
                encoding="utf-8",
            )
            (root / "server.yaml").write_text(
                "base_config: base.yaml\nremote_transport: local\n"
                "agent:\n  provider: deepseek\n",
                encoding="utf-8",
            )
            config = tuning.load_config(root / "server.yaml")
            self.assertEqual("local", config["remote_transport"])
            self.assertEqual("deepseek", config["agent"]["provider"])
            self.assertTrue(config["agent"]["keep"])

    def test_explicit_a0_definition_is_the_only_candidate_parameter_source(self) -> None:
        scenario = tuning.load_yaml(
            tuning.KB_ROOT
            / "workflow"
            / "search_space_compiler"
            / "scenario.glm52-a3-aligned-l1.yaml"
        )
        self.assertNotIn("baseline", scenario)
        self.assertEqual(
            "../baselines/a0_glm52_w8a8_existing_tuned.yaml",
            scenario["baseline_definition"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            definition = root / "a0.yaml"
            definition.write_text(
                "baseline_id: a0-test\n"
                "launch_profile: explicit_candidate\n"
                "reference_parameters:\n"
                "  max_num_seqs: 48\n"
                "  compilation_enable_sp: null\n",
                encoding="utf-8",
            )
            raw = {
                "initial_baseline": {
                    "label": "a0",
                    "launch_profile": "explicit_candidate",
                    "definition": "a0.yaml",
                },
                "baseline": {"max_num_seqs": 999},
            }
            resolved = tuning.resolve_initial_baseline_definition(raw, root)
            self.assertEqual(
                {"max_num_seqs": 48, "compilation_enable_sp": None},
                resolved["baseline"],
            )
            self.assertEqual(
                "a0-test",
                resolved["initial_baseline"]["resolved_definition"]["baseline_id"],
            )
            self.assertEqual(
                "a0.yaml",
                resolved["initial_baseline"]["resolved_definition"]["path"],
            )
            self.assertEqual(
                64,
                len(resolved["initial_baseline"]["resolved_definition"]["sha256"]),
            )

    def test_automatic_history_reuse_requires_benchmark_and_image_identity(self) -> None:
        raw = tuning.load_yaml(tuning.HERE / "config.yaml")
        project_root = tuning.KB_ROOT
        scenario = tuning.load_yaml(
            project_root
            / "workflow"
            / "search_space_compiler"
            / "scenario.glm52-a3-aligned-l1.yaml"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = root / "glm52_continuous_20260805_000000"
            history = session / "round_001_a1" / "06_agent_analysis" / "history_input.json"
            history.parent.mkdir(parents=True)
            history.write_text("[]\n", encoding="utf-8")
            selection = session / "00_search_space" / "search_space.compiled.yaml"
            selection.parent.mkdir(parents=True)
            tuning.save_yaml(
                selection,
                {"integration": {"search_space_profile": "automatic_registry_v1"}},
            )
            frozen = json.loads(json.dumps(raw))
            # A different repetition contract must not influence a new Session.
            frozen["benchmark"]["aligned_l1"]["repetitions"] = 3
            tuning.save_yaml(session / "session_config.yaml", frozen)
            tuning.save_yaml(
                session / "image_version_manifest.yaml",
                tuning.load_yaml(tuning.IMAGE_MANIFEST_FILE),
            )
            selected = _latest_history(
                root,
                benchmark_identity=_selected_benchmark_identity(raw, project_root),
                scenario_image=scenario["image"],
                project_root=project_root,
            )
            self.assertIsNone(selected)
            self.assertIsNone(
                _latest_previous_selection(
                    root,
                    "automatic_registry_v1",
                    benchmark_identity=_selected_benchmark_identity(
                        raw, project_root
                    ),
                    scenario_image=scenario["image"],
                    project_root=project_root,
                )
            )
            frozen["benchmark"]["aligned_l1"]["repetitions"] = raw["benchmark"][
                "aligned_l1"
            ]["repetitions"]
            tuning.save_yaml(session / "session_config.yaml", frozen)
            selected = _latest_history(
                root,
                benchmark_identity=_selected_benchmark_identity(raw, project_root),
                scenario_image=scenario["image"],
                project_root=project_root,
            )
            self.assertEqual(history, selected)
            self.assertEqual(
                selection,
                _latest_previous_selection(
                    root,
                    "automatic_registry_v1",
                    benchmark_identity=_selected_benchmark_identity(
                        raw, project_root
                    ),
                    scenario_image=scenario["image"],
                    project_root=project_root,
                ),
            )

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
            "best_anchor_coverage_v3",
            controller.effective_change_policy()["strategy_version"],
        )
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
                [
                    {
                        "parameter": "max_num_seqs",
                        "before": 48,
                        "after": 64,
                        "rationale": "A single change is invalid under V3 exploration.",
                    }
                ],
            )
        frozen = controller.config
        frozen["strategy"]["profiles_file"] = "missing-after-session.yaml"
        self.assertEqual(
            "best_anchor_coverage_v3",
            tuning.Controller(frozen).strategy_profile_name,
        )
        configured["strategy"]["profile"] = "missing"
        with self.assertRaisesRegex(ValueError, "Unknown strategy profile"):
            tuning.Controller(configured)

    def test_hierarchical_throughput_strategy_is_session_local_and_ordered(self) -> None:
        configured = config()
        configured["measurement_policy"] = {
            "latency_guardrail_mode": "hard",
            "aligned_l1": {"latency_guardrail_mode": "hard"},
        }
        configured["strategy"] = {
            "profile": "hierarchical_throughput_v1",
            "profiles_file": "workflow/continuous/strategy_profiles.yaml",
        }
        controller = tuning.Controller(configured)
        first = controller.effective_change_policy(
            [{"metrics": {"benchmark_mode": "aligned_l1"}}],
            [{"outcome": "success"}],
        )
        self.assertEqual("mtp_enablement", first["hierarchical_probe"]["name"])
        self.assertEqual([1, 1], first["preferred_parameters_per_round"])
        self.assertEqual(1, first["max_parameters_per_round"])
        self.assertEqual("advisory", controller.measurement_policy["latency_guardrail_mode"])
        self.assertEqual(
            "advisory",
            controller.measurement_policy["aligned_l1"]["latency_guardrail_mode"],
        )
        self.assertNotIn("reference_parameters", controller.strategy_profile)

        second = controller.effective_change_policy(
            [
                {"metrics": {"benchmark_mode": "aligned_l1"}},
                {"metrics": {"benchmark_mode": "aligned_l1"}},
            ],
            [{"outcome": "success"}, {"outcome": "success"}],
        )
        self.assertEqual("moe_communication", second["hierarchical_probe"]["name"])
        self.assertEqual("exploration", second["phase"])
        failed_attempt = controller.effective_change_policy(
            [{"metrics": {"benchmark_mode": "aligned_l1"}}],
            [{"outcome": "success"}, {"outcome": "failed"}],
        )
        self.assertEqual(
            "mtp_enablement", failed_attempt["hierarchical_probe"]["name"]
        )

        default = tuning.Controller(config())
        self.assertEqual("best_anchor_coverage_v2", default.strategy_profile_name)
        self.assertNotIn("hierarchical_probe", default.effective_change_policy())

    def test_hierarchical_layer_stays_for_promising_result_and_exits_at_budget(self) -> None:
        configured = config()
        configured["strategy"] = {
            "profile": "hierarchical_throughput_v1",
            "profiles_file": "workflow/continuous/strategy_profiles.yaml",
        }
        controller = tuning.Controller(configured)
        probes = controller.strategy_profile["hierarchy"]["ordered_probes"]
        history = [{"round": "b0"}, {"round": "a1"}]
        scores = {"b0": 100.0, "a1": 102.0, "a2": 104.0, "bad": 90.0}
        with (
            patch.object(
                controller,
                "primary_performance_score",
                side_effect=lambda item: scores[item["round"]],
            ),
            patch.object(
                controller,
                "best_accepted_anchor",
                return_value={"round": "b0"},
            ),
        ):
            promising = controller.hierarchical_search_state(
                history, probes, controller.strategy_profile["hierarchy"]
            )
            exhausted = controller.hierarchical_search_state(
                [*history, {"round": "a2"}],
                probes,
                controller.strategy_profile["hierarchy"],
            )
            negative = controller.hierarchical_search_state(
                [{"round": "b0"}, {"round": "bad"}],
                probes,
                controller.strategy_profile["hierarchy"],
            )
        self.assertEqual(0, promising["probe_index"])
        self.assertEqual(1, promising["measurements_in_probe"])
        self.assertEqual(1, exhausted["probe_index"])
        self.assertEqual(1, negative["probe_index"])

    def test_mtp_method_is_a_portrait_exempt_mechanical_companion(self) -> None:
        controller = self.controller
        controller.portrait_retriever = SimpleNamespace(
            retrieve=lambda *args, **kwargs: {
                "changed_parameters": [
                    {"canonical_name": "num_speculative_tokens", "variant_count": 1},
                    {"canonical_name": "speculative_config__method", "variant_count": 0},
                ]
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            round_dir = Path(temporary)
            (round_dir / "06_agent_analysis").mkdir()
            controller.write_selected_portrait_evidence(
                round_dir,
                [
                    {"parameter": "num_speculative_tokens"},
                    {"parameter": "speculative_config__method"},
                ],
            )
            self.assertTrue(
                (round_dir / "06_agent_analysis" / "selected_parameter_portraits.yaml").is_file()
            )

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

    def test_benchmark_profile_selects_and_freezes_one_definition(self) -> None:
        configured = config()
        controller = tuning.Controller(configured)
        self.assertEqual("aligned_l1_v4", controller.benchmark_profile_name)
        self.assertEqual("aligned_l1", controller.benchmark_mode)
        frozen = controller.config
        frozen["benchmark"]["profiles_file"] = "missing-after-session.yaml"
        self.assertEqual(
            "aligned_l1_v4",
            tuning.Controller(frozen).benchmark_profile_name,
        )
        configured["benchmark"]["profile"] = "missing"
        with self.assertRaisesRegex(ValueError, "Unknown benchmark profile"):
            tuning.Controller(configured)

    def test_public_vllm_benchmark_exports_frozen_definition_and_identity(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        configured["benchmark"]["profile"] = "vllm_bench_public_v1"
        controller = tuning.Controller(configured)
        environment = controller.candidate_env(
            configured["initial_baseline"]["label"], configured["baseline"]
        )
        self.assertEqual("vllm_bench_serve", controller.benchmark_mode)
        self.assertIn("BENCHMARK_MODE=vllm_bench_serve", environment)
        self.assertIn("BENCHMARK_DEFINITION_B64=", environment)
        self.assertIn("BENCHMARK_IDENTITY_JSON=", environment)

    def test_custom_benchmark_adapter_is_project_local_and_allowlisted(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        configured["benchmark"]["profile"] = "custom_adapter_v1"
        controller = tuning.Controller(configured)
        self.assertEqual("custom_adapter", controller.benchmark_mode)
        configured["benchmark"]["custom_benchmark"]["adapter_path"] = "../escape.py"
        with self.assertRaisesRegex(ValueError, "relative .py path"):
            tuning.Controller(configured)

    def test_benchmark_identity_prevents_cross_profile_history_comparison(self) -> None:
        one = {"benchmark_mode": "custom_adapter", "benchmark_identity": {"sha256": "a" * 64}}
        two = {"benchmark_mode": "custom_adapter", "benchmark_identity": {"sha256": "b" * 64}}
        self.assertNotEqual(
            tuning.Controller.benchmark_regime(one),
            tuning.Controller.benchmark_regime(two),
        )

    def test_public_benchmark_can_promote_an_accepted_best_anchor(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        configured["benchmark"]["profile"] = "vllm_bench_public_v1"
        configured["benchmark"]["vllm_bench_public"]["minimum_successful_requests"] = 1
        controller = tuning.Controller(configured)

        def item(round_name: str, throughput: float, ttft: float, tpot: float) -> dict:
            return {
                "round": round_name,
                "params": {"max_num_seqs": 48 if round_name == "b0" else 64},
                "metrics": {
                    "benchmark_mode": "vllm_bench_serve",
                    "metrics": {
                        "successful_requests": 1,
                        "failed_requests": 0,
                        "output_token_throughput": throughput,
                        "mean_ttft": ttft,
                        "mean_tpot": tpot,
                    },
                },
            }

        anchor = controller.best_accepted_anchor(
            [item("b0", 100.0, 10.0, 5.0), item("a1", 105.0, 10.5, 5.1)]
        )
        self.assertEqual("a1", anchor["round"])

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
            with patch.object(
                tuning.shutil, "which", return_value="C:/tools/codex.cmd"
            ):
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
        self.controller.validate_candidate(self.baseline, candidate, changes, decision)
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
        env_text = controller.candidate_env(
            configured["initial_baseline"]["label"], configured["baseline"]
        )
        self.assertIn(
            "LAUNCH_PROFILE=official_source_defaults_deployable", env_text
        )
        self.assertIn("BENCHMARK_MODE=aligned_l1", env_text)
        self.assertIn("BENCHMARK_REPETITIONS=1", env_text)
        self.assertIn(
            "BENCHMARK_SUITE='01_调优_结构化定长-v4.yaml'",
            env_text,
        )
        self.assertIn('"schema_files_sha256"', env_text)
        self.assertIn("SAFETENSORS_LOAD_STRATEGY=prefetch", env_text)
        self.assertIn("SAFETENSORS_PREFETCH_MODE=node_blocking", env_text)
        self.assertIn("SAFETENSORS_PREFETCH_NUM_THREADS=8", env_text)
        self.assertIn("SAFETENSORS_PREFETCH_BLOCK_SIZE=16777216", env_text)
        self.assertIn("MODEL_PATH=/models/share/GLM-5.2-w8a8", env_text)
        self.assertIn("SERVED_MODEL_NAME=glm-5", env_text)
        self.assertIn("SERVICE_PORT=8000", env_text)
        self.assertIn("MODEL_QUANTIZATION=ascend", env_text)
        self.assertIn("NIC_NAME=bond4.3000", env_text)
        self.assertIn("LAB_OUTPUT_ROOT=/mnt/host-model/", env_text)

        runtime = (tuning.HERE / "remote" / "common_runtime_loop.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("MODEL_PATH=/models/share/GLM-5.2-w8a8", runtime)
        self.assertIn('source "${INIT_ENV_SCRIPT}"', runtime)
        self.assertIn('--served-model-name "${SERVED_MODEL_NAME}"', runtime)
        self.assertIn(
            '--safetensors-load-strategy "${VLLM_SAFETENSORS_LOAD_STRATEGY}"',
            runtime,
        )
        self.assertIn('VLLM_SAFETENSORS_LOAD_STRATEGY=lazy', runtime)
        self.assertIn('prefetch_checkpoints_on_node()', runtime)
        self.assertIn(
            '--safetensors-prefetch-num-threads "${SAFETENSORS_PREFETCH_NUM_THREADS}"',
            runtime,
        )
        self.assertIn('if [[ "${LAUNCH_PROFILE}" == "explicit_candidate" ]]', runtime)
        self.assertLess(
            runtime.index(
                '--safetensors-load-strategy "${VLLM_SAFETENSORS_LOAD_STRATEGY}"'
            ),
            runtime.index('if [[ "${LAUNCH_PROFILE}" == "explicit_candidate" ]]'),
        )
        self.assertIn(
            'if [[ "${LAUNCH_PROFILE}" == "official_source_defaults_deployable" ]]',
            runtime,
        )
        self.assertIn(
            'VLLM_COMMON_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")', runtime
        )
        master_runtime = (
            tuning.HERE / "remote" / "run_master_loop.sh"
        ).read_text(encoding="utf-8")
        worker_runtime = (
            tuning.HERE / "remote" / "run_worker_loop.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("node_checkpoint_prefetch.py", tuning.REMOTE_SCRIPT_NAMES)
        self.assertTrue((tuning.HERE / "remote" / "node_checkpoint_prefetch.py").is_file())
        self.assertLess(
            master_runtime.index("prefetch_checkpoints_on_node"),
            master_runtime.index('vllm serve "${MODEL_PATH}"'),
        )
        self.assertLess(
            worker_runtime.index("prefetch_checkpoints_on_node"),
            worker_runtime.index('vllm serve "${MODEL_PATH}"'),
        )
        self.assertIn('${RUN_DIR}/MASTER_DONE', worker_runtime)
        self.assertIn('${RUN_DIR}/BENCHMARK_DONE', worker_runtime)
        self.assertIn('${RUN_DIR}/BENCHMARK_FAILED', worker_runtime)
        launcher = (tuning.HERE / "start_continuous.ps1").read_text(encoding="utf-8")
        self.assertIn("portrait_pipeline\\outputs\\ParameterYAML", launcher)
        self.assertIn("Test-Path -LiteralPath $portraitIndexPath", launcher)

    def test_remote_control_templates_follow_configured_server_identity(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        configured["remote_project"] = "/configured/project"
        configured["lab"]["lease_name"] = "configured-lease"
        controller = tuning.Controller(configured)
        lease = controller.render_remote_control_document("lease_loop.yaml")
        experiment = controller.render_remote_control_document("experiment_loop.yaml")
        self.assertEqual("configured-lease", lease["name"])
        for document in (lease, experiment):
            for task in document["tasks"]:
                self.assertIn("/configured/project/workflow/auto/", task["command"])
        repository, tag = controller.image_identity["reference"].rsplit(":", 1)
        self.assertEqual(repository, lease["image"])
        self.assertEqual(tag, lease["image_tag"])
        self.assertTrue(
            all(task["image"] == repository for task in experiment["tasks"])
        )

    def test_single_node_executor_renders_master_only(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        configured["topology"] = {
            "profile": "a3_single_16npu_dp2local_tp8",
            "profiles_file": "workflow/continuous/topology_profiles.yaml",
        }
        controller = tuning.Controller(configured)
        for name in ("lease_loop.yaml", "experiment_loop.yaml"):
            document = controller.render_remote_control_document(name)
            self.assertEqual(document["min_available"], 1)
            self.assertEqual(
                [task["name"].lower() for task in document["tasks"]], ["master"]
            )
        env_text = controller.candidate_env(
            configured["initial_baseline"]["label"], configured["baseline"]
        )
        self.assertIn("DATA_PARALLEL_SIZE_LOCAL=2", env_text)
        self.assertIn("TENSOR_PARALLEL_SIZE=8", env_text)
        self.assertIn("WORKER_REPLICAS=0", env_text)
        self.assertIn(
            "EXECUTOR_REMOTE_CONTRACT=single_node_local_dp_v1", env_text
        )

    def test_deployment_identity_is_configuration_driven(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        configured["deployment"].update(
            model_path="/configured/model",
            served_model_name="configured-model",
            service_port=18000,
            quantization="configured-quantization",
            network_interface="configured-nic",
        )
        configured["benchmark"]["aligned_l1"]["served_model"] = "configured-model"
        configured["benchmark"]["aligned_l1"]["service_port"] = 18000
        controller = tuning.Controller(configured)
        env_text = controller.candidate_env(
            configured["initial_baseline"]["label"], configured["baseline"]
        )
        self.assertIn("MODEL_PATH=/configured/model", env_text)
        self.assertIn("SERVED_MODEL_NAME=configured-model", env_text)
        self.assertIn("SERVICE_PORT=18000", env_text)
        self.assertIn("MODEL_QUANTIZATION=configured-quantization", env_text)
        self.assertIn("NIC_NAME=configured-nic", env_text)

    def test_benchmark_model_identity_must_match_deployment(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        configured["deployment"]["served_model_name"] = "different-model"
        controller = tuning.Controller(configured)
        with self.assertRaisesRegex(RuntimeError, "served_model must match"):
            controller.validate_deployment_configuration()

    def test_activation_approval_matches_the_runtime_manifest(self) -> None:
        tuning.validate_activation_approval()

    def test_activation_approval_rejects_version_drift(self) -> None:
        approval = tuning.load_yaml(tuning.ACTIVATION_FILE)
        approval["target"]["vllm_commit"] = "different-commit"
        with tempfile.TemporaryDirectory() as temporary:
            approval_path = Path(temporary) / "activation.approved.yaml"
            tuning.save_yaml(approval_path, approval)
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                tuning.validate_activation_approval(approval_path=approval_path)

    def test_check_ready_is_read_only_and_requires_an_idle_lease(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        controller = tuning.Controller(configured)
        with patch.object(
            controller, "ensure_lab_available", return_value="lease ready"
        ) as ensure:
            self.assertEqual("lease ready", controller.check_ready())
        ensure.assert_called_once_with()

    def test_resume_preflight_allows_the_recorded_task_to_keep_running(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        controller = tuning.Controller(configured)
        running = (
            "RESOURCE  status=active  nodes=2/2 Ready  npu=32/32\n"
            "SLOT service status running=2\n"
        )
        with patch.object(controller, "lease_status", return_value=running):
            self.assertEqual(running, controller.check_ready(require_idle_lease=False))

    def test_stop_active_task_uses_frozen_remote_configuration(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        configured["remote_project"] = "/configured/project"
        configured["lab"]["lease_name"] = "configured-lease"
        controller = tuning.Controller(configured)
        with patch.object(controller, "ssh", return_value="stopped") as ssh:
            self.assertEqual(
                "stopped",
                controller.stop_active_task(
                    {
                        "active_task_id": "service-task",
                        "execution_mode": "ktp_lab",
                        "lease_name": "configured-lease",
                    }
                ),
            )
        command = ssh.call_args.args[0]
        self.assertIn("cd /configured/project", command)
        self.assertIn("ktp-lab stop --lease configured-lease", command)

    def test_stop_without_active_task_never_contacts_server(self) -> None:
        with patch.object(self.controller, "ssh") as ssh:
            message = self.controller.stop_active_task({"active_task_id": None})
        self.assertIn("No active task", message)
        ssh.assert_not_called()

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

        configured["model_loading"]["safetensors_prefetch_num_threads"] = 8
        configured["model_loading"]["safetensors_prefetch_mode"] = "global_rank"
        with self.assertRaisesRegex(ValueError, "prefetch_mode"):
            tuning.Controller(configured)

    def test_rfork_profile_requires_scheduler_and_is_frozen_fail_closed(self) -> None:
        configured = config()
        configured["model_loading"] = {
            "profile": "rfork_external_seed_v1",
            "profiles_file": "workflow/continuous/model_loading_profiles.yaml",
        }
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "VLLMTKB_RFORK_SCHEDULER_URL"):
                resolve_model_loading_profile(configured, tuning.KB_ROOT)
        with patch.dict(
            "os.environ",
            {"VLLMTKB_RFORK_SCHEDULER_URL": "http://planner.example:1223"},
            clear=True,
        ):
            controller = tuning.Controller(configured)
        self.assertEqual("rfork", controller.model_load_format)
        self.assertTrue(controller.require_rfork_transfer)
        self.assertEqual(
            "http://planner.example:1223",
            controller.model_loader_extra_config["rfork_scheduler_url"],
        )
        environment = controller.candidate_env(
            "rfork-test", controller.config["baseline"]
        )
        self.assertIn("MODEL_LOADING_BACKEND=rfork", environment)
        self.assertIn("MODEL_LOAD_FORMAT=rfork", environment)
        self.assertIn("REQUIRE_RFORK_TRANSFER=true", environment)
        self.assertIn("planner.example:1223", environment)

    def test_b0_reconciles_source_resolved_values_before_agent_handoff(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        controller = tuning.Controller(configured)
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "session"
            round_dir = session / "round_000_b0_deployable"
            (round_dir / "02_parameters").mkdir(parents=True)
            (round_dir / "04_runtime").mkdir(parents=True)
            (round_dir / "04_runtime" / "master.log").write_text(
                "Chunked prefill is enabled with max_num_batched_tokens=2048.\n"
                "Initializing a V1 LLM engine with config: max_seq_len=64000, "
                "enable_prefix_caching=True, enable_chunked_prefill=True, "
                "compilation_config={'cudagraph_mode': "
                "<CUDAGraphMode.FULL_AND_PIECEWISE: (1, 1)>, "
                "'cudagraph_capture_sizes': [16, 32, 64, 128, 256], "
                "'max_cudagraph_capture_size': 256}\n",
                encoding="utf-8",
            )
            state = {
                "round_label": "b0_deployable",
                "current_candidate": dict(configured["baseline"]),
            }
            controller.reconcile_official_source_default_baseline(
                session, round_dir, state
            )
            self.assertTrue(state["official_source_defaults_reconciled"])
            self.assertEqual(64000, state["current_candidate"]["max_model_len"])
            self.assertEqual(
                [16, 32, 64, 128, 256],
                state["current_candidate"]["cudagraph_capture_sizes"],
            )
            self.assertTrue(
                (round_dir / "02_parameters" / "b0_effective_resolution.yaml").is_file()
            )
            evidence = tuning.load_yaml(
                round_dir / "02_parameters" / "b0_effective_resolution.yaml"
            )
            overrides = evidence["explicit_deployment_overrides"]
            self.assertEqual({"max_model_len", "rationale"}, set(overrides))
            self.assertEqual(64000, overrides["max_model_len"])
            self.assertIn("sole deployability override", overrides["rationale"])

    def test_explicit_automatic_search_space_resolves_and_manual_fallback_remains(
        self,
    ) -> None:
        project_root = tuning.KB_ROOT
        raw = tuning.load_yaml(tuning.HERE / "config.yaml")
        raw["search_space"]["profile"] = "automatic_registry_v1"
        with tempfile.TemporaryDirectory() as temporary:
            resolved, result = resolve_search_limits(
                raw,
                project_root=project_root,
                archive_root=Path(temporary),
            )
            self.assertIsNotNone(result)
            self.assertEqual(
                "automatic_registry", resolved["resolved_search_space"]["mode"]
            )
            self.assertEqual(
                "automatic_registry_v1",
                resolved["resolved_search_space"]["profile"],
            )
            self.assertGreaterEqual(
                result["summary"]["eligible_tunable_parameters"], 100
            )
            self.assertEqual(22, len(result["active_search_limits"]))
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
            self.assertEqual(raw["search_limits"], resolved["manual_search_limits"])
            self.assertEqual([64000], resolved["search_limits"]["max_model_len"])
            self.assertEqual(
                result["active_search_limits"]["async_scheduling"],
                resolved["search_limits"]["async_scheduling"],
            )
            self.assertEqual(
                {False, True}, set(resolved["search_limits"]["enable_expert_parallel"])
            )
            self.assertEqual(
                [
                    None,
                    [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192],
                ],
                resolved["search_limits"]["cudagraph_capture_sizes"],
            )
            if "cudagraph_capture_sizes" not in result["active_search_limits"]:
                self.assertIn(
                    "cudagraph_capture_sizes",
                    resolved["resolved_search_space"]["derived_runtime_parameters"],
                )
            controller = tuning.Controller(resolved)
            generated_candidate = dict(resolved["baseline"])
            generated_candidate["speculative_config__method"] = "mtp"
            generated_candidate["num_speculative_tokens"] = 1
            environment = controller.candidate_env("profile-test", generated_candidate)
            self.assertIn("RUNTIME_INJECTION_MODE=generated_v1", environment)
            encoded = next(
                line.split("=", 1)[1]
                for line in environment.splitlines()
                if line.startswith("RUNTIME_INJECTION_PAYLOAD_B64=")
            )
            payload = json.loads(base64.b64decode(encoded).decode("utf-8"))
            self.assertEqual(
                "mtp", payload["json_configs"]["speculative_config"]["method"]
            )
            self.assertFalse(
                payload["json_configs"]["speculative_config"][
                    "disable_padded_drafter_batch"
                ]
            )
            mismatched_graph = dict(resolved["baseline"])
            mismatched_graph["max_cudagraph_capture_size"] = 256
            mismatched_graph["cudagraph_capture_sizes"] = [
                16,
                32,
                48,
                64,
                80,
                96,
                112,
                128,
                144,
                160,
                176,
                192,
            ]
            with self.assertRaisesRegex(ValueError, "must equal the largest explicit"):
                controller.validate_runtime_configuration(mismatched_graph)
            mismatched_graph["cudagraph_capture_sizes"] = None
            controller.validate_runtime_configuration(mismatched_graph)

            manual_raw = dict(raw)
            manual_raw.pop("search_space", None)
            manual_raw["search_limits_mode"] = "manual"
            manual, manual_result = resolve_search_limits(
                manual_raw,
                project_root=project_root,
                archive_root=Path(temporary),
            )
            self.assertIsNone(manual_result)
            self.assertEqual(22, len(manual["search_limits"]))

    def test_default_search_space_is_automatic_registry(self) -> None:
        raw = tuning.load_yaml(tuning.HERE / "config.yaml")
        profile_document = tuning.load_yaml(
            tuning.KB_ROOT / "workflow" / "search_space_profiles.yaml"
        )
        self.assertEqual(
            profile_document["default_profile"], raw["search_space"]["profile"]
        )
        with tempfile.TemporaryDirectory() as temporary:
            resolved, result = resolve_search_limits(
                raw,
                project_root=tuning.KB_ROOT,
                archive_root=Path(temporary),
            )
        self.assertEqual(
            "automatic_registry_v1", resolved["resolved_search_space"]["profile"]
        )
        self.assertEqual(
            "automatic_registry", resolved["resolved_search_space"]["mode"]
        )
        self.assertEqual(142, result["summary"]["registry_parameters"])
        self.assertEqual(22, result["summary"]["active_parameters"])
        self.assertEqual(80, result["summary"]["reserve_parameters"])
        self.assertEqual(40, result["summary"]["fixed_parameters"])
        self.assertEqual(0, result["summary"]["rejected_parameters"])
        self.assertEqual(
            ["cudagraph_capture_sizes"],
            resolved["resolved_search_space"]["derived_runtime_parameters"],
        )
        self.assertEqual(
            resolved["search_limits"], result["integration"]["effective_search_limits"]
        )
        for name in {"num_speculative_tokens", "async_scheduling", "fused_mc2"}:
            self.assertIn(name, result["active_search_limits"])

    def test_local_and_server_experiment_defaults_are_aligned(self) -> None:
        local = tuning.load_config(tuning.HERE / "config.yaml")
        server = tuning.load_config(
            tuning.HERE / "server_autonomous" / "config.yaml"
        )
        for section, field in (
            ("runtime", "profile"),
            ("topology", "profile"),
            ("search_space", "profile"),
            ("strategy", "profile"),
            ("benchmark", "profile"),
            ("agent", "provider"),
            ("initial_baseline", "definition"),
        ):
            self.assertEqual(local[section][field], server[section][field])
        self.assertEqual(
            local["search_space"]["history_source"],
            server["search_space"]["history_source"],
        )
        self.assertEqual(local["model_loading"], server["model_loading"])
        for field in (
            "model",
            "thinking",
            "reasoning_effort",
            "response_format",
            "max_tokens",
            "max_api_retries",
        ):
            self.assertEqual(
                local["agent"]["providers"]["deepseek"][field],
                server["agent"]["providers"]["deepseek"][field],
            )
        self.assertNotEqual(local["remote_transport"], server["remote_transport"])
        self.assertNotEqual(local["lab"]["lease_name"], server["lab"]["lease_name"])

    def test_migration_parameters_are_validated_and_rendered_end_to_end(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        controller = tuning.Controller(configured)
        candidate = {
            **configured["baseline"],
            "async_scheduling": True,
            "enable_expert_parallel": True,
            "num_speculative_tokens": 3,
            "fused_mc2": 1,
            "enable_balance_scheduling": True,
            "enable_reduce_sample": True,
            "speculative_config__enforce_eager": True,
        }
        controller.validate_candidate_invariants(candidate)
        environment = controller.candidate_env("migration-chain", candidate)
        self.assertIn("ADDITIONAL_CONFIG_ENABLE_BALANCE_SCHEDULING=true", environment)
        self.assertIn("ADDITIONAL_CONFIG_ENABLE_REDUCE_SAMPLE=true", environment)
        self.assertIn("SPECULATIVE_CONFIG_ENFORCE_EAGER_JSON=true", environment)

        invalid_fused = {**candidate, "enable_expert_parallel": False}
        with self.assertRaisesRegex(ValueError, "fused_mc2 requires"):
            controller.validate_candidate_invariants(invalid_fused)
        invalid_draft = {
            **candidate,
            "num_speculative_tokens": 0,
            "async_scheduling": False,
        }
        with self.assertRaisesRegex(ValueError, "only applies"):
            controller.validate_candidate_invariants(invalid_draft)

    def test_automatic_and_curated_profiles_have_auditable_overlap(self) -> None:
        raw = tuning.load_yaml(tuning.HERE / "config.yaml")
        raw["search_space"]["profile"] = "automatic_registry_v1"
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary)
            automatic, automatic_result = resolve_search_limits(
                raw, project_root=tuning.KB_ROOT, archive_root=archive
            )
            curated_raw = tuning.load_yaml(tuning.HERE / "config.yaml")
            curated_raw["search_space"]["profile"] = "curated_registry_v1"
            curated, curated_result = resolve_search_limits(
                curated_raw, project_root=tuning.KB_ROOT, archive_root=archive
            )
            session = archive / "frozen-session"
            write_session_search_space(
                session, result=automatic_result, config=automatic
            )
            self.assertTrue(
                (session / "00_search_space" / "search_space_profile.yaml").is_file()
            )
            self.assertTrue(
                (session / "00_search_space" / "registry.generated.yaml").is_file()
            )
            self.assertTrue(
                (session / "00_search_space" / "registry.audit.yaml").is_file()
            )
        registry = tuning.load_yaml(
            tuning.KB_ROOT / "workflow" / "search_space_compiler" / "registry.yaml"
        )
        automatic_active = set(automatic_result["active_search_limits"])
        curated_active = set(curated_result["active_search_limits"])
        self.assertEqual(26, len(registry["parameters"]))
        self.assertEqual(22, len(automatic_active))
        self.assertEqual(80, automatic_result["summary"]["reserve_parameters"])
        self.assertEqual(14, len(automatic_active & curated_active))
        self.assertFalse(
            automatic_result["automatic_registry_snapshot"]["audit"][
                "existing_registry_dependency"
            ]
        )
        self.assertEqual(
            "automatic_registry_v1",
            automatic["resolved_search_space"]["profile"],
        )
        self.assertEqual(
            "curated_registry_v1", curated["resolved_search_space"]["profile"]
        )

    def test_b0_reconciliation_updates_frozen_automatic_domains(self) -> None:
        raw = tuning.load_yaml(tuning.HERE / "config.yaml")
        raw["search_space"]["profile"] = "automatic_registry_v1"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved, result = resolve_search_limits(
                raw, project_root=tuning.KB_ROOT, archive_root=root
            )
            controller = tuning.Controller(resolved, search_space_result=result)
            session = root / "session"
            round_dir = session / "round_000_b0_deployable"
            (round_dir / "02_parameters").mkdir(parents=True)
            (round_dir / "04_runtime").mkdir(parents=True)
            (round_dir / "04_runtime" / "master.log").write_text(
                "Chunked prefill is enabled with max_num_batched_tokens=2048.\n"
                "Initializing a V1 LLM engine with config: max_seq_len=64000, "
                "enable_prefix_caching=True, enable_chunked_prefill=True, "
                "compilation_config={'cudagraph_mode': "
                "<CUDAGraphMode.FULL_AND_PIECEWISE: (1, 1)>, "
                "'cudagraph_capture_sizes': [16, 32, 64, 128, 256], "
                "'max_cudagraph_capture_size': 256}\n",
                encoding="utf-8",
            )
            state = {
                "round_label": "b0_deployable",
                "current_candidate": dict(resolved["baseline"]),
            }
            controller.reconcile_official_source_default_baseline(
                session, round_dir, state
            )
            controller.validate_candidate_invariants(state["current_candidate"])
            mtp_candidate = {
                **state["current_candidate"],
                "async_scheduling": True,
                "num_speculative_tokens": 1,
                "speculative_config__method": "mtp",
            }
            # This is the first post-B0 transition the automatic profile must
            # support; before async_scheduling became Active it was impossible.
            controller.validate_candidate_invariants(mtp_candidate)
            active_names = {
                item["canonical_name"]
                for item in controller.automatic_registry_validation["compiled"][
                    "active_parameters"
                ]
            }
            self.assertNotIn("cudagraph_capture_sizes", active_names)
            self.assertIn(
                [16, 32, 64, 128, 256],
                controller.config["search_limits"]["cudagraph_capture_sizes"],
            )
            self.assertEqual(
                [256, 32, 64, 128, 192],
                controller.config["search_limits"]["max_num_seqs"],
            )
            self.assertEqual(
                [2048, 4096, 8192, 16384, 32768],
                controller.config["search_limits"]["max_num_batched_tokens"],
            )
            frozen = tuning.load_yaml(session / "session_config.yaml")
            self.assertEqual(
                controller.config["automatic_registry_validation"],
                frozen["automatic_registry_validation"],
            )

    def test_reserved_decode_context_parameter_keeps_an_injection_contract(
        self,
    ) -> None:
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
                    "repetition_count": configured["benchmark"]["aligned_l1"][
                        "repetitions"
                    ],
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

    def test_hierarchical_strategy_treats_aligned_latency_as_advisory(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        configured["strategy"]["profile"] = "hierarchical_throughput_v1"
        controller = tuning.Controller(configured)
        workloads = list(configured["benchmark"]["aligned_l1"]["workloads"])

        def payload(score: float, throughput: float, latency: float) -> dict:
            cases = []
            for workload in workloads:
                for concurrency in (1, 16, 32):
                    cases.append(
                        {
                            "workload": workload,
                            "concurrency": concurrency,
                            "aggregate_output_tps": throughput,
                            "ttft_p50_ms": 100.0 * latency,
                            "ttft_p90_ms": 120.0 * latency,
                            "tpot_p50_ms": 10.0 * latency,
                            "tpot_p90_ms": 12.0 * latency,
                        }
                    )
            return {
                "benchmark_mode": "aligned_l1",
                "l1": {
                    "all_repetitions_gate_passed": True,
                    "repetition_count": configured["benchmark"]["aligned_l1"][
                        "repetitions"
                    ],
                    "primary_concurrency": 32,
                    "primary_aggregate_output_tps_geomean": score,
                    "primary_score_cv_percent": 0.1,
                    "cases": cases,
                },
            }

        assessment = controller.assess_aligned_l1(
            [
                {"metrics": payload(100.0, 100.0, 1.0)},
                {"metrics": payload(105.0, 105.0, 2.0)},
            ]
        )
        self.assertTrue(assessment["eligible_as_improvement"])
        self.assertEqual([], assessment["violations"])
        self.assertTrue(assessment["advisories"])
        self.assertEqual("advisory", assessment["latency_guardrail_mode"])

    def test_new_session_can_import_only_identity_matched_baseline_evidence(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        configured["strategy"]["profile"] = "hierarchical_throughput_v1"
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "experiments"
            archive.mkdir()
            source = archive / "source"
            source_round = source / "round_000_b0_deployable"
            for child in (
                "02_parameters",
                "03_submission",
                "04_runtime",
                "05_results",
                "06_agent_analysis",
            ):
                (source_round / child).mkdir(parents=True)
            target = archive / "target"
            target.mkdir()
            configured["baseline_reuse"] = {"source_session": str(source)}
            controller = tuning.Controller(configured)
            tuning.save_yaml(source / "session_config.yaml", controller.config)
            tuning.save_yaml(
                source_round / "02_parameters" / "candidate_params.yaml",
                controller.config["baseline"],
            )
            (source_round / "05_results" / "metrics.json").write_text(
                json.dumps({"benchmark_mode": controller.benchmark_mode}),
                encoding="utf-8",
            )
            (source_round / "06_agent_analysis" / "old_decision.json").write_text(
                "{}\n", encoding="utf-8"
            )
            state = {
                "session_id": "target",
                "session_dir": str(target),
                "current_candidate": controller.config["baseline"],
            }
            with (
                patch.object(tuning, "ARCHIVE_ROOT", archive),
                patch.object(tuning, "STATE_FILE", Path(temporary) / "state.json"),
                patch.object(controller, "write_context"),
                patch.object(controller, "run_query"),
                patch.object(controller, "reconcile_official_source_default_baseline"),
            ):
                target_round, imported = controller.import_completed_baseline(
                    target, state
                )
            self.assertTrue((target_round / "05_results" / "metrics.json").is_file())
            self.assertFalse(
                (target_round / "06_agent_analysis" / "old_decision.json").exists()
            )
            self.assertEqual(str(source), imported["baseline_reuse"]["source_session"])
            self.assertTrue(all(imported["baseline_reuse"]["identity_checks"].values()))

            mismatched = tuning.load_yaml(source / "session_config.yaml")
            mismatched["deployment"]["served_model_name"] = "different"
            tuning.save_yaml(source / "session_config.yaml", mismatched)
            with patch.object(tuning, "ARCHIVE_ROOT", archive):
                with self.assertRaisesRegex(ValueError, "deployment"):
                    controller.import_completed_baseline(target, state)

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
                    90.0 if workload == workloads[0] and concurrency == 32 else 110.0
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
            any(
                "aggregate output TPS ratio" in item
                for item in assessment["violations"]
            )
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

    def test_transient_lease_not_ready_waits_until_recovered(self) -> None:
        unavailable = """\
LEASE  example
RESOURCE  status=active  nodes=0/2 Ready  npu=32/32
SLOT  service
  status   idle=2
"""
        ready = """\
LEASE  example
RESOURCE  status=active  nodes=2/2 Ready  npu=32/32
SLOT  service
  status   idle=2
"""
        self.controller.lab = {"lease_name": "example"}
        self.controller.lease_readiness_wait_seconds = 60
        self.controller.lease_readiness_poll_seconds = 1
        with (
            patch.object(
                self.controller,
                "lease_status",
                side_effect=[unavailable, unavailable, ready],
            ) as status,
            patch.object(tuning.time, "sleep") as sleep,
        ):
            self.assertEqual(
                ready,
                self.controller.wait_for_lab_available(deadline=10**12),
            )
        self.assertEqual(3, status.call_count)
        self.assertEqual(2, sleep.call_count)

    def test_submission_retries_protocol_error_after_readiness_race(self) -> None:
        protocol_error = RuntimeError(
            "resource admission requires control protocol v2 workers"
        )
        self.controller.lease_submission_retry_limit = 2
        self.controller.lease_readiness_poll_seconds = 1
        with (
            patch.object(
                self.controller,
                "ssh",
                side_effect=[protocol_error, "submitted"],
            ) as remote,
            patch.object(
                self.controller,
                "wait_for_lab_available",
                return_value="ready",
            ) as wait_ready,
            patch.object(tuning.time, "sleep") as sleep,
        ):
            output = self.controller.run_lab_submission_with_readiness_retry(
                "ktp-lab run", deadline=10**12
            )
        self.assertEqual("submitted", output)
        self.assertEqual(2, remote.call_count)
        wait_ready.assert_called_once_with(deadline=10**12)
        sleep.assert_called_once()

    def test_submission_does_not_retry_unrelated_error(self) -> None:
        self.controller.lease_submission_retry_limit = 6
        with (
            patch.object(
                self.controller,
                "ssh",
                side_effect=RuntimeError("invalid candidate"),
            ) as remote,
            patch.object(
                self.controller,
                "wait_for_lab_available",
            ) as wait_ready,
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid candidate"):
                self.controller.run_lab_submission_with_readiness_retry(
                    "ktp-lab run", deadline=10**12
                )
        remote.assert_called_once()
        wait_ready.assert_not_called()

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

    def test_offline_dry_run_never_queries_a_lease(self) -> None:
        self.controller.offline_dry_run = True
        self.controller.lab = {"lease_name": "isolated"}
        with tempfile.TemporaryDirectory() as temporary:
            round_dir = Path(temporary)
            (round_dir / "03_submission").mkdir()
            (round_dir / "02_parameters").mkdir()
            with (
                patch.object(self.controller, "validate_runtime_configuration"),
                patch.object(
                    self.controller,
                    "ensure_lab_available",
                    side_effect=AssertionError("offline dry-run must not query Lease"),
                ),
            ):
                task_id, _ = self.controller.submit_lab(
                    round_dir, "a0", self.baseline, dry_run=True
                )
            self.assertIsNone(task_id)
            submission = json.loads(
                (round_dir / "03_submission" / "submission.json").read_text()
            )
            self.assertTrue(submission["dry_run"])

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
            self.assertEqual(
                [item["outcome"] for item in attempts], ["success", "failed"]
            )
            self.assertTrue(
                self.controller.candidate_was_attempted(session, failed_candidate)
            )

    def test_exploration_memory_uses_accepted_anchor_and_preserves_causality(
        self,
    ) -> None:
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
                    "repetition_count": configured["benchmark"]["aligned_l1"][
                        "repetitions"
                    ],
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

    def test_optional_best_anchor_coverage_strategy_is_exposed_to_agent(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        configured["strategy"]["profile"] = "best_anchor_coverage_v2"
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

    def test_failure_recovery_allows_proven_runtime_bug_parameter_bypass(
        self,
    ) -> None:
        configured = config()
        configured["search_limits"]["num_speculative_tokens"] = [1, 2, 3]
        controller = tuning.Controller(configured)
        current = dict(self.baseline, num_speculative_tokens=2)
        candidate = dict(current, num_speculative_tokens=1)
        decision = {
            "action": "adjust_parameters",
            "classification": "model_or_runtime_bug",
            "safe_to_automate": True,
            "change_strategy": "single_parameter",
            "evidence": [
                "Two identical startup failures occurred in the fused MLAPO path."
            ],
            "interaction_analysis": [
                "Reducing speculative tokens bypasses only the proven failing path."
            ],
            "constraint_checks": [
                "num_speculative_tokens=1 is allowed by the frozen search limits."
            ],
            "candidate": candidate,
            "changes": [
                {
                    "parameter": "num_speculative_tokens",
                    "before": 2,
                    "after": 1,
                    "rationale": "Bypass the reproducibly failing runtime path.",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertIsNone(
                controller.validate_failure_decision(
                    Path(temp_dir), decision, current
                )
            )

    def test_failure_recovery_enforces_classification_action_contract(self) -> None:
        controller = tuning.Controller(config())
        current = dict(self.baseline)

        def no_change_decision(
            classification: str,
            action: str,
            safe_to_automate: bool,
        ) -> dict[str, object]:
            return {
                "action": action,
                "classification": classification,
                "safe_to_automate": safe_to_automate,
                "change_strategy": "none",
                "evidence": ["The archived logs provide concrete failure evidence."],
                "interaction_analysis": [],
                "constraint_checks": [],
                "candidate": current,
                "changes": [],
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir)
            self.assertIsNone(
                controller.validate_failure_decision(
                    session,
                    no_change_decision(
                        "transient_infrastructure", "retry_same", True
                    ),
                    current,
                )
            )
            self.assertIsNone(
                controller.validate_failure_decision(
                    session,
                    no_change_decision("image_or_dependency", "pause_for_human", False),
                    current,
                )
            )
            with self.assertRaisesRegex(ValueError, "cannot use action"):
                controller.validate_failure_decision(
                    session,
                    no_change_decision("image_or_dependency", "retry_same", True),
                    current,
                )
            with self.assertRaisesRegex(ValueError, "requires safe_to_automate=true"):
                controller.validate_failure_decision(
                    session,
                    no_change_decision(
                        "transient_infrastructure", "retry_same", False
                    ),
                    current,
                )
            with self.assertRaisesRegex(ValueError, "requires safe_to_automate=false"):
                controller.validate_failure_decision(
                    session,
                    no_change_decision("unknown", "pause_for_human", True),
                    current,
                )

    def test_failure_schema_matches_controller_action_contract(self) -> None:
        schema = json.loads(
            (tuning.HERE / "failure_decision.schema.json").read_text(encoding="utf-8")
        )
        branches = {
            branch["if"]["properties"]["action"]["const"]: branch["then"]
            for branch in schema["allOf"]
        }
        self.assertEqual(
            set(
                branches["adjust_parameters"]["properties"]["classification"][
                    "enum"
                ]
            ),
            set(tuning.FAILURE_ADJUSTABLE_CLASSIFICATIONS),
        )
        self.assertEqual(
            set(
                branches["retry_same"]["properties"]["classification"]["enum"]
            ),
            set(tuning.FAILURE_RETRYABLE_CLASSIFICATIONS),
        )
        self.assertTrue(
            branches["adjust_parameters"]["properties"]["safe_to_automate"][
                "const"
            ]
        )
        self.assertFalse(
            branches["pause_for_human"]["properties"]["safe_to_automate"][
                "const"
            ]
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

    def test_startup_zmq_collision_gets_identical_bounded_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            round_dir = Path(temporary)
            runtime_dir = round_dir / "04_runtime"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "master.log").write_text(
                "ApiServer_1 zmq.error.ZMQError: Address already in use "
                "(addr='tcp://10.1.30.10:39099')\n"
                "Process ApiServer_1 (PID: 123) died with exit code 1\n",
                encoding="utf-8",
            )

            decision = self.controller.deterministic_startup_port_retry(
                round_dir, self.baseline
            )

            self.assertIsNotNone(decision)
            self.assertEqual("retry_same", decision["action"])
            self.assertEqual("transient_infrastructure", decision["classification"])
            self.assertEqual(self.baseline, decision["candidate"])

    def test_b0_retry_preserves_launch_profile_and_reconciliation(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        controller = tuning.Controller(configured)
        env_text = controller.candidate_env(
            "a0r1",
            configured["baseline"],
            launch_profile=tuning.B0_LAUNCH_PROFILE,
        )
        self.assertIn(
            "LAUNCH_PROFILE=official_source_defaults_deployable", env_text
        )
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "session"
            round_dir = session / "round_001_a0r1"
            (round_dir / "02_parameters").mkdir(parents=True)
            (round_dir / "04_runtime").mkdir(parents=True)
            (round_dir / "02_parameters" / "candidate.env").write_text(
                env_text, encoding="utf-8"
            )
            (round_dir / "04_runtime" / "master.log").write_text(
                "Chunked prefill is enabled with max_num_batched_tokens=2048.\n"
                "Initializing a V1 LLM engine with config: max_seq_len=64000, "
                "enable_prefix_caching=True, enable_chunked_prefill=True, "
                "compilation_config={'cudagraph_mode': "
                "<CUDAGraphMode.FULL_AND_PIECEWISE: (1, 1)>, "
                "'cudagraph_capture_sizes': [16, 32, 64, 128, 256], "
                "'max_cudagraph_capture_size': 256}\n",
                encoding="utf-8",
            )
            state = {
                "round_label": "a0r1",
                "current_candidate": dict(configured["baseline"]),
            }
            controller.reconcile_official_source_default_baseline(
                session, round_dir, state
            )
            self.assertTrue(state["official_source_defaults_reconciled"])

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
