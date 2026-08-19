from __future__ import annotations

import datetime as dt
import base64
import copy
import hashlib
import json
import re
import shlex
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
    _selected_topology_identity,
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


def production_b0_config() -> dict:
    """Select the preserved legacy B0 route for B0-specific regression tests."""
    value = tuning.load_yaml(tuning.HERE / "config.yaml")
    value["runtime"]["profile"] = "glm52_w8a8_a3_dp2_tp16"
    value["benchmark"]["profile"] = "aligned_l1_v4"
    value["search_space"]["profile"] = "automatic_registry_v1"
    value["strategy"]["profile"] = "hierarchical_throughput_v1"
    value["initial_baseline"] = {
        "label": "b0_deployable",
        "launch_profile": "official_source_defaults_deployable",
        "definition": "workflow/baselines/b0_deployable_64k.yaml",
    }
    return value


class ControllerTests(unittest.TestCase):
    def test_disabled_campaign_exposes_only_fixed_dp2_identity_to_agent(self) -> None:
        configured = config()
        configured["topology_campaign"] = {"enabled": False}
        configured["topology"] = {
            "profile": "a3_dp2_tp16",
            "profiles_file": "workflow/continuous/topology_profiles.yaml",
        }
        controller = tuning.Controller(configured)
        self.assertFalse(controller.topology_campaign_enabled)
        self.assertEqual("fixed_topology_session", controller.topology_plan["stage"])
        self.assertEqual("a3_dp2_tp16", controller.topology_plan["selected_profile"])
        self.assertEqual(
            ["a3_dp2_tp16"], controller.topology_plan["eligible_profiles"]
        )
        policy = controller.effective_change_policy()
        self.assertEqual("fixed_session", policy["outer_topology_stage"]["mode"])
        self.assertEqual(
            "a3_dp2_tp16",
            policy["outer_topology_stage"]["selected_profile"],
        )

    def test_disabled_campaign_can_freeze_an_independent_dp4_identity(self) -> None:
        configured = config()
        configured["topology_campaign"] = {"enabled": False}
        configured["topology"] = {
            "profile": "a3_dp4_tp8",
            "profiles_file": "workflow/continuous/topology_profiles.yaml",
        }
        configured["initial_baseline"] = {
            "label": "a8_dp4_tp8_fixed_v1",
            "definition": "workflow/baselines/a8_glm52_w8a8_dp4_tp8_fixed_v1.yaml",
        }
        controller = tuning.Controller(configured)
        self.assertEqual("fixed_topology_session", controller.topology_plan["stage"])
        self.assertEqual("a3_dp4_tp8", controller.topology_plan["selected_profile"])
        self.assertEqual(["a3_dp4_tp8"], controller.topology_plan["eligible_profiles"])
        self.assertIn("a3_dp4_tp8 is operator-frozen", controller.topology_plan["selection_reason"])
        candidate = controller.topology_plan["candidates"][0]
        self.assertEqual("a8_dp4_tp8_fixed_v1", candidate["session_baseline_label"])
        self.assertTrue(candidate["session_baseline_definition"].endswith("dp4_tp8_fixed_v1.yaml"))
        self.assertEqual(4, controller.topology["data_parallel_size"])
        self.assertEqual(2, controller.topology["data_parallel_size_local"])
        self.assertEqual(8, controller.topology["tensor_parallel_size"])
        self.assertEqual(
            "distributed_local_dp_v1",
            controller.topology["resolved_executor"]["remote_contract"],
        )

    def test_outer_campaign_budget_pause_status_is_candidate_indexed(self) -> None:
        self.assertIsNone(
            tuning.session_budget_pause_status(
                1, 2, topology_feasibility_only=False
            )
        )
        self.assertEqual(
            "budget_paused",
            tuning.session_budget_pause_status(
                2, 2, topology_feasibility_only=False
            ),
        )
        self.assertEqual(
            "topology_feasibility_passed",
            tuning.session_budget_pause_status(
                0, 0, topology_feasibility_only=True
            ),
        )

    def test_state_write_is_atomic_and_recovers_last_known_good(self) -> None:
        controller = tuning.Controller(config())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(tuning, "STATE_FILE", root / "state.json"):
                first = {"status": "running", "round_index": 1}
                controller.save_state(first)
                second = {"status": "running", "round_index": 2}
                controller.save_state(second)
                self.assertEqual(2, tuning.load_controller_state()["round_index"])
                (root / "state.json").write_text('{"status":', encoding="utf-8")
                recovered = tuning.load_controller_state()
                self.assertEqual(2, recovered["round_index"])
                self.assertTrue((root / "state_recovery.audit.json").is_file())

    def test_collect_control_plane_outage_has_bounded_retry(self) -> None:
        controller = tuning.Controller(config())
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            controller,
            "collect",
            side_effect=[RuntimeError("ssh reset"), {"metrics.json": False}],
        ) as collect, patch.object(tuning.time, "sleep"):
            result = controller.collect_with_retry("run-1", Path(temporary))
        self.assertFalse(result["metrics.json"])
        self.assertEqual(2, collect.call_count)

    def test_collect_control_plane_outage_escalates_as_recoverable(self) -> None:
        controller = tuning.Controller(config())
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            controller, "collect", side_effect=RuntimeError("ssh reset")
        ), patch.object(tuning.time, "sleep"):
            with self.assertRaises(tuning.RecoverableControllerIOError):
                controller.collect_with_retry("run-1", Path(temporary))

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
            frozen["benchmark"]["aligned_l1_fast_v2"]["repetitions"] = 3
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
            frozen["benchmark"]["aligned_l1_fast_v2"]["repetitions"] = raw["benchmark"][
                "aligned_l1_fast_v2"
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

    def test_automatic_history_and_rotation_are_topology_keyed(self) -> None:
        raw = tuning.load_yaml(tuning.HERE / "config.yaml")
        project_root = tuning.KB_ROOT
        scenario = tuning.load_yaml(
            project_root
            / "workflow"
            / "search_space_compiler"
            / "scenario.glm52-a3-aligned-l1.yaml"
        )
        topology_identity = _selected_topology_identity(raw, project_root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = root / "dp4_session"
            history = session / "round_001_a1" / "06_agent_analysis" / "history_input.json"
            history.parent.mkdir(parents=True)
            history.write_text("[]\n", encoding="utf-8")
            selection = session / "00_search_space" / "search_space.compiled.yaml"
            selection.parent.mkdir(parents=True)
            tuning.save_yaml(
                selection,
                {
                    "integration": {
                        "search_space_profile": "automatic_registry_a8_frontier_v3"
                    }
                },
            )
            frozen = copy.deepcopy(raw)
            frozen["topology"]["profile"] = "a3_dp2_tp16"
            tuning.save_yaml(session / "session_config.yaml", frozen)
            tuning.save_yaml(
                session / "image_version_manifest.yaml",
                tuning.load_yaml(tuning.IMAGE_MANIFEST_FILE),
            )
            kwargs = {
                "benchmark_identity": _selected_benchmark_identity(raw, project_root),
                "scenario_image": scenario["image"],
                "project_root": project_root,
                "topology_identity": topology_identity,
            }
            self.assertIsNone(_latest_history(root, **kwargs))
            self.assertIsNone(
                _latest_previous_selection(
                    root, "automatic_registry_a8_frontier_v3", **kwargs
                )
            )
            frozen["topology"]["profile"] = "a3_dp4_tp8"
            tuning.save_yaml(session / "session_config.yaml", frozen)
            self.assertEqual(history, _latest_history(root, **kwargs))
            self.assertEqual(
                selection,
                _latest_previous_selection(
                    root, "automatic_registry_a8_frontier_v3", **kwargs
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

    def test_mtp_graph_shapes_use_query_length_not_tp_divisibility(self) -> None:
        configured = config()
        configured["baseline"].update(
            speculative_config__method="mtp",
            cudagraph_capture_sizes=[16, 32, 48, 64],
            max_cudagraph_capture_size=64,
        )
        configured["search_limits"].update(
            num_speculative_tokens=[2, 3],
            speculative_config__method=["mtp", None],
            cudagraph_capture_sizes=[
                [16, 32, 48, 64],
                [3, 6, 12, 24, 48, 72, 96],
                [4, 8, 16, 32, 64],
            ],
            max_cudagraph_capture_size=[64, 96],
        )
        configured["automatic_registry_validation"] = {
            "scenario": {},
            "compatibility_policy": {"schema_version": 1},
            "active_injections": {
                "speculative_config__method": {
                    "kind": "json_path",
                    "path": ["speculative_config", "method"],
                }
            }
        }
        controller = tuning.Controller(configured)
        controller.automatic_registry_validation = None
        controller.automatic_compatibility = None
        candidate = dict(controller.config["baseline"])
        candidate.update(
            num_speculative_tokens=2,
            async_scheduling=True,
            speculative_config__method="mtp",
            compilation_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[3, 6, 12, 24, 48, 72, 96],
            max_cudagraph_capture_size=96,
        )
        controller.validate_candidate_invariants(candidate)
        candidate["cudagraph_capture_sizes"] = [4, 8, 16, 32, 64]
        candidate["max_cudagraph_capture_size"] = 64
        with self.assertRaisesRegex(ValueError, "multiples"):
            controller.validate_candidate_invariants(candidate)

    def test_flashcomm1_enforces_effective_sequence_parallel_graph_shapes(self) -> None:
        configured = config()
        configured["topology"] = {
            "profile": "a3_dp4_tp8",
            "profiles_file": "workflow/continuous/topology_profiles.yaml",
        }
        configured["baseline"].update(
            num_speculative_tokens=4,
            flashcomm1=True,
            cudagraph_capture_sizes=[40, 80],
            max_cudagraph_capture_size=80,
        )
        configured["search_limits"].update(
            num_speculative_tokens=[4],
            flashcomm1=[True, False],
            cudagraph_capture_sizes=[
                [5, 10, 20, 60],
                [5, 10, 20, 40, 60, 80],
                [40, 80],
            ],
            max_cudagraph_capture_size=[60, 80],
        )
        controller = tuning.Controller(configured)
        mixed = dict(controller.config["baseline"])
        mixed["cudagraph_capture_sizes"] = [5, 10, 20, 40, 60, 80]
        controller.validate_candidate_invariants(mixed)
        invalid = dict(controller.config["baseline"])
        invalid["cudagraph_capture_sizes"] = [5, 10, 20, 60]
        invalid["max_cudagraph_capture_size"] = 60
        with self.assertRaisesRegex(ValueError, "Effective sequence-parallel"):
            controller.validate_candidate_invariants(invalid)
        controller.validate_candidate_invariants(dict(controller.config["baseline"]))

    def test_full_decode_only_allows_runtime_filtered_unreachable_graph_maximum(self) -> None:
        configured = config()
        configured["baseline"].update(
            max_num_seqs=32,
            max_num_batched_tokens=4096,
            num_speculative_tokens=3,
            async_scheduling=True,
            compilation_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[32, 64, 128],
            max_cudagraph_capture_size=128,
        )
        configured["search_limits"].update(
            max_num_seqs=[32, 64],
            cudagraph_capture_sizes=[[32, 64, 128], [32, 64, 128, 256]],
            max_cudagraph_capture_size=[128, 256],
        )
        controller = tuning.Controller(configured)
        candidate = dict(controller.config["baseline"])
        candidate.update(
            cudagraph_capture_sizes=[32, 64, 128, 256],
            max_cudagraph_capture_size=256,
        )
        controller.validate_candidate_invariants(candidate)

    def test_task_queue_two_requires_explicit_graph_disabled_diagnostic(self) -> None:
        configured = config()
        configured["baseline"].update(
            TASK_QUEUE_ENABLE=1,
            cudagraph_capture_sizes=[16, 32, 48, 64],
            max_cudagraph_capture_size=64,
        )
        configured["search_limits"].update(
            compilation_mode=["FULL_DECODE_ONLY", "NONE"],
            TASK_QUEUE_ENABLE=[0, 1, 2],
            cudagraph_capture_sizes=[[16, 32, 48, 64], None],
            max_cudagraph_capture_size=[64],
        )
        configured["automatic_registry_validation"] = {
            "scenario": {},
            "compatibility_policy": {"schema_version": 1},
            "active_injections": {
                "TASK_QUEUE_ENABLE": {
                    "kind": "env_value",
                    "name": "TASK_QUEUE_ENABLE",
                }
            }
        }
        controller = tuning.Controller(configured)
        controller.automatic_registry_validation = None
        controller.automatic_compatibility = None
        candidate = dict(controller.config["baseline"])
        candidate["TASK_QUEUE_ENABLE"] = 2
        with self.assertRaisesRegex(ValueError, "graph mode NONE"):
            controller.validate_candidate_invariants(candidate)
        candidate["compilation_mode"] = "NONE"
        candidate["cudagraph_capture_sizes"] = None
        controller.validate_candidate_invariants(candidate)

    def test_infrastructure_failure_does_not_consume_candidate_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary)
            failed = session / "round_001_a1"
            (failed / "02_parameters").mkdir(parents=True)
            (failed / "05_results").mkdir()
            (failed / "06_agent_analysis").mkdir()
            tuning.save_yaml(
                failed / "02_parameters" / "candidate_params.yaml", self.baseline
            )
            tuning.save_yaml(
                failed / "05_results" / "failure.yaml", {"reason": "stale master ip"}
            )
            tuning.save_json(
                failed / "06_agent_analysis" / "failure_decision.json",
                {
                    "classification": "transient_infrastructure",
                    "action": "retry_same",
                },
            )
            attempts = self.controller.attempted_history_summary(session)
            self.assertFalse(attempts[0]["counts_as_parameter_experiment"])
            self.assertFalse(
                self.controller.candidate_was_attempted(session, self.baseline)
            )

    def test_parameter_attributed_startup_failure_consumes_candidate_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary)
            failed = session / "round_001_a1"
            (failed / "02_parameters").mkdir(parents=True)
            (failed / "05_results").mkdir()
            (failed / "06_agent_analysis").mkdir()
            tuning.save_yaml(
                failed / "02_parameters" / "candidate_params.yaml", self.baseline
            )
            tuning.save_yaml(
                failed / "05_results" / "failure.yaml",
                {"reason": "parameter-exclusive runtime path crashed"},
            )
            tuning.save_json(
                failed / "06_agent_analysis" / "failure_decision.json",
                {
                    "classification": "model_or_runtime_bug",
                    "action": "adjust_parameters",
                },
            )
            attempts = self.controller.attempted_history_summary(session)
            self.assertTrue(attempts[0]["counts_as_parameter_experiment"])
            self.assertTrue(
                self.controller.candidate_was_attempted(session, self.baseline)
            )

    def test_audited_reclassification_can_restore_unmeasured_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary)
            failed = session / "round_001_a1"
            (failed / "02_parameters").mkdir(parents=True)
            (failed / "05_results").mkdir()
            (failed / "06_agent_analysis").mkdir()
            tuning.save_yaml(
                failed / "02_parameters" / "candidate_params.yaml", self.baseline
            )
            tuning.save_yaml(
                failed / "05_results" / "failure.yaml", {"reason": "startup failed"}
            )
            tuning.save_json(
                failed / "06_agent_analysis" / "failure_decision.json",
                {
                    "classification": "model_or_runtime_bug",
                    "action": "adjust_parameters",
                },
            )
            tuning.save_json(
                failed / "06_agent_analysis" / "failure_reclassification.json",
                {
                    "classification": "transient_infrastructure",
                    "counts_as_parameter_experiment": False,
                    "evidence": ["worker used a stale master address"],
                },
            )
            attempts = self.controller.attempted_history_summary(session)
            self.assertFalse(attempts[0]["counts_as_parameter_experiment"])
            self.assertFalse(
                self.controller.candidate_was_attempted(session, self.baseline)
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

        generic_policy_controller = tuning.Controller(config())
        self.assertEqual(
            "best_anchor_coverage_v2",
            generic_policy_controller.strategy_profile_name,
        )
        self.assertNotIn(
            "hierarchical_probe",
            generic_policy_controller.effective_change_policy(),
        )

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

    def test_agentic_cross_layer_stage_has_no_controller_selected_layer(self) -> None:
        configured = config()
        configured["strategy"] = {
            "profile": "hierarchical_agentic_topology_v2",
            "profiles_file": "workflow/continuous/strategy_profiles.yaml",
        }
        controller = tuning.Controller(configured)
        history = [
            {
                "round": f"round_{index:03d}",
                "metrics": {"benchmark_mode": "aligned_l1"},
            }
            for index in range(8)
        ]
        policy = controller.effective_change_policy(history, history)
        self.assertEqual("cross_layer_refinement", policy["hierarchical_stage"])
        self.assertEqual("agent", policy["cross_layer_selection_owner"])
        self.assertIsNone(policy["controller_preselected_layer"])
        self.assertNotIn("cross_layer_revisit", policy)

    def test_guided_v4_selects_only_layer_and_agent_owns_experiment(self) -> None:
        configured = config()
        configured["strategy"] = {
            "profile": "hierarchical_agentic_guided_v4",
            "profiles_file": "workflow/continuous/strategy_profiles.yaml",
        }
        controller = tuning.Controller(configured)
        baseline_history = [{"round": "b0", "metrics": {"benchmark_mode": "aligned_l1"}}]
        policy = controller.effective_change_policy(
            baseline_history, baseline_history
        )
        probe = policy["hierarchical_probe"]
        self.assertEqual("context_capacity_geometry", probe["name"])
        self.assertEqual("agent", probe["parameter_selection_owner"])
        self.assertNotIn("interaction_plan", probe)
        self.assertFalse(
            controller.strategy_profile["hierarchy"]["coupling_hints_are_binding"]
        )
        self.assertEqual([2, 4], policy["preferred_parameters_per_round"])
        self.assertEqual(1, policy["minimum_parameters_per_round"])
        self.assertEqual(4, policy["max_parameters_per_round"])

        # One low/unknown-value measurement per layer is enough for compact
        # coverage; after all six, the Controller supplies evidence but no layer.
        history = [
            {
                "round": f"round_{index:03d}",
                "metrics": {"benchmark_mode": "aligned_l1"},
            }
            for index in range(7)
        ]
        cross_layer = controller.effective_change_policy(history, history)
        self.assertEqual(
            "cross_layer_refinement", cross_layer["hierarchical_stage"]
        )
        self.assertEqual("agent", cross_layer["cross_layer_selection_owner"])
        self.assertIsNone(cross_layer["controller_preselected_layer"])
        self.assertEqual(
            {"exploitation": 0, "cross_layer_interaction": 0, "frontier_novelty": 0},
            cross_layer["measured_exploration_budget_state"]["counts"],
        )

    def test_guided_v5_completion_gate_requires_layers_and_cross_layer_evidence(self) -> None:
        configured = config()
        configured["strategy"] = {
            "profile": "hierarchical_agentic_guided_v5",
            "profiles_file": "workflow/continuous/strategy_profiles.yaml",
        }
        controller = tuning.Controller(configured)
        baseline = {"round": "a0", "metrics": {"benchmark_mode": "aligned_l1"}}
        ordered_history = [baseline]
        for index in range(1, 7):
            ordered_history.append(
                {
                    "round": f"a{index}",
                    "metrics": {"benchmark_mode": "aligned_l1"},
                }
            )
        with patch.object(controller, "primary_performance_score", return_value=None):
            ordered_policy = controller.effective_change_policy(
                ordered_history, ordered_history
            )
            self.assertEqual(
                "cross_layer_refinement", ordered_policy["hierarchical_stage"]
            )
            self.assertFalse(ordered_policy["completion_gate"]["allowed"])
            with self.assertRaisesRegex(ValueError, "cross-layer evidence floor"):
                controller.validate_stop_complete_allowed(ordered_policy)

            complete_history = [
                *ordered_history,
                *[
                    {
                        "round": f"x{index}",
                        "metrics": {"benchmark_mode": "aligned_l1"},
                    }
                    for index in range(1, 5)
                ],
            ]
            complete_policy = controller.effective_change_policy(
                complete_history, complete_history
            )
        self.assertTrue(complete_policy["completion_gate"]["allowed"])
        controller.validate_stop_complete_allowed(complete_policy)

    def test_ordered_probe_rejects_intent_label_with_only_cross_layer_changes(self) -> None:
        policy = {
            "hierarchical_stage": "ordered_probe",
            "hierarchical_probe": {
                "name": "moe_routing_and_balance",
                "parameters": [
                    "enable_expert_parallel",
                    "enable_eplb",
                    "eplb_num_redundant_experts",
                    "enable_balance_scheduling",
                ],
            },
        }
        with self.assertRaisesRegex(ValueError, "skip_layer is disabled"):
            self.controller.validate_ordered_probe_parameter_alignment(
                ["cudagraph_capture_sizes"], policy
            )
        self.controller.validate_ordered_probe_parameter_alignment(
            ["enable_balance_scheduling", "gpu_memory_utilization"], policy
        )

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
        self.assertEqual("aligned_fast_c32_v2", controller.benchmark_profile_name)
        self.assertEqual("aligned_l1", controller.benchmark_mode)
        frozen = controller.config
        frozen["benchmark"]["profiles_file"] = "missing-after-session.yaml"
        self.assertEqual(
            "aligned_fast_c32_v2",
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
            controller.write_decision_schemas(session)
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

    def test_conditional_failure_rejects_exact_combination_only(self) -> None:
        configured = config()
        failed = dict(self.baseline, max_num_seqs=64)
        configured["conditional_search_exclusions"] = [
            {
                "trial_id": "round_oom",
                "failure_classification": "parameter_oom",
                "conditions": failed,
                "attributed_parameters": ["max_num_seqs"],
            }
        ]
        controller = tuning.Controller(configured)
        with self.assertRaisesRegex(ValueError, "exactly matches"):
            controller.validate_candidate(
                self.baseline,
                failed,
                [
                    {
                        "parameter": "max_num_seqs",
                        "before": 48,
                        "after": 64,
                        "rationale": "Repeat the complete failed capacity combination.",
                    }
                ],
            )
        alternative = dict(failed, gpu_memory_utilization=0.9)
        controller.validate_candidate(
            self.baseline,
            alternative,
            [
                {
                    "parameter": "max_num_seqs",
                    "before": 48,
                    "after": 64,
                    "rationale": "Retest sequence capacity with different memory headroom.",
                },
                {
                    "parameter": "gpu_memory_utilization",
                    "before": 0.93,
                    "after": 0.9,
                    "rationale": "Change the companion value so this is a new combination.",
                },
            ],
        )

    def test_frontier_budget_is_measured_without_controller_preselection(self) -> None:
        configured = config()
        configured["strategy"]["profile"] = "hierarchical_agentic_frontier_v3"
        controller = tuning.Controller(configured)
        history = [
            {"round": "r1", "decision": {"exploration_intent": "exploitation"}},
            {"round": "r2", "decision": {"exploration_intent": "exploitation"}},
            {
                "round": "r3",
                "decision": {"exploration_intent": "cross_layer_interaction"},
            },
        ]
        attempted = [
            *history,
            {"round": "r4", "decision": {"exploration_intent": "frontier_novelty"}},
            {
                "round": "r5",
                "decision": {"exploration_intent": "diagnostic_ablation"},
            },
        ]
        state = controller.measured_exploration_budget_state(
            history,
            attempted,
            {
                "programmatic_tracking": True,
                "agent_final_choice": True,
                "exploitation_fraction": 0.65,
                "cross_layer_interaction_fraction": 0.25,
                "frontier_novelty_fraction": 0.10,
            },
        )
        self.assertEqual(
            {"exploitation": 2, "cross_layer_interaction": 1, "frontier_novelty": 1},
            state["counts"],
        )
        self.assertEqual(["exploitation"], state["underrepresented_intents"])
        self.assertIsNone(state["controller_preselected_intent"])
        self.assertEqual(1, state["diagnostic_ablation_count"])
        self.assertEqual(0, state["diagnostic_ablation_remaining"])

    def test_frontier_mtp_off_requires_single_diagnostic_ablation(self) -> None:
        configured = config()
        configured["strategy"]["profile"] = "hierarchical_agentic_frontier_v3"
        configured["search_limits"]["num_speculative_tokens"] = [0, 3]
        controller = tuning.Controller(configured)
        candidate = dict(self.baseline, num_speculative_tokens=0)
        changes = [
            {
                "parameter": "num_speculative_tokens",
                "before": 3,
                "after": 0,
                "rationale": "One bounded MTP-off diagnostic isolates draft overhead.",
            }
        ]
        base_decision = {
            "change_strategy": "single_parameter",
            "knowledge_evidence": [
                "Measured decode behavior motivates one bounded diagnostic ablation."
            ],
            "interaction_analysis": [],
            "constraint_checks": [
                "A zero speculative depth disables MTP without changing topology."
            ],
        }
        policy = controller.effective_change_policy()
        policy.update(
            minimum_parameters_per_round=1,
            max_parameters_per_round=1,
            max_grid_steps_per_parameter=1,
            max_total_grid_steps=1,
        )
        with self.assertRaisesRegex(ValueError, "diagnostic_ablation"):
            controller.validate_candidate(
                self.baseline,
                candidate,
                changes,
                {**base_decision, "exploration_intent": "exploitation"},
                policy,
            )
        controller.validate_candidate(
            self.baseline,
            candidate,
            changes,
            {**base_decision, "exploration_intent": "diagnostic_ablation"},
            policy,
        )
        exhausted = copy.deepcopy(policy)
        exhausted["measured_exploration_budget_state"][
            "diagnostic_ablation_count"
        ] = 1
        with self.assertRaisesRegex(ValueError, "budget is exhausted"):
            controller.validate_candidate(
                self.baseline,
                candidate,
                changes,
                {**base_decision, "exploration_intent": "diagnostic_ablation"},
                exhausted,
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
        self.assertIn("LAUNCH_PROFILE=explicit_candidate", env_text)
        self.assertIn("BENCHMARK_MODE=aligned_l1", env_text)
        self.assertIn("BENCHMARK_REPETITIONS=1", env_text)
        self.assertIn("BENCHMARK_SUITE='01_调优_快速筛选-v2.yaml'", env_text)
        self.assertIn("BENCHMARK_EXPECTED_FORMAL_CASES=4", env_text)
        self.assertNotIn("BENCHMARK_TIME_BUDGET_SECONDS", env_text)
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
        self.assertIn('MASTER_ENDPOINT_FILE="${RUN_DIR}/master_endpoint.env"', runtime)
        self.assertIn('MASTER_IP="${NODE_IP}"', runtime)
        self.assertIn('MASTER_ENDPOINT_RESOLVED role=${VLLMTKB_ROLE}', runtime)
        self.assertIn('prefetch_checkpoints_on_node()', runtime)
        self.assertIn(
            'RUNTIME_INSTANCE_ID="${NODE_IP//[^A-Za-z0-9_.-]/_}"', runtime
        )
        self.assertIn(
            'generated_json_configs.${RUNTIME_INSTANCE_ID}.json', runtime
        )
        self.assertNotIn(
            'GENERATED_JSON_CONFIGS="${RUN_DIR}/generated_json_configs.json"',
            runtime,
        )
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
        self.assertIn('export VLLMTKB_ROLE=master', master_runtime)
        self.assertIn('export VLLMTKB_ROLE=worker', worker_runtime)
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
        # This profile belongs to the independent W4A8C8 runtime; remove the
        # W8A8 runtime allowlist so this test can isolate executor rendering.
        configured.pop("runtime", None)
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
        configured = production_b0_config()
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
            self.assertEqual(
                [64000, 16384, 32768, 49152],
                resolved["search_limits"]["max_model_len"],
            )
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
            self.assertNotIn(
                "disable_padded_drafter_batch",
                payload["json_configs"]["speculative_config"],
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
            "automatic_registry_a8_frontier_v4",
            resolved["resolved_search_space"]["profile"],
        )
        self.assertEqual(
            "automatic_registry", resolved["resolved_search_space"]["mode"]
        )
        self.assertEqual(142, result["summary"]["registry_parameters"])
        self.assertEqual(30, result["summary"]["active_parameters"])
        self.assertEqual(73, result["summary"]["reserve_parameters"])
        self.assertEqual(39, result["summary"]["fixed_parameters"])
        self.assertEqual(0, result["summary"]["rejected_parameters"])
        self.assertEqual(
            [], resolved["resolved_search_space"]["derived_runtime_parameters"]
        )
        self.assertEqual(
            resolved["search_limits"], result["integration"]["effective_search_limits"]
        )
        for name in {
            "max_model_len",
            "num_speculative_tokens",
            "async_scheduling",
            "fused_mc2",
            "speculative_config__attention_backend",
            "additional_config__ascend_compilation_config__fuse_norm_quant",
            "compilation_enable_sp",
            "speculative_config__disable_padded_drafter_batch",
        }:
            self.assertIn(name, result["active_search_limits"])
        self.assertNotIn("VLLM_RAY_PER_WORKER_GPUS", result["active_search_limits"])

    def test_frontier_v4_keeps_coupled_mtp_graph_and_prefill_values(self) -> None:
        raw = tuning.load_config(
            tuning.HERE / "server_autonomous" / "config.dp4_tp8.search_v4.yaml"
        )
        raw, _ = tuning.resolve_runtime_profile(raw, tuning.KB_ROOT)
        raw = tuning.apply_topology_baseline_binding(raw)
        raw = tuning.resolve_initial_baseline_definition(raw, tuning.KB_ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            history_seed = Path(temporary) / "history.json"
            history_seed.write_text("[]\n", encoding="utf-8")
            raw["search_space"]["history_source"] = "explicit"
            raw["search_space"]["history_path"] = str(history_seed)
            resolved, result = resolve_search_limits(
                raw,
                project_root=tuning.KB_ROOT,
                archive_root=Path(temporary),
            )
        self.assertEqual(30, result["summary"]["active_parameters"])
        self.assertEqual(
            {0, 1, 2, 3, 4},
            set(resolved["search_limits"]["num_speculative_tokens"]),
        )
        self.assertEqual(
            {0, 512, 1024, 2048, 4096, 8192},
            set(resolved["search_limits"]["long_prefill_token_threshold"]),
        )
        self.assertEqual(
            {0, 1, 2}, set(resolved["search_limits"]["TASK_QUEUE_ENABLE"])
        )
        self.assertIn(
            [3, 6, 12, 24, 48, 72, 96],
            resolved["search_limits"]["cudagraph_capture_sizes"],
        )
        self.assertIn(
            [4, 8, 12, 16, 24, 32, 48, 56, 64, 72, 84, 96, 108, 112, 128],
            resolved["search_limits"]["cudagraph_capture_sizes"],
        )
        self.assertIn(
            [5, 10, 15, 20, 30, 40, 60, 70, 80, 90, 105, 120, 135, 140, 160],
            resolved["search_limits"]["cudagraph_capture_sizes"],
        )
        self.assertIn(
            "max_cudagraph_capture_size", result["active_search_limits"]
        )
        self.assertEqual(
            1,
            tuning.Controller.grid_step_distance(
                [[16, 32, 48, 64], [3, 6, 12, 24, 48, 72, 96]],
                [16, 32, 48, 64],
                [3, 6, 12, 24, 48, 72, 96],
            ),
        )

    def test_decode_only_profile_focuses_budget_and_separates_eplb_surfaces(
        self,
    ) -> None:
        raw = tuning.load_config(
            tuning.HERE / "server_autonomous" / "config.dp4_tp8.search_v4.yaml"
        )
        raw, _ = tuning.resolve_runtime_profile(raw, tuning.KB_ROOT)
        raw = tuning.apply_topology_baseline_binding(raw)
        raw["search_space"]["profile"] = "automatic_registry_decode_only_v1"
        raw["initial_baseline"]["definition"] = (
            "workflow/baselines/expert_decode_glm52_w8a8_dp4_tp8_v1.yaml"
        )
        raw = tuning.resolve_initial_baseline_definition(raw, tuning.KB_ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            history_seed = Path(temporary) / "history.json"
            history_seed.write_text("[]\n", encoding="utf-8")
            raw["search_space"]["history_source"] = "explicit"
            raw["search_space"]["history_path"] = str(history_seed)
            resolved, result = resolve_search_limits(
                raw,
                project_root=tuning.KB_ROOT,
                archive_root=Path(temporary),
            )

        active = set(result["active_search_limits"])
        self.assertEqual(14, result["summary"]["active_parameters"])
        self.assertEqual(
            {True, False},
            set(resolved["search_limits"]["enable_chunked_prefill"]),
        )
        self.assertNotIn("enable_eplb", resolved["search_limits"])
        self.assertNotIn("eplb_num_redundant_experts", resolved["search_limits"])
        self.assertNotIn("enable_eplb", resolved["baseline"])
        self.assertNotIn("eplb_num_redundant_experts", resolved["baseline"])
        self.assertEqual(
            {
                "enable_eplb": False,
                "eplb_num_redundant_experts": 0,
                "mlapo": True,
                "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": None,
                "additional_config__ascend_compilation_config__fuse_norm_quant": True,
            },
            result["integration"]["implicit_source_defaults"],
        )
        self.assertNotIn("enable_eplb", active)
        self.assertNotIn("eplb_num_redundant_experts", active)
        self.assertIn("max_model_len", active)
        self.assertIn(2304, resolved["search_limits"]["max_model_len"])
        self.assertIn("cudagraph_capture_sizes", active)
        self.assertNotIn("async_scheduling", active)
        self.assertNotIn("enable_expert_parallel", active)
        self.assertNotIn("mlapo", resolved["search_limits"])
        self.assertNotIn("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", resolved["search_limits"])
        self.assertNotIn(
            "additional_config__ascend_compilation_config__fuse_norm_quant",
            resolved["search_limits"],
        )
        self.assertEqual([False], resolved["search_limits"]["compilation_enable_sp"])

    def test_decode_priority_package_partitions_all_priority_axes(
        self,
    ) -> None:
        raw = tuning.load_config(
            tuning.HERE
            / "server_autonomous"
            / "config.dp4_tp8.decode_priority_v1.yaml"
        )
        raw, runtime = tuning.resolve_runtime_profile(raw, tuning.KB_ROOT)
        raw = tuning.apply_topology_baseline_binding(raw)
        raw = tuning.resolve_initial_baseline_definition(raw, tuning.KB_ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            resolved, result = resolve_search_limits(
                raw,
                project_root=tuning.KB_ROOT,
                archive_root=Path(temporary),
            )
        resolved["remote_transport"] = "paramiko"
        resolved["operation_mode"] = "windows_remote"
        resolved["remote_host"] = "hetao-npu"
        controller = tuning.Controller(resolved)
        controller.validate_candidate_invariants(dict(resolved["baseline"]))
        priority = controller.strategy_profile["priority_search"]
        primary = set(priority["primary_parameters"])
        conditional = set(priority["conditional_parameters"])
        secondary = set(priority["secondary_parameters"])
        active = set(result["active_search_limits"])
        self.assertFalse(primary & secondary)
        self.assertFalse(primary & conditional)
        self.assertFalse(conditional & secondary)
        self.assertEqual(10, len(primary))
        self.assertEqual(4, len(conditional))
        self.assertEqual(11, len(secondary))
        self.assertEqual(25, len(active))
        self.assertEqual(active, primary | conditional | secondary)
        probes = controller.strategy_profile["hierarchy"]["ordered_probes"]
        self.assertEqual(
            [
                "list2_capacity_geometry",
                "list2_mtp_graph_joint",
                "list2_scheduler_capacity_refinement",
                "list1_3_conditional_switches",
            ],
            [probe["name"] for probe in probes],
        )
        self.assertEqual(
            8,
            controller.strategy_profile["completion_gate"][
                "minimum_cross_layer_successful_measurements"
            ],
        )
        self.assertEqual(
            "glm52_w8a8_a3_dp4_tp8_decode_priority_v1", runtime["profile"]
        )
        self.assertEqual("decode_only_c32_v1", controller.benchmark_profile_name)
        self.assertEqual(
            {"decode-256-2048"},
            set(controller.benchmark["aligned_l1"]["workloads"]),
        )
        self.assertEqual("FULL_DECODE_ONLY", resolved["baseline"]["compilation_mode"])
        self.assertEqual(3, resolved["baseline"]["num_speculative_tokens"])
        self.assertTrue(resolved["baseline"]["flashcomm1"])
        self.assertEqual(2, resolved["baseline"]["fused_mc2"])

    def test_decode_priority_reserves_list1_but_recovery_bypasses_quota(self) -> None:
        state = {
            "secondary_parameters": ["flashcomm1", "fused_mc2"],
            "secondary_successful_measurements_remaining": 2,
            "failure_recovery_bypasses_phase_priority": True,
        }
        early = {
            "hierarchical_stage": "ordered_probe",
            "hierarchical_probe": {"name": "list2_capacity_geometry"},
            "priority_search_budget_state": state,
        }
        with self.assertRaisesRegex(ValueError, "reserved for autonomous"):
            tuning.Controller.validate_priority_search_alignment(
                ["flashcomm1"], early
            )
        recovery = {**early, "failure_recovery_mode": True}
        tuning.Controller.validate_priority_search_alignment(
            ["flashcomm1"], recovery
        )
        cross_layer = {
            "hierarchical_stage": "cross_layer_refinement",
            "priority_search_budget_state": state,
        }
        tuning.Controller.validate_priority_search_alignment(
            ["flashcomm1"], cross_layer
        )
        exhausted = {
            "hierarchical_stage": "cross_layer_refinement",
            "priority_search_budget_state": {
                **state,
                "secondary_successful_measurements_remaining": 0,
            },
        }
        with self.assertRaisesRegex(ValueError, "quota is exhausted"):
            tuning.Controller.validate_priority_search_alignment(
                ["fused_mc2"], exhausted
            )

    def test_decode_priority_counts_only_successful_secondary_measurements(self) -> None:
        configured = config()
        configured["strategy"]["profile"] = "decode_priority_agentic_v1"
        controller = tuning.Controller(configured)
        history = [
            {"round": "a0"},
            {
                "round": "a1",
                "decision": {"changes": [{"parameter": "flashcomm1"}]},
            },
            {
                "round": "a2",
                "decision": {"changes": [{"parameter": "max_num_seqs"}]},
            },
        ]
        state = controller.priority_search_budget_state(history)
        self.assertEqual(1, state["secondary_successful_measurements_used"])
        self.assertEqual(3, state["secondary_successful_measurements_remaining"])

    def test_decode_only_benchmark_asset_matches_frozen_definition(self) -> None:
        configured = tuning.load_yaml(tuning.HERE / "config.yaml")
        definition = configured["benchmark"]["aligned_l1_decode_only_v1"]
        suite_path = (
            tuning.HERE
            / "benchmark_assets"
            / "decode-only-c32-v1"
            / "spec"
            / "suites"
            / definition["suite"]
        )
        suite = tuning.load_yaml(suite_path)
        self.assertEqual(
            definition["suite_sha256"],
            hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(1, definition["expected_formal_cases"])
        self.assertEqual(
            {"decode-256-2048"},
            {item["标识"] for item in suite["工作负载"]},
        )
        self.assertEqual(
            ["decode-256-2048"], suite["阶段"][0]["工作负载"]
        )

    def test_frozen_continuation_history_is_visible_to_agent_and_dedupe(self) -> None:
        controller = tuning.Controller(config())
        imported = {
            "round": "round_016_a13",
            "params": dict(controller.config["baseline"]),
            "outcome": "success",
            "counts_as_parameter_experiment": True,
            "experiment_evidence_status": "benchmarked",
            "metrics": {"metrics": {"output_token_throughput": 591.1866}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary)
            frozen = session / "00_search_space" / "continuation_history.json"
            frozen.parent.mkdir(parents=True)
            frozen.write_text(
                json.dumps([imported], ensure_ascii=False), encoding="utf-8"
            )
            history = controller.attempted_history_summary(session)
        self.assertEqual([imported], history)

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

        legal_but_inactive_fused = {**candidate, "enable_expert_parallel": False}
        controller.validate_candidate_invariants(legal_but_inactive_fused)
        legal_synchronous_draft = {
            **candidate,
            "num_speculative_tokens": 0,
            "async_scheduling": False,
        }
        controller.validate_candidate_invariants(legal_synchronous_draft)

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
        # The curated registry carries separate fixed surfaces for upstream
        # EPLB and Ascend-native dynamic EPLB; they must not collapse by leaf
        # name even though neither is Active in this scenario.
        self.assertEqual(28, len(registry["parameters"]))
        self.assertEqual(22, len(automatic_active))
        self.assertEqual(81, automatic_result["summary"]["reserve_parameters"])
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
        raw = production_b0_config()
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
                [256, 32, 64, 128, 192, 384, 512],
                controller.config["search_limits"]["max_num_seqs"],
            )
            self.assertEqual(
                [2048, 1024, 4096, 8192, 16384, 32768],
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
                    "action": "adjust_parameters",
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
            tuning.save_json(
                failed / "06_agent_analysis" / "failure_decision.json",
                {
                    "classification": "parameter_invalid",
                    "action": "adjust_parameters",
                    "candidate": self.baseline,
                },
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

    def test_failure_recovery_allows_precise_single_parameter_repair(self) -> None:
        configured = config()
        configured["search_limits"]["mlapo"] = [True, False]
        configured["baseline"]["mlapo"] = True
        controller = tuning.Controller(configured)
        current = {**self.baseline, "mlapo": True}
        candidate = {**current, "mlapo": False}
        decision = {
            "summary": "Disable the directly implicated MLAPO path",
            "classification": "model_or_runtime_bug",
            "root_cause": "MLAPO fused weight processing timed out",
            "evidence": ["aclnnMuls vector core timeout in fused MLAPO"],
            "action": "adjust_parameters",
            "safe_to_automate": True,
            "change_strategy": "single_parameter",
            "interaction_analysis": [],
            "constraint_checks": ["mlapo=false is inside the frozen search grid"],
            "changes": [
                {
                    "parameter": "mlapo",
                    "before": True,
                    "after": False,
                    "rationale": "Bypass the exact fused MLAPO path that timed out.",
                }
            ],
            "recovery_changes": [],
            "candidate": candidate,
        }
        exploration_policy = {
            "phase": "exploration",
            "minimum_parameters_per_round": 2,
            "max_parameters_per_round": 4,
            "max_grid_steps_per_parameter": 2,
            "max_total_grid_steps": 4,
            "derived_parameters": {},
        }
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            controller, "effective_change_policy", return_value=exploration_policy
        ):
            controller.validate_failure_decision(
                Path(temporary), decision, current
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
                        "unknown", "diagnostic_retry_same", True
                    ),
                    current,
                )
            )
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

    def test_fused_moe_failure_uses_recovery_registry_outside_active_search(self) -> None:
        configured = config()
        configured["failure_recovery"] = {
            "recovery_parameters": {
                "additional_config__multistream_overlap_shared_expert": {
                    "initial_value": True,
                    "allowed_values": [False, True],
                    "injection": {
                        "kind": "json_path",
                        "path": [
                            "additional_config",
                            "multistream_overlap_shared_expert",
                        ],
                    },
                }
            }
        }
        controller = tuning.Controller(configured)
        self.assertNotIn(
            "additional_config__multistream_overlap_shared_expert",
            controller.candidate_schema,
        )
        with tempfile.TemporaryDirectory() as temporary:
            round_dir = Path(temporary)
            (round_dir / "04_runtime").mkdir(parents=True)
            (round_dir / "05_results").mkdir()
            (round_dir / "04_runtime" / "worker.log").write_text(
                "ValueError: FusedMoE shared experts split computation does not "
                "match the integrated computation.\n",
                encoding="utf-8",
            )
            decision = controller.deterministic_fused_moe_shared_expert_recovery(
                round_dir, self.baseline
            )
            self.assertIsNotNone(decision)
            assert decision is not None
            self.assertEqual("adjust_parameters", decision["action"])
            self.assertEqual([], decision["changes"])
            self.assertEqual(False, decision["recovery_changes"][0]["after"])
            controller.validate_failure_decision(
                round_dir, decision, self.baseline
            )
            updated = controller.validate_recovery_changes(
                decision["recovery_changes"]
            )
            controller.runtime_recovery_values = updated
            env_text = controller.candidate_env("a0f1", self.baseline)
            encoded = next(
                line.split("=", 1)[1]
                for line in env_text.splitlines()
                if line.startswith("RUNTIME_INJECTION_PAYLOAD_B64=")
            )
            payload = json.loads(base64.b64decode(shlex.split(encoded)[0]))
            self.assertFalse(
                payload["json_configs"]["additional_config"][
                    "multistream_overlap_shared_expert"
                ]
            )

    def test_shared_expert_overlap_is_not_hardcoded_in_remote_launcher(self) -> None:
        script = (tuning.HERE / "remote" / "common_runtime_loop.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('"multistream_overlap_shared_expert": True', script)

    def test_mlapo_vector_timeout_preempts_cascading_gloo_retry(self) -> None:
        configured = config()
        configured["search_limits"]["mlapo"] = [True, False]
        configured["baseline"]["mlapo"] = True
        controller = tuning.Controller(configured)
        current = {**self.baseline, "mlapo": True}
        with tempfile.TemporaryDirectory() as temporary:
            round_dir = Path(temporary)
            (round_dir / "04_runtime").mkdir(parents=True)
            (round_dir / "05_results").mkdir()
            (round_dir / "04_runtime" / "worker.log").write_text(
                "File vllm_ascend/attention/sfa_v1.py, in "
                "_process_weights_for_fused_mlapo\n"
                "RuntimeError: current operator aclnnMuls\n"
                "Vector core execution timed out\n"
                "RuntimeError: gloo Connection closed by peer\n",
                encoding="utf-8",
            )
            decision = controller.deterministic_mlapo_vector_timeout_recovery(
                round_dir, current
            )
            self.assertIsNotNone(decision)
            assert decision is not None
            self.assertEqual("adjust_parameters", decision["action"])
            self.assertEqual("model_or_runtime_bug", decision["classification"])
            self.assertEqual("mlapo", decision["changes"][0]["parameter"])
            self.assertFalse(decision["candidate"]["mlapo"])
            self.assertEqual([], decision["recovery_changes"])
            controller.validate_failure_decision(round_dir, decision, current)

    def test_generic_vector_timeout_does_not_disable_mlapo(self) -> None:
        configured = config()
        configured["search_limits"]["mlapo"] = [True, False]
        configured["baseline"]["mlapo"] = True
        controller = tuning.Controller(configured)
        current = {**self.baseline, "mlapo": True}
        with tempfile.TemporaryDirectory() as temporary:
            round_dir = Path(temporary)
            (round_dir / "04_runtime").mkdir(parents=True)
            (round_dir / "04_runtime" / "worker.log").write_text(
                "RuntimeError: current operator aclnnMuls\n"
                "Vector core execution timed out\n",
                encoding="utf-8",
            )
            self.assertIsNone(
                controller.deterministic_mlapo_vector_timeout_recovery(
                    round_dir, current
                )
            )

    def test_historical_mlapo_root_cause_remains_visible_after_later_failure(self) -> None:
        configured = config()
        configured["search_limits"]["mlapo"] = [True, False]
        configured["baseline"]["mlapo"] = True
        controller = tuning.Controller(configured)
        current = {**self.baseline, "mlapo": True}
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary)
            earlier = session / "round_000_a0"
            later = session / "round_001_a0r1"
            (earlier / "04_runtime").mkdir(parents=True)
            (later / "04_runtime").mkdir(parents=True)
            (earlier / "04_runtime" / "worker.log").write_text(
                "attention/sfa_v1.py _process_weights_for_fused_mlapo\n"
                "current operator aclnnMuls\nVector core execution timed out\n",
                encoding="utf-8",
            )
            (later / "04_runtime" / "worker.log").write_text(
                "ERR02200 HcclAllGather EI0006\n", encoding="utf-8"
            )
            decision = controller.pending_historical_deterministic_recovery(
                session, current
            )
            self.assertIsNotNone(decision)
            assert decision is not None
            self.assertFalse(decision["candidate"]["mlapo"])
            self.assertIn("round_000_a0", decision["summary"])

    def test_exact_hccl_communicator_failure_is_retryable_not_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            round_dir = Path(temporary)
            (round_dir / "04_runtime").mkdir(parents=True)
            (round_dir / "05_results").mkdir()
            (round_dir / "04_runtime" / "worker.log").write_text(
                "hcclCommInitRootInfoConfig ERR02200\n"
                "current operator name is HcclAllGather\n"
                "Communication_Error_Get_Socket(EI0006)\n",
                encoding="utf-8",
            )
            decision = self.controller.deterministic_hccl_communicator_retry(
                round_dir, self.baseline
            )
            self.assertIsNotNone(decision)
            assert decision is not None
            self.assertEqual("network_or_hccl", decision["classification"])
            self.assertEqual("retry_same", decision["action"])

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
        self.assertEqual(
            set(
                branches["diagnostic_retry_same"]["properties"]["classification"][
                    "enum"
                ]
            ),
            set(tuning.FAILURE_DIAGNOSTIC_RETRY_CLASSIFICATIONS),
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
        self.assertIn("recovery_changes", schema["required"])

    def test_agent_schemas_match_controller_change_budget(self) -> None:
        configured = config()
        configured["change_policy"]["max_parameters_per_round"] = 4
        controller = tuning.Controller(configured)
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary)
            controller.write_decision_schemas(session)
            for filename in (
                "agent_decision.schema.json",
                "failure_decision.schema.json",
            ):
                schema = json.loads(
                    (session / "00_search_space" / filename).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    controller.max_parameters_per_round,
                    schema["properties"]["changes"]["maxItems"],
                )
            runtime_agent_schema = json.loads(
                controller.agent_schema_path(session).read_text(encoding="utf-8")
            )
            runtime_failure_schema = json.loads(
                controller.failure_schema_path(session).read_text(encoding="utf-8")
            )
            self.assertEqual(
                4, runtime_agent_schema["properties"]["changes"]["maxItems"]
            )
            self.assertEqual(
                4, runtime_failure_schema["properties"]["changes"]["maxItems"]
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

    def test_guidellm_zero_measurement_case_gets_identical_bounded_retry(self) -> None:
        self.controller.benchmark_mode = "aligned_l1"
        with tempfile.TemporaryDirectory() as temporary:
            round_dir = Path(temporary)
            runtime_dir = round_dir / "04_runtime"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "SERVICE_READY").touch()
            (runtime_dir / "benchmark_runner.log").write_text(
                "CASE FAILED balanced/c1-warmup/guidellm.yaml runner_exit=1\n"
                "ValueError: Cannot compile GenerativeMetrics: No measurement "
                "start or end times available.\n",
                encoding="utf-8",
            )
            (runtime_dir / "master.log").write_text(
                "Application startup complete. HTTP 200 OK\n", encoding="utf-8"
            )
            decision = self.controller.deterministic_benchmark_retry(
                round_dir, self.baseline
            )
            self.assertIsNotNone(decision)
            self.assertEqual("benchmark_failure", decision["classification"])
            self.assertEqual("retry_same", decision["action"])
            self.assertEqual(self.baseline, decision["candidate"])

    def test_guidellm_zero_measurement_retry_fails_closed_on_serving_error(self) -> None:
        self.controller.benchmark_mode = "aligned_l1"
        with tempfile.TemporaryDirectory() as temporary:
            round_dir = Path(temporary)
            runtime_dir = round_dir / "04_runtime"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "SERVICE_READY").touch()
            (runtime_dir / "benchmark_runner.log").write_text(
                "Cannot compile GenerativeMetrics: No measurement start or end "
                "times available\n",
                encoding="utf-8",
            )
            (runtime_dir / "worker.log").write_text(
                "HCCL operation failed\n", encoding="utf-8"
            )
            self.assertIsNone(
                self.controller.deterministic_benchmark_retry(
                    round_dir, self.baseline
                )
            )

    def test_unknown_healthy_service_benchmark_failure_gets_bounded_retry(self) -> None:
        self.controller.benchmark_mode = "aligned_l1"
        with tempfile.TemporaryDirectory() as temporary:
            round_dir = Path(temporary)
            runtime_dir = round_dir / "04_runtime"
            runtime_dir.mkdir(parents=True)
            (round_dir / "05_results").mkdir()
            (runtime_dir / "SERVICE_READY").touch()
            (runtime_dir / "benchmark_runner.log").write_text(
                "CASE FAILED new-harness-signature runner_exit=1\n",
                encoding="utf-8",
            )
            (runtime_dir / "master.log").write_text(
                "Application startup complete. HTTP 200 OK\n", encoding="utf-8"
            )
            decision = self.controller.deterministic_healthy_service_benchmark_retry(
                round_dir, self.baseline
            )
            self.assertIsNotNone(decision)
            self.assertEqual("retry_same", decision["action"])
            self.assertEqual("benchmark_failure", decision["classification"])

    def test_generic_benchmark_retry_rejects_dangerous_serving_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            round_dir = Path(temporary)
            runtime_dir = round_dir / "04_runtime"
            runtime_dir.mkdir(parents=True)
            (round_dir / "05_results").mkdir()
            (runtime_dir / "SERVICE_READY").touch()
            (runtime_dir / "benchmark_runner.log").write_text(
                "CASE FAILED request runner_exit=1\n", encoding="utf-8"
            )
            (runtime_dir / "worker.log").write_text(
                "HCCL operation failed\n", encoding="utf-8"
            )
            self.assertIsNone(
                self.controller.deterministic_healthy_service_benchmark_retry(
                    round_dir, self.baseline
                )
            )

    def test_generic_benchmark_retry_rejects_static_dataset_contract_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            round_dir = Path(temporary)
            runtime_dir = round_dir / "04_runtime"
            runtime_dir.mkdir(parents=True)
            (round_dir / "05_results").mkdir()
            (runtime_dir / "SERVICE_READY").touch()
            (runtime_dir / "BENCHMARK_FAILED").touch()
            (runtime_dir / "benchmark_runner.log").write_text(
                "错误：数据集缺少切片：fast-c32-primary/chat-1024-256/c32-warmup\n",
                encoding="utf-8",
            )
            self.assertIsNone(
                self.controller.deterministic_healthy_service_benchmark_retry(
                    round_dir, self.baseline
                )
            )

    def test_inconclusive_pause_uses_remaining_diagnostic_budget(self) -> None:
        failure = {
            "summary": "unclassified runtime exit",
            "root_cause": "insufficient evidence",
            "evidence": ["no dangerous signature"],
            "classification": "unknown",
            "action": "pause_for_human",
            "safe_to_automate": False,
            "change_strategy": "none",
            "interaction_analysis": [],
            "constraint_checks": [],
            "changes": [],
            "candidate": self.baseline,
        }
        selected = self.controller.prefer_bounded_diagnostic_over_pause(
            failure,
            {"current_candidate": self.baseline, "failure_diagnostic_retries": 0},
        )
        self.assertEqual("diagnostic_retry_same", selected["action"])
        self.assertTrue(selected["safe_to_automate"])

    def test_manual_dependency_pause_is_not_auto_overridden(self) -> None:
        failure = {
            "summary": "image digest mismatch requires an approved image",
            "root_cause": "image identity mismatch",
            "evidence": [],
            "classification": "image_or_dependency",
            "action": "pause_for_human",
            "safe_to_automate": False,
            "candidate": self.baseline,
            "changes": [],
        }
        selected = self.controller.prefer_bounded_diagnostic_over_pause(
            failure,
            {"current_candidate": self.baseline, "failure_diagnostic_retries": 0},
        )
        self.assertEqual("pause_for_human", selected["action"])
        intervention = self.controller.hard_terminal_intervention(failure)
        self.assertIsNotNone(intervention)
        assert intervention is not None
        self.assertEqual("image_or_version_identity", intervention["category"])
        self.assertTrue(intervention["operator_steps"])

    def test_negated_image_mismatch_is_not_a_hard_terminal(self) -> None:
        failure = {
            "summary": "HCCL failed; no image identity mismatch is present",
            "root_cause": "socket timeout",
            "evidence": ["image digest is verified and unchanged"],
            "classification": "network_or_hccl",
            "action": "pause_for_human",
            "candidate": self.baseline,
            "changes": [],
        }
        self.assertIsNone(self.controller.hard_terminal_intervention(failure))

    def test_network_pause_uses_remaining_same_candidate_budget(self) -> None:
        failure = {
            "summary": "persistent HCCL socket timeout without a hard block",
            "root_cause": "cross-node communicator did not initialize",
            "evidence": ["ERR02200 followed by EI0006"],
            "classification": "network_or_hccl",
            "action": "pause_for_human",
            "safe_to_automate": False,
            "change_strategy": "none",
            "interaction_analysis": [],
            "constraint_checks": [],
            "changes": [],
            "recovery_changes": [],
            "candidate": self.baseline,
        }
        selected = self.controller.prefer_bounded_diagnostic_over_pause(
            failure,
            {
                "current_candidate": self.baseline,
                "failure_retries": 1,
                "failure_diagnostic_retries": 0,
            },
        )
        self.assertEqual("retry_same", selected["action"])
        self.assertTrue(selected["safe_to_automate"])

    def test_hard_terminal_only_mode_does_not_pause_at_retry_budget(self) -> None:
        configured = config()
        configured["failure_recovery"] = {
            "hard_terminal_only": True,
            "same_candidate_retries": 1,
            "agent_diagnostic_retries": 0,
            "parameter_adjustments": 0,
            "total_recovery_rounds": 0,
        }
        controller = tuning.Controller(configured)
        failure = {
            "summary": "HCCL socket timeout remains diagnostically recoverable",
            "root_cause": "communicator initialization timed out",
            "evidence": ["ERR02200 and EI0006"],
            "classification": "network_or_hccl",
            "action": "pause_for_human",
            "safe_to_automate": False,
            "candidate": self.baseline,
            "changes": [],
        }
        selected = controller.prefer_bounded_diagnostic_over_pause(
            failure,
            {
                "current_candidate": self.baseline,
                "failure_retries": 99,
                "failure_diagnostic_retries": 99,
            },
        )
        self.assertEqual("retry_same", selected["action"])

    def test_control_plane_runtime_errors_are_restart_classified(self) -> None:
        self.assertTrue(
            tuning.controller_exception_is_recoverable(
                RuntimeError("SSH connection timed out while reading remote status")
            )
        )
        self.assertTrue(
            tuning.controller_exception_is_recoverable(
                tuning.RepeatedCandidateRejection("three invalid selections")
            )
        )
        self.assertFalse(
            tuning.controller_exception_is_recoverable(
                RuntimeError("image digest mismatch requires approval")
            )
        )
        self.assertTrue(
            tuning.controller_exception_is_recoverable(
                RuntimeError(
                    "FileNotFoundError: [Errno 2] No such file or directory: "
                    "'/mnt/host-model/workspace/run/request.json.tmp-959' -> "
                    "'/mnt/host-model/workspace/run/request.json'"
                )
            )
        )

    def test_benchmark_shell_disarms_err_trap_for_captured_failures(self) -> None:
        script = (tuning.HERE / "remote" / "run_aligned_l1.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("begin_captured_failure()", script)
        self.assertIn("end_captured_failure()", script)
        # The only raw `set +e` is inside begin_captured_failure itself; every
        # expected failing command site must use the trap-safe helper.
        self.assertEqual(1, len(re.findall(r"^\s*set \+e\s*$", script, re.MULTILINE)))
        self.assertGreaterEqual(script.count("begin_captured_failure"), 6)
        self.assertNotIn("timeout --foreground", script)
        self.assertNotIn("hard budget", script)
        retry_script = (
            tuning.HERE / "remote" / "run_servebench_attempt.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("timeout --foreground", retry_script)

    def test_paused_auto_retry_tracks_the_new_submitted_round(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = root / "session"
            failed_round = session / "round_016_a10"
            failed_round.mkdir(parents=True)
            state_path = root / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "status": "paused_for_human",
                        "session_dir": str(session),
                        "round_index": 16,
                        "round_label": "a10",
                        "candidate_index": 10,
                        "current_candidate": self.baseline,
                        "failure_retries": 0,
                        "total_failure_recovery_rounds": 1,
                    }
                ),
                encoding="utf-8",
            )
            decision = {
                "summary": "healthy-service benchmark retry",
                "classification": "benchmark_failure",
                "root_cause": "benchmark harness ended without metrics",
                "evidence": ["service remained healthy"],
                "action": "retry_same",
                "safe_to_automate": True,
                "change_strategy": "none",
                "interaction_analysis": [],
                "constraint_checks": [],
                "candidate": self.baseline,
                "changes": [],
                "recovery_changes": [],
            }
            with patch.object(tuning, "STATE_FILE", state_path), patch.object(
                self.controller, "assert_state_image_identity"
            ), patch.object(
                self.controller, "load_session_sidecars"
            ), patch.object(
                self.controller,
                "analyze_failure",
                return_value=decision,
            ), patch.object(
                self.controller, "round_launch_profile", return_value="default"
            ), patch.object(
                self.controller,
                "prepare_and_submit_round",
                return_value=(session / "round_017_a10r1", "task-new", "run-new"),
            ):
                updated = self.controller.auto_retry_paused_current()

            self.assertEqual(17, updated["round_index"])
            self.assertEqual("a10r1", updated["round_label"])
            self.assertEqual("task-new", updated["active_task_id"])
            self.assertEqual("run-new", updated["active_run_id"])
            self.assertEqual("benchmark_failure", updated["last_failure_classification"])
            self.assertEqual(2, updated["total_failure_recovery_rounds"])

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

    def test_engine_frontend_handshake_timeout_gets_bounded_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            round_dir = Path(temporary)
            runtime_dir = round_dir / "04_runtime"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "worker.log").write_text(
                "EngineCore_DP1: Did not receive response from front-end process "
                "within 5 minutes\nLeaseProcessesPartialFailure active=1 inactive=1\n",
                encoding="utf-8",
            )
            decision = self.controller.deterministic_engine_frontend_handshake_retry(
                round_dir, self.baseline
            )
            self.assertIsNotNone(decision)
            self.assertEqual("retry_same", decision["action"])
            self.assertEqual(self.baseline, decision["candidate"])

    def test_engine_frontend_retry_fails_closed_when_oom_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            round_dir = Path(temporary)
            runtime_dir = round_dir / "04_runtime"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "worker.log").write_text(
                "Did not receive response from front-end process within 5 minutes\n"
                "LeaseProcessesPartialFailure active=1 inactive=1\nNPU OOM\n",
                encoding="utf-8",
            )
            self.assertIsNone(
                self.controller.deterministic_engine_frontend_handshake_retry(
                    round_dir, self.baseline
                )
            )

    def test_b0_retry_preserves_launch_profile_and_reconciliation(self) -> None:
        configured = production_b0_config()
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
