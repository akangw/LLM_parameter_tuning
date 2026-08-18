from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from .compiler import (
    CONTINUOUS_DIR,
    MODULE_DIR,
    PROJECT_ROOT,
    SearchSpaceCompiler,
    exact_tag_match,
    validate_candidate,
    write_outputs,
)


class RecallTests(unittest.TestCase):
    def test_exact_tag_matching_does_not_use_substrings(self) -> None:
        param = {"tags": {"hardware": ["a30"], "model": ["moe", "mla"]}}
        self.assertFalse(exact_tag_match(param, "hardware", ["a3"]))
        self.assertTrue(exact_tag_match(param, "model", ["dense", "mla"]))


class ConstraintTests(unittest.TestCase):
    def test_physical_parallelism_must_fit_available_devices(self) -> None:
        scenario = {
            "topology": {
                "data_parallel_size": 2,
                "tensor_parallel_size": 16,
                "pipeline_parallel_size": 1,
                "total_npu": 32,
            }
        }
        self.assertIn(
            "physical_parallelism_within_available_devices",
            validate_candidate(
                {"prefill_context_parallel_size": 2}, scenario
            ),
        )

    def setUp(self) -> None:
        self.scenario = {"topology": {"tensor_parallel_size": 16}}

    def test_speculation_factor_is_not_artificially_coupled_to_tp(self) -> None:
        candidate = {
            "max_num_seqs": 48,
            "max_num_batched_tokens": 4096,
            "num_speculative_tokens": 2,
        }
        self.assertNotIn(
            "mtp_factor_divides_tensor_parallel",
            validate_candidate(candidate, self.scenario),
        )
        candidate["num_speculative_tokens"] = 3
        self.assertNotIn(
            "mtp_factor_divides_tensor_parallel",
            validate_candidate(candidate, self.scenario),
        )

    def test_long_prefill_threshold_is_bounded_by_batch_budget(self) -> None:
        candidate = {
            "max_num_seqs": 8,
            "max_num_batched_tokens": 4096,
            "num_speculative_tokens": 0,
            "long_prefill_token_threshold": 8192,
        }
        self.assertIn(
            "long_prefill_within_batch_budget",
            validate_candidate(candidate, self.scenario),
        )


class CompilerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = SearchSpaceCompiler(
            knowledge_dir=PROJECT_ROOT / "tag_params" / "output" / "params",
            scenario_path=MODULE_DIR / "scenario.glm52-a3-aligned-l1.yaml",
        )
        cls.result = cls.compiler.compile()

    def test_current_knowledge_recall_and_activation_budget(self) -> None:
        summary = self.result["summary"]
        # The knowledge base may grow when the fail-closed source coverage
        # audit finds a new current-version tuning surface.  Keep the previous
        # recall floor, but do not make a legitimate portrait addition require
        # an unrelated test edit.
        self.assertGreaterEqual(summary["tag_recalled_parameters"], 100)
        self.assertLessEqual(
            summary["tag_recalled_parameters"], summary["knowledge_parameters"]
        )
        self.assertGreaterEqual(summary["active_parameters"], 10)
        self.assertLessEqual(summary["active_parameters"], 15)
        self.assertGreater(summary["naive_active_combinations"], 100_000)
        self.assertFalse(self.result["integration"]["connected_to_mainflow"])

    def test_opted_in_knowledge_suggestions_expand_discrete_values(self) -> None:
        threshold = next(
            item
            for item in self.result["active_parameters"]
            if item["canonical_name"] == "long_prefill_token_threshold"
        )
        self.assertIn(512, threshold["values"])
        self.assertTrue(
            any(
                source["source"] == "knowledge_suggested_values"
                for source in threshold["value_sources"]
            )
        )

    def test_unverified_planned_parameter_requires_human_approval(self) -> None:
        planned = [
            item
            for item in self.result["active_parameters"]
            if item["integration_status"] == "planned"
        ]
        self.assertTrue(planned)
        for item in planned:
            self.assertEqual("unverified_for_current_image", item["availability"]["status"])
            self.assertEqual("human_required", item["approval"])

    def test_boolean_additional_config_axes_do_not_mix_int_aliases(self) -> None:
        for name in ("flashcomm1", "mlapo"):
            parameter = next(
                item
                for section in ("active_parameters", "reserve_candidates")
                for item in self.result[section]
                if item["canonical_name"] == name
            )
            values = parameter["values"]
            self.assertTrue(values)
            self.assertTrue(all(type(value) is bool for value in values))
            self.assertEqual({False, True}, set(values))

    def test_outputs_are_standalone_and_manifested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "compiled"
            files = write_outputs(self.result, output)
            self.assertEqual(7, len(files))
            agent_limits = yaml.safe_load(
                (output / "agent_search_limits.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                self.result["active_search_limits"], agent_limits["search_limits"]
            )
            self.assertNotIn("rejected_parameters", agent_limits)
            self.assertTrue(all(path.is_file() for path in files))
            self.assertTrue((output / "rotation_report.yaml").is_file())
            classified = yaml.safe_load(
                (output / "classified_search_limits.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(self.result["active_search_limits"], classified["active"])
            self.assertEqual(self.result["reserve_search_limits"], classified["reserve"])

    def test_material_migration_axes_are_active_or_reserve(self) -> None:
        classified = {
            **{name: "active" for name in self.result["active_search_limits"]},
            **{name: "reserve" for name in self.result["reserve_search_limits"]},
        }
        for name in {
            "enable_expert_parallel",
            "fused_mc2",
            "enable_balance_scheduling",
            "enable_reduce_sample",
            "speculative_config__enforce_eager",
        }:
            self.assertIn(name, classified)
        self.assertEqual("active", classified["enable_expert_parallel"])

    def test_live_controller_directory_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            write_outputs(self.result, CONTINUOUS_DIR / "search-space-test")

    def test_history_does_not_rotate_without_required_score_margin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            history_path = Path(temporary) / "history.json"
            history_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "session_id": "synthetic-test",
                        "trials": [
                            {
                                "trial_id": "trial_1",
                                "params": {
                                    "cudagraph_capture_sizes": [
                                        4,
                                        8,
                                        12,
                                        16,
                                        20,
                                        24,
                                        28,
                                        32,
                                        36,
                                        40,
                                        44,
                                        48,
                                    ]
                                },
                                "status": "success",
                                "objective_gain_percent": -10,
                                "metrics": {
                                    "total_token_throughput": 800,
                                    "median_ttft": 100,
                                    "median_tpot": 10,
                                    "failed_requests": 0,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            compiler = SearchSpaceCompiler(
                knowledge_dir=PROJECT_ROOT / "tag_params" / "output" / "params",
                scenario_path=MODULE_DIR / "scenario.glm52-a3-aligned-l1.yaml",
                history_path=history_path,
            )
            result = compiler.compile()
            swaps = result["rotation_audit"]["swaps"]
            self.assertEqual([], swaps)
            self.assertTrue(result["rotation_audit"]["enabled"])
            self.assertEqual(1, result["summary"]["attributed_history_trials"])
            evidence = result["history_analysis"]["parameters"][
                "cudagraph_capture_sizes"
            ]
            self.assertEqual(-10, evidence["mean_gain_percent"])
            self.assertIn("non_positive_measured_gain", evidence["reasons"])

    def test_baseline_only_history_does_not_trigger_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            history_path = Path(temporary) / "history.json"
            history_path.write_text(
                json.dumps(
                    [
                        {
                            "round": "round_000_a0",
                            "params": self.compiler.scenario["baseline"],
                            "metrics": {
                                "parse_status": "ok",
                                "metrics": {
                                    "total_token_throughput": 1000,
                                    "failed_requests": 0,
                                },
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            compiler = SearchSpaceCompiler(
                knowledge_dir=PROJECT_ROOT / "tag_params" / "output" / "params",
                scenario_path=MODULE_DIR / "scenario.glm52-a3-aligned-l1.yaml",
                history_path=history_path,
            )
            result = compiler.compile()
            self.assertEqual(0, result["summary"]["attributed_history_trials"])
            self.assertEqual([], result["rotation_audit"]["swaps"])
            self.assertEqual(
                "history_has_no_attributable_parameter_changes",
                result["rotation_audit"]["reason"],
            )

    def test_infrastructure_failure_is_not_charged_to_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            history_path = Path(temporary) / "history.json"
            baseline = dict(self.compiler.scenario["baseline"])
            changed = {**baseline, "max_num_batched_tokens": 8192}
            history_path.write_text(
                json.dumps(
                    [
                        {
                            "round": "baseline",
                            "params": baseline,
                            "outcome": "success",
                            "metrics": {
                                "parse_status": "ok",
                                "metrics": {
                                    "total_token_throughput": 1000,
                                    "failed_requests": 0,
                                },
                            },
                        },
                        {
                            "round": "infra_failure",
                            "params": changed,
                            "outcome": "failed",
                            "failure_decision": {
                                "classification": "network_or_hccl"
                            },
                        },
                        {
                            "round": "retry_success",
                            "params": changed,
                            "outcome": "success",
                            "metrics": {
                                "parse_status": "ok",
                                "metrics": {
                                    "total_token_throughput": 1100,
                                    "failed_requests": 0,
                                },
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )
            compiler = SearchSpaceCompiler(
                knowledge_dir=PROJECT_ROOT / "tag_params" / "output" / "params",
                scenario_path=MODULE_DIR / "scenario.glm52-a3-aligned-l1.yaml",
                history_path=history_path,
            )
            result = compiler.compile()
            history = result["history_analysis"]
            stats = history["parameters"]["max_num_batched_tokens"]
            self.assertEqual(1, history["ignored_non_parameter_failure_trials"])
            self.assertEqual(1, stats["trial_count"])
            self.assertEqual(0.0, stats["failure_rate"])
            self.assertAlmostEqual(10.0, stats["mean_gain_percent"])

    def test_parameter_oom_excludes_only_complete_combination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            history_path = Path(temporary) / "history.json"
            baseline = dict(self.compiler.scenario["baseline"])
            failed = {**baseline, "max_num_batched_tokens": 16384}
            history_path.write_text(
                json.dumps(
                    [
                        {
                            "round": "baseline",
                            "params": baseline,
                            "outcome": "success",
                            "metrics": {
                                "parse_status": "ok",
                                "metrics": {
                                    "total_token_throughput": 1000,
                                    "failed_requests": 0,
                                },
                            },
                        },
                        {
                            "round": "oom",
                            "params": failed,
                            "outcome": "failed",
                            "failure_decision": {
                                "classification": "parameter_oom",
                                "changes": [
                                    {
                                        "parameter": "max_num_batched_tokens",
                                        "before": 16384,
                                        "after": 8192,
                                    }
                                ],
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )
            compiler = SearchSpaceCompiler(
                knowledge_dir=PROJECT_ROOT / "tag_params" / "output" / "params",
                scenario_path=MODULE_DIR / "scenario.glm52-a3-aligned-l1.yaml",
                history_path=history_path,
            )
            result = compiler.compile()
            parameter = next(
                item
                for item in [
                    *result["active_parameters"],
                    *result["reserve_candidates"],
                ]
                if item["canonical_name"] == "max_num_batched_tokens"
            )
            self.assertIn(16384, parameter["values"])
            self.assertNotIn("history_pruned_values", parameter)
            self.assertEqual(
                1.0, parameter["history_evidence"]["failure_rate"]
            )
            exclusions = result["history_analysis"]["conditional_exclusions"]
            self.assertEqual(1, len(exclusions))
            self.assertEqual("parameter_oom", exclusions[0]["failure_classification"])
            self.assertEqual(
                16384,
                exclusions[0]["conditions"]["max_num_batched_tokens"],
            )


if __name__ == "__main__":
    unittest.main()
