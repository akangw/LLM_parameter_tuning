from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from . import hierarchical_strategy as strategy


class HierarchicalStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        profiles_path = Path(__file__).with_name("strategy_profiles.yaml")
        with profiles_path.open(encoding="utf-8") as handle:
            profiles = yaml.safe_load(handle)
        self.assertEqual(profiles["default_strategy"], "best_anchor_coverage_v2")
        self.profile = profiles["strategies"]["best_anchor_coverage_v3"]
        self.assertEqual(self.profile["status"], "standalone_not_integrated")
        self.anchor = [
            {"workload": "chat-1024-256", "concurrency": 32, "aggregate_output_tps": 100},
            {"workload": "prefill-8192-512", "concurrency": 32, "aggregate_output_tps": 100},
            {"workload": "balanced-1024-1024", "concurrency": 32, "aggregate_output_tps": 100},
            {"workload": "decode-256-2048", "concurrency": 32, "aggregate_output_tps": 100},
        ]

    def test_screening_plan_is_explicit_c32_subset(self) -> None:
        plan = strategy.screening_plan(self.profile)
        self.assertEqual(plan["benchmark_mode"], "aligned_l1_screen")
        self.assertEqual(len(plan["formal_cases"]), 4)
        self.assertTrue(all(case["concurrency"] == 32 for case in plan["formal_cases"]))

    def test_good_screen_promotes_only_to_full_verification(self) -> None:
        candidate = [dict(case, aggregate_output_tps=101) for case in self.anchor]
        result = strategy.assess_screening(self.profile, self.anchor, candidate)
        self.assertTrue(result["promote_to_full_verification"])
        self.assertEqual(result["next_stage"], "full_verification")
        self.assertEqual(strategy.next_stage_after_full_verification({}), "exploration")

    def test_exploration_requires_two_independent_parameters_unless_excepted(self) -> None:
        strategy.validate_exploration_change_count(
            self.profile,
            ["max_num_seqs", "max_num_batched_tokens"],
        )
        with self.assertRaisesRegex(ValueError, "explicit exception"):
            strategy.validate_exploration_change_count(
                self.profile,
                ["gpu_memory_utilization"],
            )
        strategy.validate_exploration_change_count(
            self.profile,
            ["gpu_memory_utilization"],
            exception_reason="high-risk memory limit needs isolated confirmation",
        )

    def test_screen_rejects_workload_regression_or_error(self) -> None:
        candidate = [dict(case, aggregate_output_tps=101) for case in self.anchor]
        candidate[2]["aggregate_output_tps"] = 90
        candidate[3]["failed_requests"] = 1
        result = strategy.assess_screening(self.profile, self.anchor, candidate)
        self.assertFalse(result["promote_to_full_verification"])
        self.assertEqual(result["next_stage"], "exploration")
        self.assertGreaterEqual(len(result["violations"]), 2)


if __name__ == "__main__":
    unittest.main()
