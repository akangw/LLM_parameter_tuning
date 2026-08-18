from __future__ import annotations

import unittest
from pathlib import Path

from .topology_advisor import (
    build_plan,
    load_document,
    select_measured_topology,
    validate_topology_baseline,
)

import yaml


HERE = Path(__file__).resolve().parent


class TopologyAdvisorTests(unittest.TestCase):
    def test_w8a8_plan_keeps_proven_incumbent_and_experimental_dp4(self) -> None:
        plan = build_plan(
            load_document(HERE / "topology_profiles.yaml"),
            model_contract="glm-5.2-w8a8",
            available_nodes=2,
            npu_per_node=16,
        )
        self.assertEqual(
            ["a3_dp2_tp16", "a3_dp4_tp8"], plan["eligible_profiles"]
        )
        self.assertEqual(["a3_dp2_tp16"], plan["production_eligible_profiles"])
        self.assertEqual(["a3_dp4_tp8"], plan["experimental_eligible_profiles"])
        self.assertEqual("a3_dp2_tp16", plan["recommended_profile"])
        dp4 = next(item for item in plan["candidates"] if item["profile"] == "a3_dp4_tp8")
        self.assertTrue(dp4["eligible"])
        self.assertTrue(dp4["requires_live_preflight"])
        dp1 = next(item for item in plan["candidates"] if item["profile"] == "a3_dp1_tp32")
        self.assertFalse(dp1["eligible"])

    def test_measured_selection_uses_only_gate_passed_eligible_profiles(self) -> None:
        plan = {
            "eligible_profiles": ["one", "two"],
            "recommended_profile": "one",
        }
        selected = select_measured_topology(
            plan,
            [
                {"profile": "one", "gate_passed": True, "output_token_throughput": 10},
                {"profile": "two", "gate_passed": True, "output_token_throughput": 12},
                {"profile": "blocked", "gate_passed": True, "output_token_throughput": 99},
            ],
            agent_selected_profile="two",
        )
        self.assertEqual("two", selected["selected_profile"])
        self.assertEqual(
            "agent_selected_controller_validated", selected["selection_basis"]
        )

    def test_controller_does_not_replace_missing_agent_choice_with_argmax(self) -> None:
        selected = select_measured_topology(
            {"eligible_profiles": ["one", "two"], "recommended_profile": "one"},
            [
                {"profile": "one", "gate_passed": True, "output_token_throughput": 10},
                {"profile": "two", "gate_passed": True, "output_token_throughput": 12},
            ],
        )
        self.assertIsNone(selected["selected_profile"])
        self.assertEqual("agent_selection_required", selected["selection_basis"])

    def test_dp4_probe_baseline_matches_frozen_geometry(self) -> None:
        document = load_document(HERE / "topology_profiles.yaml")
        baseline = yaml.safe_load(
            (HERE.parent / "baselines" / "a8_glm52_w8a8_dp4_tp8_probe_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
        result = validate_topology_baseline(
            "a3_dp4_tp8", document["profiles"]["a3_dp4_tp8"], baseline
        )
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
