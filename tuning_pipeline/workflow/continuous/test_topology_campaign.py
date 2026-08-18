from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from .topology_campaign import (
    initialize_state,
    validate_agent_decision,
)


HERE = Path(__file__).resolve().parent


class TopologyCampaignTests(unittest.TestCase):
    def state(self) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            source = HERE / "server_autonomous" / "config.yaml"
            config = yaml.safe_load(source.read_text(encoding="utf-8"))
            config["base_config"] = str((HERE / "config.yaml").resolve())
            config["runtime"]["profile"] = "glm52_w8a8_a3_topology_campaign_v4"
            # The dormant topology Campaign remains a frozen v3 branch while
            # the fixed-DP4/TP8 mainline advances to Frontier V4.
            config["search_space"]["profile"] = "automatic_registry_a8_frontier_v3"
            config["strategy"]["profile"] = "hierarchical_agentic_frontier_v3"
            config["benchmark"]["profile"] = "aligned_fast_c32_v1"
            config["topology_campaign"]["enabled"] = True
            config_path = Path(temporary) / "campaign-config.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            _, state = initialize_state(
                config_path,
                Path(temporary) / "campaign",
            )
        return state

    def test_initial_state_isolates_sessions_and_keeps_tp32_blocked(self) -> None:
        state = self.state()
        self.assertEqual(
            ["a3_dp2_tp16", "a3_dp4_tp8"],
            state["plan"]["campaign_eligible_profiles"],
        )
        self.assertEqual("blocked", state["arms"]["a3_dp1_tp32"]["status"])
        self.assertNotEqual(
            state["arms"]["a3_dp2_tp16"]["runtime_root"],
            state["arms"]["a3_dp4_tp8"]["runtime_root"],
        )
        self.assertTrue(state["arms"]["a3_dp4_tp8"]["requires_live_preflight"])

    def test_agent_owns_feasibility_order_but_controller_requires_all_arms(self) -> None:
        state = self.state()
        decision = {
            "action": "allocate_budget",
            "incumbent_profile": None,
            "challenger_profile": None,
            "allocations": [
                {
                    "profile": "a3_dp4_tp8",
                    "additional_candidate_rounds": 1,
                    "intent": "high-upside topology first",
                },
                {
                    "profile": "a3_dp2_tp16",
                    "additional_candidate_rounds": 1,
                    "intent": "production control",
                },
            ],
            "summary": "Probe both feasible geometries.",
            "evidence": ["DP4 has a static-wiring-valid probe contract."],
            "risk_assessment": "DP4 may fail model fit; its budget is capped at one.",
        }
        validate_agent_decision(state, decision)
        decision["allocations"] = decision["allocations"][:1]
        with self.assertRaisesRegex(ValueError, "exactly one round"):
            validate_agent_decision(state, decision)

    def test_competitive_agent_allocation_keeps_challenger_floor(self) -> None:
        state = self.state()
        state["stage"] = "competitive"
        for name in state["plan"]["campaign_eligible_profiles"]:
            state["arms"][name]["measurement_count"] = 3
            state["arms"][name]["status"] = "ready"
        decision = {
            "action": "allocate_budget",
            "incumbent_profile": "a3_dp2_tp16",
            "challenger_profile": "a3_dp4_tp8",
            "allocations": [
                {
                    "profile": "a3_dp2_tp16",
                    "additional_candidate_rounds": 3,
                    "intent": "refine incumbent",
                },
                {
                    "profile": "a3_dp4_tp8",
                    "additional_candidate_rounds": 1,
                    "intent": "retain challenger exploration",
                },
            ],
            "summary": "Use a 3:1 measured allocation.",
            "evidence": ["Both arms have three comparable measurements."],
            "risk_assessment": "The challenger remains bounded but active.",
        }
        validate_agent_decision(state, decision)
        decision["allocations"] = [decision["allocations"][0]]
        decision["allocations"][0]["additional_candidate_rounds"] = 4
        with self.assertRaisesRegex(ValueError, "challenger allocation"):
            validate_agent_decision(state, decision)

    def test_controller_rejects_completion_before_final_verification(self) -> None:
        state = self.state()
        state["stage"] = "competitive"
        for name in state["plan"]["campaign_eligible_profiles"]:
            state["arms"][name]["measurement_count"] = 3
            state["arms"][name]["status"] = "ready"
        decision = {
            "action": "complete_campaign",
            "incumbent_profile": "a3_dp2_tp16",
            "challenger_profile": "a3_dp4_tp8",
            "allocations": [],
            "summary": "Premature completion.",
            "evidence": ["Screening only."],
            "risk_assessment": "Challenger has not received final verification.",
        }
        with self.assertRaisesRegex(ValueError, "before final"):
            validate_agent_decision(state, decision)


if __name__ == "__main__":
    unittest.main()
