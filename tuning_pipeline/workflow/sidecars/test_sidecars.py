from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from .portrait_retriever import (
    CONTINUOUS_DIR,
    PortraitRetriever,
    write_evidence,
)
from .runtime_rule_store import RuntimeRuleStore


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
SCENARIO_PATH = (
    PROJECT_ROOT
    / "workflow"
    / "search_space_compiler"
    / "scenario.glm52-a3-aligned-l1.yaml"
)


class PortraitRetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = PortraitRetriever()

    def test_changed_parameter_and_one_hop_portraits_are_preserved(self) -> None:
        result = self.retriever.retrieve(
            ["compilation_enable_sp"],
            search_limits={"compilation_enable_sp": [False, True]},
        )
        changed = result["changed_parameters"][0]
        self.assertEqual("compilation_enable_sp", changed["canonical_name"])
        self.assertGreaterEqual(changed["variant_count"], 1)
        constraints = [
            constraint
            for variant in changed["variants"]
            for constraint in variant["portrait"].get("constraints", [])
        ]
        self.assertTrue(any("tensor_parallel_size" in item for item in constraints))
        related = {
            item["canonical_name"]
            for item in result["one_hop_related_parameters"]
        }
        # One-hop names come from the current portrait rather than the old
        # 0706 portrait's fixed relation list.
        source_relations = {
            self.retriever.canonical_name(str(item["name"]))
            for variant in changed["variants"]
            for item in variant["portrait"].get("related_parameters", [])
            if isinstance(item, dict) and item.get("name")
        }
        self.assertTrue(source_relations)
        self.assertTrue(source_relations.intersection(related))

    def test_alias_resolves_to_current_portrait_variant(self) -> None:
        result = self.retriever.retrieve(
            ["max_num_seqs"],
            include_one_hop=False,
        )
        changed = result["changed_parameters"][0]
        self.assertEqual("max_num_seqs", changed["canonical_name"])
        self.assertGreaterEqual(changed["variant_count"], 1)
        self.assertEqual("--max-num-seqs", changed["variants"][0]["portrait"]["name"])

    def test_automatic_nested_axis_resolves_without_legacy_registry_entry(self) -> None:
        name = "additional_config__ascend_compilation_config__fuse_allreduce_rms"
        result = self.retriever.retrieve(
            [name],
            search_limits={name: [False, True]},
            include_one_hop=False,
        )
        changed = result["changed_parameters"][0]
        self.assertEqual(name, changed["canonical_name"])
        self.assertGreaterEqual(changed["variant_count"], 1)
        self.assertTrue(
            any(
                variant["portrait"]["name"]
                == "additional_config.ascend_compilation_config.fuse_allreduce_rms"
                for variant in changed["variants"]
            )
        )

    def test_speculative_config_axes_resolve_class_named_portraits(self) -> None:
        names = (
            "speculative_config__method",
            "speculative_config__disable_padded_drafter_batch",
            "speculative_config__attention_backend",
        )
        result = self.retriever.retrieve(
            names,
            search_limits={name: [None, True] for name in names},
            include_one_hop=False,
        )
        self.assertEqual([], result["unresolved_names"])
        self.assertTrue(
            all(item["variant_count"] >= 1 for item in result["changed_parameters"])
        )

    def test_parameter_outside_search_limits_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.retriever.retrieve(
                ["enable_eplb"],
                search_limits={"max_num_seqs": [8, 16]},
            )

    def test_writes_below_continuous_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            write_evidence(
                {"schema_version": 1},
                CONTINUOUS_DIR / "forbidden-sidecar-output.yaml",
            )


class RuntimeRuleStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = RuntimeRuleStore.initialize(self.root / "rules.yaml")
        self.scenario = yaml.safe_load(SCENARIO_PATH.read_text(encoding="utf-8"))
        self.candidate = {
            "max_num_seqs": 48,
            "max_model_len": 64000,
            "max_num_batched_tokens": 4096,
            "num_speculative_tokens": 3,
            "async_scheduling": True,
            "long_prefill_token_threshold": 0,
            "enable_chunked_prefill": True,
            "enable_eplb": False,
            "eplb_num_redundant_experts": 0,
            "cudagraph_capture_sizes": None,
            "max_cudagraph_capture_size": 192,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_active_deterministic_rule_rejects_candidate(self) -> None:
        candidate = {**self.candidate, "eplb_num_redundant_experts": 8}
        result = self.store.evaluate(candidate, scenario=self.scenario)
        self.assertFalse(result["allowed"])
        self.assertIn(
            "eplb_redundant_experts_requires_eplb",
            {item["id"] for item in result["violations"]},
        )

    def test_non_chunked_prefill_requires_full_model_token_budget(self) -> None:
        invalid = {**self.candidate, "enable_chunked_prefill": False}
        result = self.store.evaluate(invalid, scenario=self.scenario)
        self.assertFalse(result["allowed"])
        self.assertIn(
            "non_chunked_prefill_requires_full_model_token_budget",
            {item["id"] for item in result["violations"]},
        )
        self.assertTrue(
            self.store.evaluate(self.candidate, scenario=self.scenario)["allowed"]
        )

    def test_single_parameter_hard_failure_auto_quarantines_exact_value(self) -> None:
        history = self.root / "history.json"
        history.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "session_id": "session-a",
                    "trials": [
                        {
                            "trial_id": "trial-1",
                            "params": {
                                **self.candidate,
                                "max_num_batched_tokens": 16384,
                            },
                            "status": "failure",
                            "failure_classification": "parameter_oom",
                            "attributed_parameters": ["max_num_batched_tokens"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        audit = self.store.ingest_history(history, scenario=self.scenario)
        self.assertEqual(1, audit["quarantines_added"])
        result = self.store.evaluate(
            {**self.candidate, "max_num_batched_tokens": 16384},
            scenario=self.scenario,
        )
        self.assertFalse(result["allowed"])
        self.assertTrue(
            any(item["source"] == "history_quarantine" for item in result["violations"])
        )
        repeated = self.store.ingest_history(history, scenario=self.scenario)
        self.assertEqual("already_processed", repeated["status"])

    def test_multi_parameter_failure_only_creates_non_blocking_proposal(self) -> None:
        history = self.root / "history.json"
        trials = []
        for index in range(2):
            trials.append(
                {
                    "trial_id": f"trial-{index}",
                    "params": {
                        **self.candidate,
                        "enable_eplb": True,
                        "eplb_num_redundant_experts": 8,
                    },
                    "status": "failure",
                    "failure_classification": "parameter_runtime",
                    "attributed_parameters": [
                        "enable_eplb",
                        "eplb_num_redundant_experts",
                    ],
                }
            )
        history.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "session_id": "session-b",
                    "trials": trials,
                }
            ),
            encoding="utf-8",
        )
        audit = self.store.ingest_history(history, scenario=self.scenario)
        self.assertEqual(1, audit["proposals_added"])
        self.assertEqual(0, audit["generalized_rules_activated"])
        candidate = {
            **self.candidate,
            "enable_eplb": True,
            "eplb_num_redundant_experts": 8,
        }
        result = self.store.evaluate(candidate, scenario=self.scenario)
        self.assertTrue(result["allowed"])
        self.assertTrue(result["warnings"])

        proposal_id = self.store.data["proposals"][0]["id"]
        self.store.transition_proposal(proposal_id, "active")
        result = self.store.evaluate(candidate, scenario=self.scenario)
        self.assertFalse(result["allowed"])

    def test_search_limits_are_an_independent_hard_gate(self) -> None:
        result = self.store.evaluate(
            self.candidate,
            scenario=self.scenario,
            search_limits={
                **{name: [value] for name, value in self.candidate.items()},
                "max_num_seqs": [8, 16],
            },
        )
        self.assertFalse(result["allowed"])
        self.assertIn(
            "search_limits_value:max_num_seqs",
            {item["id"] for item in result["violations"]},
        )

    def test_migration_accelerators_keep_their_coupled_prerequisites(self) -> None:
        fused_without_ep = self.store.evaluate(
            {
                **self.candidate,
                "fused_mc2": 1,
                "enable_expert_parallel": False,
                "num_speculative_tokens": 0,
                "speculative_config__enforce_eager": None,
            },
            scenario=self.scenario,
        )
        self.assertFalse(fused_without_ep["allowed"])
        self.assertIn(
            "fused_mc2_requires_expert_parallel",
            {item["id"] for item in fused_without_ep["violations"]},
        )

        draft_eager_without_mtp = self.store.evaluate(
            {
                **self.candidate,
                "fused_mc2": 0,
                "enable_expert_parallel": False,
                "num_speculative_tokens": 0,
                "speculative_config__enforce_eager": True,
            },
            scenario=self.scenario,
        )
        self.assertFalse(draft_eager_without_mtp["allowed"])
        self.assertIn(
            "draft_eager_requires_speculation",
            {item["id"] for item in draft_eager_without_mtp["violations"]},
        )


if __name__ == "__main__":
    unittest.main()
