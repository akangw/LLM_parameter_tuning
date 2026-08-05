from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from .builder import CONTINUOUS_DIR, MODULE_DIR, PROJECT_ROOT
from .pipeline import (
    AutomaticRegistryPipeline,
    DEFAULT_SOURCE_ROOT,
    compile_generic_runtime_payload,
    render_generic_injection,
    semantic_groups,
    verify_source_identity,
    write_full_outputs,
)
from .candidate_validator import validate_trial_candidate
from .compatibility import CompatibilityValidator


class GenericInjectionTests(unittest.TestCase):
    def test_supported_injection_contracts_render_deterministically(self) -> None:
        self.assertEqual(
            {"cli_args": ["--max-num-seqs", "32"]},
            render_generic_injection(
                {"kind": "cli_value", "flag": "--max-num-seqs"}, 32
            ),
        )
        self.assertEqual(
            {"environment": {"VLLM_FEATURE": "1"}},
            render_generic_injection(
                {"kind": "env_bool", "name": "VLLM_FEATURE"}, True
            ),
        )
        self.assertEqual(
            {
                "json_patch": {
                    "path": ["speculative_config", "method"],
                    "value": "mtp",
                }
            },
            render_generic_injection(
                {"kind": "json_path", "path": ["speculative_config", "method"]},
                "mtp",
            ),
        )

    def test_unrelated_generic_nested_leaf_names_are_not_merged(self) -> None:
        records = [
            {
                "canonical_name": "SpeculativeConfig.method",
                "source_type": "nested",
                "knowledge_names": ["SpeculativeConfig.method"],
                "related_parameters": [],
            },
            {
                "canonical_name": "CompilationConfig.method",
                "source_type": "nested",
                "knowledge_names": ["CompilationConfig.method"],
                "related_parameters": [],
            },
        ]
        groups = semantic_groups(records)
        self.assertEqual(2, len(groups))
        self.assertNotIn("method", groups)

    def test_generated_payload_combines_cli_environment_and_json_contracts(
        self,
    ) -> None:
        payload = compile_generic_runtime_payload(
            {
                "rows": 32,
                "feature": True,
                "method": "mtp",
                "omitted": None,
            },
            {
                "rows": {"kind": "cli_value", "flag": "--rows"},
                "feature": {"kind": "env_bool", "name": "FEATURE"},
                "method": {
                    "kind": "json_path",
                    "path": ["speculative_config", "method"],
                },
                "omitted": {
                    "kind": "json_path",
                    "path": ["compilation_config", "optional"],
                },
            },
        )
        self.assertEqual(["--rows", "32"], payload["cli_args"])
        self.assertEqual({"FEATURE": "1"}, payload["environment"])
        self.assertEqual(
            {"speculative_config": {"method": "mtp"}},
            payload["json_configs"],
        )


class CurrentAutomaticPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        compiler_dir = MODULE_DIR.parent / "search_space_compiler"
        cls.pipeline = AutomaticRegistryPipeline(
            knowledge_dir=PROJECT_ROOT / "tag_params" / "output" / "params",
            scenario_path=compiler_dir / "scenario.glm52-a3-aligned-l1.yaml",
            policy_path=compiler_dir / "policy.yaml",
        )
        cls.registry, cls.search_result = cls.pipeline.compile()

    def test_full_chain_is_automatic_and_isolated(self) -> None:
        audit = self.registry["audit"]
        summary = self.search_result["summary"]
        self.assertEqual(109, audit["tag_recalled_parameters"])
        self.assertGreater(audit["compatible_registry_parameters"], 23)
        self.assertFalse(audit["existing_registry_dependency"])
        self.assertFalse(audit["connected_to_mainflow"])
        self.assertFalse(self.search_result["integration"]["connected_to_mainflow"])
        self.assertEqual("automatic_registry_v1", self.registry["mode"])
        self.assertGreaterEqual(summary["active_parameters"], 10)
        self.assertLessEqual(summary["active_parameters"], 15)
        self.assertGreater(summary["reserve_parameters"], 0)

    def test_mismatched_source_identity_fails_closed(self) -> None:
        mismatched = yaml.safe_load(
            yaml.safe_dump(self.pipeline.scenario, sort_keys=False)
        )
        mismatched["image"]["vllm_commit"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "Pinned source mismatch"):
            verify_source_identity(DEFAULT_SOURCE_ROOT, mismatched)

    def test_final_tunable_pool_excludes_known_non_executable_axes(self) -> None:
        summary = self.search_result["summary"]
        self.assertEqual(36, summary["eligible_tunable_parameters"])
        self.assertEqual(12, summary["active_parameters"])
        self.assertEqual(24, summary["reserve_parameters"])
        tunable = {
            item["canonical_name"]
            for item in self.search_result["active_parameters"]
            + self.search_result["reserve_candidates"]
        }
        for name in {
            "all2all_backend",
            "attention_backend",
            "attention_config__mla_prefill_backend",
            "compilation_config__backend",
            "additional_config__finegrained_tp_config__lmhead_tensor_parallel_size",
            "speculative_config__model",
            "speculative_config__quantization",
            "eplb_num_redundant_experts",
        }:
            self.assertNotIn(name, tunable)

    def test_every_generated_value_passes_generic_injection_validation(self) -> None:
        for parameter in self.registry["parameters"]:
            for value in parameter["candidate_values"]:
                rendered = render_generic_injection(parameter["injection"], value)
                self.assertIsInstance(rendered, dict)

    def test_every_recalled_parameter_has_a_final_audit_outcome(self) -> None:
        outcomes = self.registry["recall_outcomes"]
        self.assertEqual(
            self.registry["audit"]["tag_recalled_parameters"], len(outcomes)
        )
        self.assertTrue(
            all(
                item["final_status"]
                in {
                    "accepted_to_automatic_registry",
                    "merged_as_deprecated_alias",
                    "excluded_fail_closed",
                }
                for item in outcomes
            )
        )

    def test_ambiguous_raw_nested_names_do_not_enter_active_limits(self) -> None:
        self.assertNotIn("backend", self.search_result["active_search_limits"])
        self.assertNotIn("method", self.search_result["active_search_limits"])

    def test_automatic_active_limits_are_scenario_compatible(self) -> None:
        expected = {
            "max_num_seqs",
            "max_num_batched_tokens",
            "gpu_memory_utilization",
            "compilation_mode",
            "num_speculative_tokens",
            "enable_chunked_prefill",
            "cudagraph_capture_sizes",
            "max_cudagraph_capture_size",
            "mlapo",
            "enable_prefix_caching",
            "flashcomm1",
            "speculative_config__method",
        }
        self.assertEqual(expected, set(self.search_result["active_search_limits"]))
        self.assertNotIn("TORCH_COMPILE_DISABLE", expected)
        self.assertNotIn("COMPILE_CUSTOM_KERNELS", expected)

    def test_omit_tokens_are_actions_not_literal_environment_values(self) -> None:
        parameter = next(
            item
            for item in self.registry["parameters"]
            if item["canonical_name"] == "TORCH_COMPILE_DISABLE"
        )
        self.assertEqual([None, "1"], parameter["candidate_values"])
        self.assertNotIn("unset", parameter["candidate_values"])

    def test_disabled_feature_families_are_excluded(self) -> None:
        names = {item["canonical_name"] for item in self.registry["parameters"]}
        self.assertFalse(
            any(name.startswith("k_v_transfer_config__") for name in names)
        )
        self.assertNotIn("DYNAMIC_EPLB", names)
        self.assertNotIn("eplb_num_redundant_experts", names)
        self.assertNotIn("max_loras", names)

    def test_combination_validator_rejects_incompatible_settings(self) -> None:
        validator = CompatibilityValidator(scenario=self.pipeline.scenario)
        self.assertIn(
            "speculative_method_requires_tokens",
            validator.validate_combination(
                {
                    "speculative_config__method": "mtp",
                    "num_speculative_tokens": 0,
                }
            ),
        )
        self.assertIn(
            "speculative_tokens_require_method",
            validator.validate_combination(
                {
                    "speculative_config__method": None,
                    "num_speculative_tokens": 1,
                }
            ),
        )
        self.assertIn(
            "eager_conflicts_with_npugraph",
            validator.validate_combination(
                {
                    "speculative_config__enforce_eager": True,
                    "additional_config__ascend_compilation_config__enable_npugraph_ex": True,
                }
            ),
        )

    def test_trial_candidate_is_domain_constraint_and_injection_validated(self) -> None:
        report = validate_trial_candidate(
            candidate={"max_num_seqs": 32, "speculative_config__method": "mtp"},
            compiled=self.search_result,
            scenario=self.pipeline.scenario,
            compatibility=self.pipeline.compatibility,
        )
        self.assertTrue(report["valid"], report["violations"])
        invalid = validate_trial_candidate(
            candidate={"max_num_seqs": 999999},
            compiled=self.search_result,
            scenario=self.pipeline.scenario,
            compatibility=self.pipeline.compatibility,
        )
        self.assertFalse(invalid["valid"])
        self.assertEqual("value_not_in_compiled_domain", invalid["violations"][0]["id"])

    def test_full_chain_does_not_modify_manual_registry(self) -> None:
        manual_registry = MODULE_DIR.parent / "search_space_compiler" / "registry.yaml"
        before = manual_registry.read_bytes()
        self.pipeline.compile()
        self.assertEqual(before, manual_registry.read_bytes())

    def test_outputs_are_complete_manifested_and_still_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "full"
            files = write_full_outputs(self.registry, self.search_result, output)
            self.assertEqual(10, len(files))
            self.assertTrue((output / "registry.generated.yaml").is_file())
            self.assertTrue((output / "registry.audit.json").is_file())
            self.assertTrue((output / "pipeline_manifest.json").is_file())
            self.assertTrue((output / "compatibility_constraints.yaml").is_file())
            self.assertTrue(
                (output / "search_limits" / "agent_search_limits.yaml").is_file()
            )
            limits = yaml.safe_load(
                (output / "search_limits" / "agent_search_limits.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                self.search_result["active_search_limits"], limits["search_limits"]
            )

    def test_live_controller_directory_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            write_full_outputs(
                self.registry,
                self.search_result,
                CONTINUOUS_DIR / "auto-registry-full-test",
            )


if __name__ == "__main__":
    unittest.main()
