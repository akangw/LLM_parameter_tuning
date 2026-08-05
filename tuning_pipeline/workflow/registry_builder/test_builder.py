from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from .builder import (
    CONTINUOUS_DIR,
    MODULE_DIR,
    PROJECT_ROOT,
    RegistryBuilder,
    candidate_values,
    canonical_name,
    exact_tag_match,
    inferred_injection,
    write_outputs,
)


class InferenceTests(unittest.TestCase):
    def test_exact_tag_matching_does_not_use_substrings(self) -> None:
        param = {"tags": {"hardware": ["a30"]}}
        self.assertFalse(exact_tag_match(param, "hardware", ["a3"]))

    def test_entrypoint_inference(self) -> None:
        cli, _ = inferred_injection(
            {"name": "--max-num-seqs", "type": "cli", "value_type": "int"}
        )
        env, _ = inferred_injection(
            {"name": "VLLM_FEATURE", "type": "env", "value_type": "bool"}
        )
        nested, reasons = inferred_injection(
            {"name": "CompilationConfig.mode", "type": "nested", "value_type": "str"}
        )
        self.assertEqual("max_num_seqs", canonical_name("--max-num-seqs", "cli"))
        self.assertEqual("cli_value", cli["kind"])
        self.assertEqual("env_bool", env["kind"])
        self.assertEqual("nested_field", nested["kind"])
        self.assertIn("generic_nested_path_requires_controller_adapter", reasons)

    def test_candidate_values_are_discrete_and_deduplicated(self) -> None:
        values, sources = candidate_values(
            {
                "value_type": "bool",
                "default": "0",
                "valid_choices": ["0", "1"],
                "tuning_advice": {"suggested_values": [{"value": "1"}]},
            }
        )
        self.assertEqual([False, True], values)
        self.assertTrue(sources)

    def test_source_default_null_is_preserved_as_omit_candidate(self) -> None:
        values, _ = candidate_values(
            {
                "value_type": "int",
                "default": None,
                "tuning_advice": {"suggested_values": [{"value": 32}]},
            }
        )
        self.assertEqual([None, 32], values)


class CurrentKnowledgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        compiler_dir = MODULE_DIR.parent / "search_space_compiler"
        cls.builder = RegistryBuilder(
            knowledge_dir=PROJECT_ROOT / "tag_params" / "output" / "params",
            scenario_path=compiler_dir / "scenario.glm52-a3-aligned-l1.yaml",
            policy_path=compiler_dir / "policy.yaml",
        )
        cls.result = cls.builder.build()

    def test_current_recall_is_covered_without_existing_registry(self) -> None:
        summary = self.result["summary"]
        self.assertGreaterEqual(summary["tag_recalled_parameters"], 100)
        self.assertGreater(
            summary["generated_candidates"] + summary["review_required_candidates"],
            23,
        )
        self.assertFalse(summary["existing_registry_dependency"])
        self.assertFalse(summary["connected_to_mainflow"])

    def test_every_recalled_parameter_has_an_audited_outcome(self) -> None:
        summary = self.result["summary"]
        classified = (
            summary["generated_candidates"]
            + summary["review_required_candidates"]
            + summary["unsupported_candidates"]
        )
        self.assertEqual(summary["tag_recalled_parameters"], classified)

    def test_deprecated_parameters_never_enter_generated_or_review(self) -> None:
        for section in ("generated_candidates", "review_queue"):
            for item in self.result[section]:
                self.assertNotIn("deprecated_parameter", item.get("review_reasons", []))

    def test_build_does_not_modify_existing_registry(self) -> None:
        registry = MODULE_DIR.parent / "search_space_compiler" / "registry.yaml"
        before = registry.read_bytes()
        self.builder.build()
        self.assertEqual(before, registry.read_bytes())

    def test_outputs_are_isolated_and_manifested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "proposal"
            files = write_outputs(self.result, output)
            self.assertEqual(5, len(files))
            generated = yaml.safe_load(
                (output / "registry.generated.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                "proposal_only_not_connected_to_mainflow", generated["mode"]
            )
            self.assertTrue((output / "manifest.json").is_file())

    def test_live_controller_directory_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            write_outputs(self.result, CONTINUOUS_DIR / "generated-registry-test")


if __name__ == "__main__":
    unittest.main()
