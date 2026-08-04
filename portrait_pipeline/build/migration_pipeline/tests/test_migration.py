from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from build.migration_pipeline.cjx_bridge import _augment_config_value_fallbacks
from build.migration_pipeline.coverage import (
    audit_extraction_coverage,
    augment_indirect_surfaces,
)
from build.migration_pipeline import cjx_bridge
from build.migration_pipeline.migration import add_migration_context, classify
from build.migration_pipeline.stage1 import filter_parameters


class MigrationTests(unittest.TestCase):
    def test_high_impact_legacy_sleep_switch_is_rescued(self) -> None:
        # Regression guard for a real 0706 high-impact portrait that the base
        # hard-skip list classifies as an operational mode.
        source = Path(cjx_bridge.__file__).read_text(encoding="utf-8")
        self.assertIn(r'^--enable-sleep-mode$', source)
        self.assertIn("Q_SCALE_CONSTANT|K_SCALE_CONSTANT|V_SCALE_CONSTANT", source)
        self.assertIn(r'^VLLM_SPARSE_INDEXER_MAX_LOGITS_MB$', source)
        self.assertIn('parameter["category"] = "memory"', source)
        self.assertIn(r'^VLLM_MEDIA_LOADING_THREAD_COUNT$', source)

    def _legacy_dir(self, profiles: list[dict]) -> tempfile.TemporaryDirectory:
        tmp = tempfile.TemporaryDirectory()
        for index, profile in enumerate(profiles):
            Path(tmp.name, f"legacy-{index}.yaml").write_text(
                yaml.safe_dump(profile, allow_unicode=True), encoding="utf-8")
        return tmp

    def test_direct_stable_profile_is_a_and_is_attached_as_hint(self) -> None:
        current = [{"name": "--tensor-parallel-size", "type": "cli", "scope": "vllm", "default": 1}]
        legacy = [{"name": "--tensor-parallel-size", "type": "cli", "scope": "vllm", "default": 1,
                   "performance_impact": "high"}]
        with self._legacy_dir(legacy) as directory:
            manifest = classify(current, current, Path(directory))
        self.assertEqual(manifest["candidate_plan"][0]["migration_class"], "A")
        enriched = add_migration_context(current, manifest)
        self.assertEqual(enriched[0]["_migration"]["legacy_profiles"][0]["performance_impact"], "high")

    def test_default_change_is_b_and_missing_profile_is_current_only(self) -> None:
        current = [
            {"name": "--max-model-len", "type": "cli", "scope": "vllm", "default": -1},
            {"name": "--new-knob", "type": "cli", "scope": "vllm", "default": False},
        ]
        legacy = [{"name": "--max-model-len", "type": "cli", "scope": "vllm", "default": 8192}]
        with self._legacy_dir(legacy) as directory:
            manifest = classify(current, current, Path(directory))
        plan = {item["name"]: item["migration_class"] for item in manifest["candidate_plan"]}
        self.assertEqual(plan["--max-model-len"], "B")
        self.assertEqual(plan["--new-knob"], "CURRENT_ONLY")

    def test_supported_replacement_inherits_deprecated_profile_as_b(self) -> None:
        current = [{
            "name": "additional_config.enable_foo",
            "type": "nested",
            "scope": "vllm-ascend",
            "default": False,
            "replaces_deprecated": "VLLM_ASCEND_ENABLE_FOO",
        }]
        legacy = [{
            "name": "VLLM_ASCEND_ENABLE_FOO",
            "type": "env",
            "scope": "vllm-ascend",
            "default": "0",
            "performance_impact": "high",
        }]
        with self._legacy_dir(legacy) as directory:
            manifest = classify(current, current, Path(directory))
        plan = manifest["candidate_plan"][0]
        self.assertEqual("B", plan["migration_class"])
        self.assertEqual(
            "VLLM_ASCEND_ENABLE_FOO", plan["legacy_profiles"][0]["name"]
        )

    def test_flattened_legacy_config_maps_to_current_class_name(self) -> None:
        current = [{
            "name": "CompilationConfig.cudagraph_mode",
            "type": "nested",
            "scope": "vllm",
            "default": "NONE",
        }]
        legacy = [{
            "name": "compilation.cudagraph_mode",
            "type": "nested",
            "scope": "vllm",
            "default": "FULL_AND_PIECEWISE",
        }]
        with self._legacy_dir(legacy) as directory:
            manifest = classify(current, current, Path(directory))
        plan = manifest["candidate_plan"][0]
        self.assertEqual("B", plan["migration_class"])
        self.assertEqual(
            "compilation.cudagraph_mode", plan["legacy_profiles"][0]["name"]
        )
        self.assertEqual("structural_alias", manifest["profiles"][0]["mapping_method"])

    def test_misclassified_legacy_slo_cli_maps_to_nested_key(self) -> None:
        current = [{
            "name": "additional_config.SLO_limits_for_dynamic_batch",
            "type": "nested", "scope": "vllm-ascend", "default": -1,
        }]
        legacy = [{
            "name": "--SLO_limits_for_dynamic_batch",
            "type": "cli", "scope": "vllm-ascend", "default": -1,
        }]
        with self._legacy_dir(legacy) as directory:
            manifest = classify(current, current, Path(directory))
        self.assertEqual("B", manifest["candidate_plan"][0]["migration_class"])
        self.assertEqual(
            "corrected_legacy_interface", manifest["profiles"][0]["mapping_method"]
        )

    def test_stage1_keeps_ascend_surfaces_and_rejects_cuda_environment(self) -> None:
        passed, skipped, _ = filter_parameters([
            {"name": "VLLM_ASCEND_ENABLE_FOO", "type": "env", "description": ""},
            {"name": "VLLM_CUDA_FOO", "type": "env", "description": "cache"},
        ])
        self.assertEqual([item["name"] for item in passed], ["VLLM_ASCEND_ENABLE_FOO"])
        self.assertEqual(skipped[0]["reason"], "non_ascend_backend")

    def test_config_value_helper_promotes_supported_additional_config_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "vllm_ascend" / "ascend_config.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                "self.enable_foo = self._get_config_value(\n"
                "    additional_config, 'enable_foo', 'VLLM_ASCEND_ENABLE_FOO', fallback\n"
                ")\n",
                encoding="utf-8",
            )
            extraction = {
                "schema_version": "test",
                "sources": {},
                "parameters": [{
                    "id": "old",
                    "name": "VLLM_ASCEND_ENABLE_FOO",
                    "type": "env",
                    "scope": "vllm-ascend",
                    "category": "other",
                    "default": "1",
                    "value_type": "str",
                    "source_locations": [{"excerpt": "bool(int(os.getenv(...)))"}],
                }],
            }
            result = _augment_config_value_fallbacks(extraction, root)
            promoted = next(
                item for item in result["parameters"]
                if item["name"] == "additional_config.enable_foo"
            )
            self.assertIs(promoted["default"], True)
            self.assertEqual("bool", promoted["value_type"])
            self.assertEqual(
                "VLLM_ASCEND_ENABLE_FOO", promoted["replaces_deprecated"]
            )

    def test_indirect_additional_config_and_env_setdefault_are_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "pkg" / "runtime.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                "import os\n"
                "def enabled(additional_config):\n"
                "    value = False\n"
                "    if 'enable_dsa_cp' in additional_config:\n"
                "        value = bool(additional_config['enable_dsa_cp'])\n"
                "    return value\n"
                "os.environ.setdefault('TRITON_CACHE_AUTOTUNING', '1')\n",
                encoding="utf-8",
            )
            extraction = {"schema_version": "test", "parameters": []}
            result = augment_indirect_surfaces(extraction, root, root)
            by_name = {item["name"]: item for item in result["parameters"]}
            self.assertIs(by_name["additional_config.enable_dsa_cp"]["default"], False)
            self.assertEqual(by_name["additional_config.enable_dsa_cp"]["value_type"], "bool")
            self.assertEqual(by_name["TRITON_CACHE_AUTOTUNING"]["default"], "1")
            report = audit_extraction_coverage(result, root, root)
            self.assertTrue(report["ok"], report)

    def test_documented_external_runtime_controls_are_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs" / "source" / "developer_guide" / "performance_and_debug"
            docs.mkdir(parents=True)
            (docs / "optimization_and_tuning.md").write_text(
                "export TASK_QUEUE_ENABLE=2\n"
                "Use `HCCL_RDMA_TC` to configure the RDMA traffic class.\n",
                encoding="utf-8",
            )
            stale = root / "docs" / "source" / "user_guide" / "release_notes.md"
            stale.parent.mkdir(parents=True)
            stale.write_text("export ASCEND_BUFFER_POOL=99:99\n", encoding="utf-8")
            result = augment_indirect_surfaces(
                {"schema_version": "test", "parameters": []}, root, root
            )
            by_name = {item["name"]: item for item in result["parameters"]}
            self.assertEqual(2, by_name["TASK_QUEUE_ENABLE"]["default"])
            self.assertIn("HCCL_RDMA_TC", by_name)
            self.assertNotIn("ASCEND_BUFFER_POOL", by_name)
            report = audit_extraction_coverage(result, root, root)
            self.assertTrue(report["ok"], report)
            self.assertEqual(
                ["HCCL_RDMA_TC", "TASK_QUEUE_ENABLE"],
                report["documented_external_envs"],
            )

if __name__ == "__main__":
    unittest.main()
