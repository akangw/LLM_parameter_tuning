import json
import tempfile
import unittest
from pathlib import Path

import yaml

from build.codex_portrait_pipeline.pipeline import accept, prepare, sync_prepared_run


class PipelineTests(unittest.TestCase):
    def test_sync_refreshes_migration_metadata_without_resetting_status(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            extraction = root / "extraction.json"
            extraction.write_text(json.dumps({
                "extraction_hash": "h", "sources": {},
                "parameters": [{
                    "id": "param.1", "name": "--foo", "type": "cli",
                    "category": "memory", "scope": "vllm", "default": 1,
                    "source_locations": [],
                }],
            }), encoding="utf-8")
            old_manifest = root / "old.json"
            new_manifest = root / "new.json"
            old_manifest.write_text(json.dumps({"candidate_plan": [{
                "name": "--foo", "migration_class": "CURRENT_ONLY",
                "legacy_files": [], "legacy_profiles": [],
            }]}), encoding="utf-8")
            new_manifest.write_text(json.dumps({"candidate_plan": [{
                "name": "--foo", "migration_class": "B",
                "legacy_files": ["foo.yaml"], "legacy_profiles": [{"name": "--foo"}],
            }]}), encoding="utf-8")
            current, prepared = root / "current", root / "prepared"
            prepare(extraction, old_manifest, current)
            index = json.loads((current / "index.json").read_text(encoding="utf-8"))
            index["tasks"][0]["status"] = "skipped"
            (current / "index.json").write_text(json.dumps(index), encoding="utf-8")
            prepare(extraction, new_manifest, prepared)
            sync_prepared_run(current, prepared)
            merged = json.loads((current / "index.json").read_text(encoding="utf-8"))
            self.assertEqual("B", merged["tasks"][0]["migration_class"])
            self.assertEqual("skipped", merged["tasks"][0]["status"])

    def test_prepare_and_accept_full_schema(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            extraction = root / "extraction.json"
            manifest = root / "manifest.json"
            run = root / "run"
            extraction.write_text(json.dumps({
                "extraction_hash": "h",
                "sources": {},
                "parameters": [{
                    "id": "param.1", "name": "--foo", "type": "cli",
                    "category": "memory", "scope": "vllm", "default": 1,
                    "source_locations": [],
                }],
            }), encoding="utf-8")
            manifest.write_text(json.dumps({"candidate_plan": [{
                "name": "--foo", "migration_class": "CURRENT_ONLY",
                "legacy_files": [], "legacy_profiles": [],
            }]}), encoding="utf-8")
            prepare(extraction, manifest, run)
            draft = run / "draft.yaml"
            draft.write_text(yaml.safe_dump({
                "name": "--foo", "type": "cli", "category": "memory",
                "scope": "vllm", "source_file": [], "value_type": "int",
                "default": 1, "valid_choices": None,
                "cli_example": "--foo <value>", "deprecated": False,
                "performance_impact": "low", "performance_scope": ["memory"],
                "impact_detail": "Controls a memory setting.",
                "usage_locations": [], "related_parameters": [], "constraints": [],
                "tuning_advice": {
                    "summary": "Keep the default.", "suggested_values": [],
                    "caveats": [], "quick_guide": "Use 1.",
                },
            }, sort_keys=False), encoding="utf-8")
            accept(run, "param.1", draft)
            self.assertTrue((run / "params" / "foo.yaml").is_file())
            index = json.loads((run / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["summary"]["completed"], 1)


if __name__ == "__main__":
    unittest.main()
