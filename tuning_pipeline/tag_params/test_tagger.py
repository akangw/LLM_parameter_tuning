from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from . import codex_tagger as tagger


class StructuredTaggerTests(unittest.TestCase):
    def test_tag_one_uses_provider_adapter_and_preserves_portrait(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "parameter.yaml"
            source.write_text(
                yaml.safe_dump({"name": "--test-parameter", "type": "int"}),
                encoding="utf-8",
            )
            output = root / "output"
            logs = root / "logs"
            output.mkdir()
            logs.mkdir()

            def fake_agent(_agent: dict, **kwargs: object) -> SimpleNamespace:
                Path(kwargs["output_path"]).write_text(
                    json.dumps({
                        "tags": {
                            "model": ["moe"],
                            "optimize_target": ["throughput"],
                            "deploy_topology": ["multi_node"],
                            "hardware": ["a3"],
                            "deploy_scenario": ["high_concurrency"],
                        }
                    }),
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    provider="anthropic", returncode=0, stdout="ok", stderr=""
                )

            with patch.object(tagger, "run_structured_agent", side_effect=fake_agent):
                tags, name, filename = tagger.tag_one(
                    {"provider": "anthropic", "settings": {}},
                    source,
                    output,
                    logs,
                    1,
                )

            self.assertEqual("--test-parameter", name)
            self.assertEqual(["a3"], tags["hardware"])
            tagged = yaml.safe_load((output / filename).read_text(encoding="utf-8"))
            self.assertEqual("int", tagged["type"])
            self.assertEqual(["throughput"], tagged["tags"]["optimize_target"])


if __name__ == "__main__":
    unittest.main()
