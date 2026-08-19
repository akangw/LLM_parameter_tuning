from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .server_autonomous.prepare_decode_only_benchmark import prepare


class PrepareDecodeOnlyBenchmarkTests(unittest.TestCase):
    def test_preparation_is_additive_and_keeps_source_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "fast-v2" / "spec"
            target = root / "decode-v1" / "spec"
            overlay = root / "repo" / "01_decode.yaml"
            (source / "schemas").mkdir(parents=True)
            (source / "schemas" / "schema.json").write_text("{}", encoding="utf-8")
            (source / "suites").mkdir()
            (source / "suites" / "old.yaml").write_text("old", encoding="utf-8")
            overlay.parent.mkdir(parents=True)
            overlay.write_text("decode", encoding="utf-8")

            result = prepare(
                allowed_root=root,
                source_spec_root=source,
                target_spec_root=target,
                suite_overlay=overlay,
            )

            self.assertEqual(target / "suites" / "01_decode.yaml", result)
            self.assertEqual("decode", result.read_text(encoding="utf-8"))
            self.assertEqual(
                "old", (source / "suites" / "old.yaml").read_text(encoding="utf-8")
            )
            self.assertTrue((target / "schemas" / "schema.json").is_file())

    def test_preparation_rejects_target_outside_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            overlay = root / "suite.yaml"
            overlay.write_text("decode", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside allowed_root"):
                prepare(
                    allowed_root=root / "allowed",
                    source_spec_root=source,
                    target_spec_root=root / "target",
                    suite_overlay=overlay,
                )


if __name__ == "__main__":
    unittest.main()
