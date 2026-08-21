from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "remote" / "validate_mtp_model.py"


class ValidateMtpModelTests(unittest.TestCase):
    def run_validator(
        self,
        root: Path,
        *,
        tokens: int,
        frozen: Path,
        output: Path,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--model-path",
                str(root),
                "--num-speculative-tokens",
                str(tokens),
                "--identity-output",
                str(output),
                "--frozen-identity",
                str(frozen),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_validates_n_predict_and_freezes_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(
                json.dumps({"model_type": "test", "n_predict": 3}),
                encoding="utf-8",
            )
            frozen = root / "frozen.json"
            output = root / "identity.json"
            valid = self.run_validator(
                root, tokens=3, frozen=frozen, output=output
            )
            self.assertEqual(0, valid.returncode, valid.stderr)
            identity = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(3, identity["n_predict"])
            self.assertTrue(identity["compatible"])

            invalid_k = self.run_validator(
                root, tokens=4, frozen=frozen, output=root / "invalid.json"
            )
            self.assertNotEqual(0, invalid_k.returncode)
            self.assertIn("must be divisible", invalid_k.stderr)

    def test_rejects_model_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            config.write_text(json.dumps({"n_predict": 1}), encoding="utf-8")
            frozen = root / "frozen.json"
            first = self.run_validator(
                root, tokens=3, frozen=frozen, output=root / "first.json"
            )
            self.assertEqual(0, first.returncode, first.stderr)
            config.write_text(
                json.dumps({"n_predict": 1, "revision": 2}), encoding="utf-8"
            )
            drift = self.run_validator(
                root, tokens=3, frozen=frozen, output=root / "drift.json"
            )
            self.assertNotEqual(0, drift.returncode)
            self.assertIn("MTP_IDENTITY_DRIFT", drift.stderr)


if __name__ == "__main__":
    unittest.main()
