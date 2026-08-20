import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from .remote.validate_aligned_l1_inputs import (
    validate_suite_dataset_contract,
    validate_suite_schema,
)


class ValidateAlignedL1InputsTests(unittest.TestCase):
    def test_suite_schema_is_checked_before_model_launch(self) -> None:
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "阶段": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "最大错误数": {"type": "integer", "minimum": 1}
                        },
                    },
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite_path = root / "suite.yaml"
            schema_path = root / "schema.json"
            suite_path.write_text("阶段:\n- 最大错误数: 0\n", encoding="utf-8")
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            with self.assertRaisesRegex(
                SystemExit, "Benchmark suite schema validation failed"
            ):
                validate_suite_schema(suite_path, schema_path)
            suite_path.write_text("阶段:\n- 最大错误数: 1\n", encoding="utf-8")
            validate_suite_schema(suite_path, schema_path)

    def _fixture(self, root: Path, phase_id: str) -> tuple[Path, Path]:
        dataset_root = root / "datasets"
        manifest_dir = dataset_root / "workload-v1"
        split_dir = manifest_dir / "splits"
        split_dir.mkdir(parents=True)
        splits = {}
        for suffix, count in (("c32-warmup", 1), ("c32", 2)):
            split_path = split_dir / f"core__workload__{suffix}.jsonl"
            split_path.write_text("".join(f'{{"id": {i}}}\n' for i in range(count)), encoding="utf-8")
            splits[f"core-fixed-matrix/workload/c32" + ("-warmup" if suffix.endswith("warmup") else "")] = {
                "path": f"splits/{split_path.name}",
                "count": count,
                "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            }
        (manifest_dir / "manifest.json").write_text(
            json.dumps({"splits": splits}), encoding="utf-8"
        )
        suite_path = root / "suite.yaml"
        suite_path.write_text(
            yaml.safe_dump(
                {
                    "工作负载": [
                        {
                            "标识": "workload",
                            "数据": {"清单": "datasets/workload-v1/manifest.json"},
                        }
                    ],
                    "阶段": [
                        {
                            "标识": phase_id,
                            "工作负载": ["workload"],
                            "测试点": [
                                {"并发": 32, "预热请求数": 1, "正式请求数": 2}
                            ],
                        }
                    ],
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        return suite_path, dataset_root

    def test_matching_phase_verifies_warmup_and_formal_slices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            suite, datasets = self._fixture(Path(temporary), "core-fixed-matrix")
            self.assertEqual(2, validate_suite_dataset_contract(suite, datasets))

    def test_phase_drift_is_rejected_before_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            suite, datasets = self._fixture(Path(temporary), "fast-c32-primary")
            with self.assertRaisesRegex(SystemExit, "dataset slice missing"):
                validate_suite_dataset_contract(suite, datasets)


if __name__ == "__main__":
    unittest.main()
