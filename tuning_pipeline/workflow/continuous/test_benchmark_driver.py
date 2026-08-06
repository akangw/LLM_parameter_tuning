from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .remote import benchmark_driver


class BenchmarkDriverTests(unittest.TestCase):
    def test_vllm_output_is_normalized_to_canonical_metrics(self) -> None:
        parsed = benchmark_driver.parse_vllm_output(
            """
Successful requests: 128
Failed requests: 0
Output token throughput (tok/s): 321.5
Mean TTFT (ms): 12.5
Mean TPOT (ms): 4.25
"""
        )
        self.assertEqual(128, parsed["successful_requests"])
        self.assertEqual(321.5, parsed["output_token_throughput"])
        self.assertEqual(parsed, benchmark_driver.validate_metrics(parsed))

    def test_canonical_contract_fails_closed_on_missing_metrics(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing canonical metrics"):
            benchmark_driver.validate_metrics({"successful_requests": 1})

    def test_custom_adapter_must_resolve_below_allowlisted_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            allowed = project / "workflow" / "benchmark_adapters"
            allowed.mkdir(parents=True)
            adapter = allowed / "adapter.py"
            adapter.write_text("# test\n", encoding="utf-8")
            resolved = benchmark_driver._safe_adapter(
                project,
                {
                    "adapter_path": "workflow/benchmark_adapters/adapter.py",
                    "allowlisted_roots": ["workflow/benchmark_adapters"],
                },
            )
            self.assertEqual(adapter.resolve(), resolved)
            outside = project / "outside.py"
            outside.write_text("# test\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside its frozen allowlist"):
                benchmark_driver._safe_adapter(
                    project,
                    {
                        "adapter_path": "outside.py",
                        "allowlisted_roots": ["workflow/benchmark_adapters"],
                    },
                )


if __name__ == "__main__":
    unittest.main()
