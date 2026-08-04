from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "remote"))
import aligned_l1_metrics as metrics  # noqa: E402


WORKLOADS = {
    "chat-1024-256": (1024, 256),
    "prefill-8192-512": (8192, 512),
    "balanced-1024-1024": (1024, 1024),
    "decode-256-2048": (256, 2048),
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def successful_metric(mean: float, total_sum: float | None = None) -> dict:
    return {
        "successful": {
            "mean": mean,
            "total_sum": mean if total_sum is None else total_sum,
            "percentiles": {"p50": mean, "p90": mean * 1.1},
        }
    }


def build_run(
    root: Path,
    *,
    wrong_shape: bool = False,
    servebench_1_layout: bool = False,
    c32_shortfall: int = 0,
) -> Path:
    datasets = []
    cases = []
    report_cases = []
    for workload, (input_tokens, output_tokens) in WORKLOADS.items():
        snapshot = f"datasets/{workload}.json"
        write_json(
            root / snapshot,
            {
                "shape": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            },
        )
        datasets.append({"workload": workload, "snapshot": snapshot})
        for concurrency, expected in ((1, 2), (16, 32), (32, 64)):
            actual_requests = (
                expected - c32_shortfall
                if workload == "decode-256-2048" and concurrency == 32
                else expected
            )
            config = f"core/{workload}/c{concurrency}/guidellm.yaml"
            summary = {
                "labels": {
                    "case_role": metrics.FORMAL_ROLE,
                    "workload": workload,
                },
                "profile": {"streams": [concurrency]},
                "constraints": [{"kind": "max_requests", "count": expected}],
            }
            cases.append({"config": config, "summary": summary})
            report_cases.append(
                {
                    "config": config,
                    "labels": summary["labels"],
                    "status": "completed",
                    "has_errors": False,
                    "request_totals": [
                        {
                            "successful": actual_requests,
                            "incomplete": 0,
                            "errored": 0,
                            "total": actual_requests,
                        }
                    ],
                }
            )
            prompt_mean = input_tokens + (
                1 if wrong_shape and workload == "chat-1024-256" and concurrency == 1 else 0
            )
            output_total = output_tokens * actual_requests
            native_metrics = {
                "prompt_token_count": successful_metric(prompt_mean),
                "output_token_count": successful_metric(
                    output_tokens, output_total
                ),
                "output_tokens_per_second": successful_metric(101.0),
                "time_to_first_token_ms": successful_metric(100.0),
                "time_per_output_token_ms": successful_metric(10.0),
            }
            write_json(
                root / Path(config).parent / "benchmarks.json",
                {
                    "benchmarks": [
                        {
                            "metrics": native_metrics,
                            "scheduler_metrics": {
                                "measure_start_time": 10.0,
                                "measure_end_time": 10.0 + output_total / 100.0,
                            },
                        }
                    ]
                },
            )
    manifest = {"datasets": datasets, "cases": cases}
    report = {
            "run_status": "completed",
            "suite": "01_调优_固定矩阵-v3.yaml",
            "execution_fingerprint": "fixture",
            "case_counts": {
                "failed": 0,
                "completed_unverified": 0,
                "cache_contaminated": 0,
                "cache_unverified": 0,
            },
            "cases": report_cases,
        }
    if servebench_1_layout:
        report["manifest"] = manifest
        write_json(root / "artifacts" / "manifest.json", manifest)
        write_json(root / "result.json", report)
    else:
        write_json(root / "manifest.json", manifest)
        write_json(root / "report.json", report)
    return root


class AlignedL1MetricsTests(unittest.TestCase):
    def test_consolidates_strict_aggregate_tps_and_twelve_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = build_run(Path(directory) / "rep1")
            result = metrics.consolidate([run], 32)
        self.assertEqual(result["parse_status"], "ok")
        self.assertEqual(result["l1"]["repetition_count"], 1)
        self.assertEqual(len(result["l1"]["cases"]), 12)
        self.assertAlmostEqual(
            result["l1"]["primary_aggregate_output_tps_geomean"],
            100.0,
        )
        self.assertEqual(result["metrics"]["successful_requests"], 392)

    def test_rejects_token_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = build_run(Path(directory) / "rep1", wrong_shape=True)
            with self.assertRaisesRegex(ValueError, "prompt token shape mismatch"):
                metrics.consolidate([run], 32)

    def test_servebench_1_result_layout_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = build_run(
                Path(directory) / "rep1",
                servebench_1_layout=True,
            )
            result = metrics.consolidate([run], 32)
        self.assertEqual("ok", result["parse_status"])
        repetition = result["l1"]["repetitions"][0]
        self.assertTrue(repetition["report_path"].endswith("result.json"))
        self.assertEqual(
            "embedded:result.json.manifest",
            repetition["manifest_path"],
        )

    def test_zero_error_one_request_c32_shortfall_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = build_run(
                Path(directory) / "rep1",
                servebench_1_layout=True,
                c32_shortfall=1,
            )
            result = metrics.consolidate([run], 32)
        self.assertEqual("ok", result["parse_status"])
        decode = next(
            case
            for case in result["l1"]["repetitions"][0]["cases"]
            if case["workload"] == "decode-256-2048"
            and case["concurrency"] == 32
        )
        self.assertEqual(63, decode["request_totals"]["successful"])
        self.assertAlmostEqual(
            63 / 64,
            decode["request_totals"]["completion_ratio"],
        )

    def test_more_than_two_percent_shortfall_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = build_run(
                Path(directory) / "rep1",
                servebench_1_layout=True,
                c32_shortfall=2,
            )
            with self.assertRaisesRegex(ValueError, "zero-error request gate"):
                metrics.consolidate([run], 32)

    def test_identifies_exactly_one_retryable_formal_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = build_run(Path(directory) / "rep1", servebench_1_layout=True)
            result_path = run / "result.json"
            report = json.loads(result_path.read_text(encoding="utf-8"))
            target = next(
                case
                for case in report["cases"]
                if case["config"].endswith("decode-256-2048/c16/guidellm.yaml")
            )
            target["request_totals"] = [
                {"successful": 31, "incomplete": 0, "errored": 0, "total": 31}
            ]
            write_json(result_path, report)

            retry = metrics.retryable_single_missing_case([run])

            self.assertIsNotNone(retry)
            self.assertEqual(retry["successful"], 31)
            self.assertEqual(retry["planned"], 32)

    def test_does_not_retry_two_missing_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = build_run(Path(directory) / "rep1", servebench_1_layout=True)
            result_path = run / "result.json"
            report = json.loads(result_path.read_text(encoding="utf-8"))
            target = next(
                case
                for case in report["cases"]
                if case["config"].endswith("decode-256-2048/c16/guidellm.yaml")
            )
            target["request_totals"] = [
                {"successful": 30, "incomplete": 0, "errored": 0, "total": 30}
            ]
            write_json(result_path, report)

            self.assertIsNone(metrics.retryable_single_missing_case([run]))


if __name__ == "__main__":
    unittest.main()
