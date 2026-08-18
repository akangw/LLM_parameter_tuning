#!/usr/bin/env python3
"""Validate and consolidate repeated ServeBench tuning-fixed L1 runs.

The original ServeBench results remain authoritative and untouched. This
sidecar adds a strict service aggregate throughput:

    successful output tokens / formal measurement wall-clock duration

and emits the compatibility metrics consumed by the tuning controller.
"""

from __future__ import annotations

import argparse
import json
import os
import math
import statistics
from pathlib import Path
from typing import Any


FORMAL_ROLE = "\u6b63\u5f0f\u6d4b\u91cf"


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def first_existing(run_dir: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = run_dir / name
        if path.is_file():
            return path
    rendered = ", ".join(str(run_dir / name) for name in names)
    raise FileNotFoundError(f"none of the compatible result files exist: {rendered}")


def finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def successful_metric(metrics: dict[str, Any], name: str) -> dict[str, Any]:
    metric = metrics.get(name)
    successful = metric.get("successful") if isinstance(metric, dict) else None
    if not isinstance(successful, dict):
        raise ValueError(f"missing successful metric: {name}")
    return successful


def metric_stat(metrics: dict[str, Any], name: str, field: str) -> float:
    successful = successful_metric(metrics, name)
    return finite_number(successful.get(field), f"{name}.successful.{field}")


def metric_percentile(
    metrics: dict[str, Any], name: str, percentile: str
) -> float:
    successful = successful_metric(metrics, name)
    percentiles = successful.get("percentiles")
    if not isinstance(percentiles, dict):
        raise ValueError(f"missing percentiles: {name}")
    return finite_number(
        percentiles.get(percentile),
        f"{name}.successful.percentiles.{percentile}",
    )


def formal_plan(
    manifest: dict[str, Any], expected_formal_cases: int = 12
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("manifest.cases must be a list")
    for case in cases:
        if not isinstance(case, dict):
            continue
        summary = case.get("summary")
        labels = summary.get("labels") if isinstance(summary, dict) else None
        if not isinstance(labels, dict) or labels.get("case_role") != FORMAL_ROLE:
            continue
        config = case.get("config")
        profile = summary.get("profile") if isinstance(summary, dict) else None
        streams = profile.get("streams") if isinstance(profile, dict) else None
        if isinstance(streams, list) and len(streams) == 1:
            streams = streams[0]
        if (
            not isinstance(config, str)
            or isinstance(streams, bool)
            or not isinstance(streams, int)
            or streams <= 0
        ):
            raise ValueError("formal case has invalid config or concurrency")
        expected = None
        constraints = summary.get("constraints")
        if isinstance(constraints, list):
            for constraint in constraints:
                if (
                    isinstance(constraint, dict)
                    and constraint.get("kind") == "max_requests"
                ):
                    expected = constraint.get("count")
        if (
            isinstance(expected, bool)
            or not isinstance(expected, int)
            or expected <= 0
        ):
            raise ValueError(f"formal case has no max_requests: {config}")
        workload = labels.get("workload")
        if not isinstance(workload, str) or not workload:
            raise ValueError(f"formal case has no workload: {config}")
        result[config] = {
            "workload": workload,
            "concurrency": streams,
            "expected_requests": expected,
        }
    if len(result) != expected_formal_cases:
        raise ValueError(
            "aligned L1 formal case count mismatch: "
            f"expected {expected_formal_cases}, got {len(result)}"
        )
    return result


def dataset_shapes(
    run_dir: Path, manifest: dict[str, Any]
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("manifest.datasets must be a non-empty list")
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("manifest dataset entry must be an object")
        workload, snapshot = dataset.get("workload"), dataset.get("snapshot")
        if not isinstance(workload, str) or not isinstance(snapshot, str):
            raise ValueError("manifest dataset is missing workload or snapshot")
        frozen = read_object(run_dir / snapshot)
        shape = frozen.get("shape")
        if not isinstance(shape, dict):
            raise ValueError(f"dataset snapshot is missing shape: {snapshot}")
        input_tokens = shape.get("input_tokens")
        output_tokens = shape.get("output_tokens")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in (input_tokens, output_tokens)
        ):
            raise ValueError(f"invalid dataset token shape: {snapshot}")
        result[workload] = {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
        }
    if len(result) != 4:
        raise ValueError(f"aligned L1 requires 4 workloads, got {len(result)}")
    return result


def load_native(run_dir: Path, config: str) -> dict[str, Any]:
    raw = read_object(run_dir / Path(config).parent / "benchmarks.json")
    benchmarks = raw.get("benchmarks")
    if (
        not isinstance(benchmarks, list)
        or len(benchmarks) != 1
        or not isinstance(benchmarks[0], dict)
    ):
        raise ValueError(f"invalid GuideLLM evidence for {config}")
    native = benchmarks[0]
    if not isinstance(native.get("metrics"), dict):
        raise ValueError(f"GuideLLM metrics missing for {config}")
    if not isinstance(native.get("scheduler_metrics"), dict):
        raise ValueError(f"GuideLLM scheduler metrics missing for {config}")
    return native


def request_gate(case: dict[str, Any], expected: int) -> dict[str, Any]:
    totals_list = case.get("request_totals")
    if not isinstance(totals_list, list) or len(totals_list) != 1:
        raise ValueError("formal case must have one request_totals record")
    totals = totals_list[0]
    if not isinstance(totals, dict):
        raise ValueError("request_totals entry must be an object")
    actual = {
        field: totals.get(field)
        for field in ("successful", "incomplete", "errored", "total")
    }
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in actual.values()
    ):
        raise ValueError(f"request totals must be integers: {actual}")
    minimum_successful = math.ceil(expected * 0.98)
    if (
        actual["incomplete"] != 0
        or actual["errored"] != 0
        or actual["successful"] != actual["total"]
        or not minimum_successful <= actual["successful"] <= expected
    ):
        raise ValueError(
            "zero-error request gate failed: "
            f"{actual}, planned={expected}, minimum_successful={minimum_successful}"
        )
    return {
        **{key: int(value) for key, value in actual.items()},
        "planned": expected,
        "minimum_successful": minimum_successful,
        "completion_ratio": actual["successful"] / expected,
    }


def retryable_single_missing_case(
    run_dirs: list[Path], expected_formal_cases: int = 12
) -> dict[str, Any] | None:
    """Locate one clean formal case that is short by exactly one request."""
    candidates: list[dict[str, Any]] = []
    for raw_run_dir in run_dirs:
        run_dir = raw_run_dir.resolve()
        report = read_object(first_existing(run_dir, ("result.json", "report.json")))
        embedded_manifest = report.get("manifest")
        if isinstance(embedded_manifest, dict):
            manifest = embedded_manifest
        else:
            manifest = read_object(
                first_existing(run_dir, ("artifacts/manifest.json", "manifest.json"))
            )
        if report.get("run_status") != "completed":
            return None
        counts = report.get("case_counts")
        if not isinstance(counts, dict) or any(
            counts.get(field) != 0
            for field in (
                "failed",
                "completed_unverified",
                "cache_contaminated",
                "cache_unverified",
            )
        ):
            return None
        plan = formal_plan(manifest, expected_formal_cases)
        report_cases = report.get("cases")
        if not isinstance(report_cases, list):
            return None
        formal_cases = {
            case.get("config"): case
            for case in report_cases
            if isinstance(case, dict)
            and isinstance(case.get("labels"), dict)
            and case["labels"].get("case_role") == FORMAL_ROLE
        }
        if set(formal_cases) != set(plan):
            return None
        for config, planned in plan.items():
            case = formal_cases[config]
            if case.get("status") != "completed" or case.get("has_errors") is not False:
                return None
            totals_list = case.get("request_totals")
            if not isinstance(totals_list, list) or len(totals_list) != 1:
                return None
            totals = totals_list[0]
            if not isinstance(totals, dict):
                return None
            values = {
                field: totals.get(field)
                for field in ("successful", "incomplete", "errored", "total")
            }
            if not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in values.values()
            ):
                return None
            expected = planned["expected_requests"]
            minimum = math.ceil(expected * 0.98)
            valid = (
                values["incomplete"] == 0
                and values["errored"] == 0
                and values["successful"] == values["total"]
                and minimum <= values["successful"] <= expected
            )
            if valid:
                continue
            retryable = (
                values["incomplete"] == 0
                and values["errored"] == 0
                and values["successful"] == values["total"]
                and expected - values["successful"] == 1
                and values["successful"] < minimum <= expected
            )
            if not retryable:
                return None
            candidates.append(
                {
                    "run_dir": str(run_dir),
                    "config": config,
                    "successful": values["successful"],
                    "planned": expected,
                    "minimum_successful": minimum,
                }
            )
    return candidates[0] if len(candidates) == 1 else None


def evaluate_repetition(
    run_dir: Path,
    primary_concurrency: int,
    expected_formal_cases: int = 12,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    # ServeBench <=0.7 wrote manifest.json/report.json at the result root.
    # ServeBench 1.0 writes artifacts/manifest.json and result.json.
    report_path = first_existing(run_dir, ("result.json", "report.json"))
    report = read_object(report_path)
    embedded_manifest = report.get("manifest")
    if isinstance(embedded_manifest, dict):
        manifest = embedded_manifest
        manifest_path = "embedded:result.json.manifest"
    else:
        resolved_manifest = first_existing(
            run_dir, ("artifacts/manifest.json", "manifest.json")
        )
        manifest = read_object(resolved_manifest)
        manifest_path = str(resolved_manifest)
    if report.get("run_status") != "completed":
        raise ValueError(f"run_status is not completed: {report.get('run_status')}")
    counts = report.get("case_counts")
    if not isinstance(counts, dict):
        raise ValueError("report.case_counts is missing")
    for field in (
        "failed",
        "completed_unverified",
        "cache_contaminated",
        "cache_unverified",
    ):
        if counts.get(field) != 0:
            raise ValueError(f"report.case_counts.{field} must be zero")

    plan = formal_plan(manifest, expected_formal_cases)
    shapes = dataset_shapes(run_dir, manifest)
    report_cases = report.get("cases")
    if not isinstance(report_cases, list):
        raise ValueError("report.cases must be a list")
    formal_cases = {
        case.get("config"): case
        for case in report_cases
        if isinstance(case, dict)
        and isinstance(case.get("labels"), dict)
        and case["labels"].get("case_role") == FORMAL_ROLE
    }
    if set(formal_cases) != set(plan):
        raise ValueError("formal report cases do not match the frozen plan")

    cases: list[dict[str, Any]] = []
    for config, planned in plan.items():
        case = formal_cases[config]
        if case.get("status") != "completed" or case.get("has_errors") is not False:
            raise ValueError(f"formal case did not complete cleanly: {config}")
        totals = request_gate(case, planned["expected_requests"])
        shape = shapes[planned["workload"]]
        native = load_native(run_dir, config)
        metrics = native["metrics"]
        prompt_mean = metric_stat(metrics, "prompt_token_count", "mean")
        output_mean = metric_stat(metrics, "output_token_count", "mean")
        if not math.isclose(prompt_mean, shape["input_tokens"], abs_tol=1e-6):
            raise ValueError(f"prompt token shape mismatch: {config}")
        if not math.isclose(output_mean, shape["output_tokens"], abs_tol=1e-6):
            raise ValueError(f"output token shape mismatch: {config}")
        scheduler = native["scheduler_metrics"]
        start = finite_number(scheduler.get("measure_start_time"), "measure_start_time")
        end = finite_number(scheduler.get("measure_end_time"), "measure_end_time")
        duration = end - start
        if duration <= 0:
            raise ValueError(f"non-positive formal duration: {config}")
        output_total = metric_stat(metrics, "output_token_count", "total_sum")
        strict_tps = output_total / duration
        cases.append(
            {
                **planned,
                "config": config,
                "request_totals": totals,
                "prompt_tokens_mean": prompt_mean,
                "output_tokens_mean": output_mean,
                "formal_duration_seconds": duration,
                "aggregate_output_tps": strict_tps,
                "guidellm_output_tps_mean": metric_stat(
                    metrics, "output_tokens_per_second", "mean"
                ),
                "ttft_p50_ms": metric_percentile(
                    metrics, "time_to_first_token_ms", "p50"
                ),
                "ttft_p90_ms": metric_percentile(
                    metrics, "time_to_first_token_ms", "p90"
                ),
                "tpot_p50_ms": metric_percentile(
                    metrics, "time_per_output_token_ms", "p50"
                ),
                "tpot_p90_ms": metric_percentile(
                    metrics, "time_per_output_token_ms", "p90"
                ),
                "speculative_decode_metrics": case.get(
                    "speculative_decode_metrics"
                ),
            }
        )
    primary_values = [
        case["aggregate_output_tps"]
        for case in cases
        if case["concurrency"] == primary_concurrency
    ]
    if len(primary_values) != len(shapes) or any(value <= 0 for value in primary_values):
        raise ValueError("primary concurrency must cover all workloads")
    score = math.exp(
        sum(math.log(value) for value in primary_values) / len(primary_values)
    )
    return {
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "manifest_path": manifest_path,
        "suite": report.get("suite"),
        "execution_fingerprint": report.get("execution_fingerprint"),
        "gate_passed": True,
        "formal_case_count": len(cases),
        "primary_concurrency": primary_concurrency,
        "primary_aggregate_output_tps_geomean": score,
        "cases": cases,
    }


def coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean * 100 if mean else math.inf


def consolidate(
    run_dirs: list[Path],
    primary_concurrency: int,
    expected_formal_cases: int = 12,
) -> dict[str, Any]:
    repetitions = [
        evaluate_repetition(path, primary_concurrency, expected_formal_cases)
        for path in run_dirs
    ]
    scores = [
        repetition["primary_aggregate_output_tps_geomean"]
        for repetition in repetitions
    ]
    case_keys = {
        (case["workload"], case["concurrency"])
        for case in repetitions[0]["cases"]
    }
    case_summary: list[dict[str, Any]] = []
    fields = (
        "aggregate_output_tps",
        "guidellm_output_tps_mean",
        "ttft_p50_ms",
        "ttft_p90_ms",
        "tpot_p50_ms",
        "tpot_p90_ms",
    )
    for workload, concurrency in sorted(case_keys):
        matches = [
            next(
                case
                for case in repetition["cases"]
                if case["workload"] == workload
                and case["concurrency"] == concurrency
            )
            for repetition in repetitions
        ]
        summary: dict[str, Any] = {
            "workload": workload,
            "concurrency": concurrency,
            "repetition_count": len(matches),
        }
        for field in fields:
            values = [float(case[field]) for case in matches]
            summary[field] = statistics.median(values)
            summary[f"{field}_values"] = values
            if field == "aggregate_output_tps":
                summary["aggregate_output_tps_cv_percent"] = (
                    coefficient_of_variation(values)
                )
        case_summary.append(summary)

    successful = sum(
        case["request_totals"]["successful"]
        for repetition in repetitions
        for case in repetition["cases"]
    )
    primary_cases = [
        case for case in case_summary if case["concurrency"] == primary_concurrency
    ]
    if not primary_cases:
        raise ValueError("no primary-concurrency cases are available for reporting")
    metrics = {
        "successful_requests": successful,
        "failed_requests": 0,
        # Compatibility key used by the existing comparison controller.
        "output_token_throughput": statistics.median(scores),
        "primary_score_cv_percent": coefficient_of_variation(scores),
        "ttft_p50_ms": statistics.median(
            [case["ttft_p50_ms"] for case in primary_cases]
        ),
        "ttft_p90_ms": statistics.median(
            [case["ttft_p90_ms"] for case in primary_cases]
        ),
        "tpot_p50_ms": statistics.median(
            [case["tpot_p50_ms"] for case in primary_cases]
        ),
        "tpot_p90_ms": statistics.median(
            [case["tpot_p90_ms"] for case in primary_cases]
        ),
    }
    return {
        "benchmark_mode": "aligned_l1",
        "parse_status": "ok",
        "metrics": metrics,
        "l1": {
            "suite": os.environ.get("BENCHMARK_SUITE_ID", "tuning-fixed"),
            "all_repetitions_gate_passed": True,
            "repetition_count": len(repetitions),
            "primary_concurrency": primary_concurrency,
            "primary_aggregate_output_tps_geomean": statistics.median(scores),
            "primary_score_values": scores,
            "primary_score_cv_percent": coefficient_of_variation(scores),
            "cases": case_summary,
            "repetitions": repetitions,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--primary-concurrency", type=int, default=32)
    parser.add_argument("--expected-formal-cases", type=int, default=12)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--retry-plan-output", type=Path)
    args = parser.parse_args()
    retry_plan: dict[str, Any] = {"retryable": False}
    try:
        if args.expected_formal_cases < 1:
            raise ValueError("expected formal case count must be positive")
        result = consolidate(
            args.run,
            args.primary_concurrency,
            args.expected_formal_cases,
        )
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        try:
            candidate = retryable_single_missing_case(
                args.run, args.expected_formal_cases
            )
        except (FileNotFoundError, ValueError, KeyError, TypeError):
            candidate = None
        if candidate is not None:
            retry_plan = {"retryable": True, **candidate}
        result = {
            "benchmark_mode": "aligned_l1",
            "parse_status": "failed",
            "metrics": {},
            "l1": {
                "all_repetitions_gate_passed": False,
                "repetition_count": len(args.run),
                "failures": [f"{type(exc).__name__}: {exc}"],
            },
        }
    result["schema_version"] = "vllmtkb-benchmark-result/v1"
    result["benchmark_profile"] = os.environ.get(
        "BENCHMARK_PROFILE", "aligned_l1_v4"
    )
    try:
        identity = json.loads(os.environ.get("BENCHMARK_IDENTITY_JSON", "{}"))
    except json.JSONDecodeError:
        identity = {}
    if identity:
        result["benchmark_identity"] = identity
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.retry_plan_output:
        args.retry_plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.retry_plan_output.write_text(
            json.dumps(retry_plan, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(rendered, end="")
    return 0 if result["parse_status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
