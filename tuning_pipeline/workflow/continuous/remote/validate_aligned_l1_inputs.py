#!/usr/bin/env python3
"""Fail fast on frozen Benchmark dependency drift before loading vLLM."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _dataset_manifest_path(dataset_root: Path, reference: str) -> Path:
    relative = Path(reference)
    if relative.parts and relative.parts[0] == "datasets":
        relative = Path(*relative.parts[1:])
    return dataset_root / relative


def validate_suite_dataset_contract(suite_path: Path, dataset_root: Path) -> int:
    """Verify every suite-selected warmup/formal split before loading vLLM."""
    suite_data = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    workloads = {
        item["标识"]: item for item in suite_data.get("工作负载", [])
    }
    requested_phase = os.environ.get("BENCHMARK_PHASE", "all")
    selected_phases = [
        item
        for item in suite_data.get("阶段", [])
        if requested_phase in ("", "all", item.get("标识"))
    ]
    if not selected_phases:
        raise SystemExit(f"Benchmark phase is absent from suite: {requested_phase}")

    verified = 0
    for phase in selected_phases:
        phase_id = phase["标识"]
        for workload_id in phase.get("工作负载", []):
            if workload_id not in workloads:
                raise SystemExit(
                    f"Benchmark workload is absent from suite: {workload_id}"
                )
            manifest_path = _dataset_manifest_path(
                dataset_root, workloads[workload_id]["数据"]["清单"]
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            splits = manifest.get("splits", {})
            for point in phase.get("测试点", []):
                concurrency = int(point["并发"])
                required = []
                if int(point.get("预热请求数", 0)) > 0:
                    required.append(
                        (f"{phase_id}/{workload_id}/c{concurrency}-warmup", int(point["预热请求数"]))
                    )
                if int(point.get("正式请求数", 0)) > 0:
                    required.append(
                        (f"{phase_id}/{workload_id}/c{concurrency}", int(point["正式请求数"]))
                    )
                for split_key, expected_count in required:
                    if split_key not in splits:
                        raise SystemExit(
                            f"Benchmark dataset slice missing: {split_key} ({manifest_path})"
                        )
                    split = splits[split_key]
                    split_path = manifest_path.parent / split["path"]
                    if int(split.get("count", -1)) != expected_count:
                        raise SystemExit(
                            f"Benchmark dataset slice count mismatch: {split_key} "
                            f"{split.get('count')} != {expected_count}"
                        )
                    if not split_path.is_file():
                        raise SystemExit(f"Benchmark dataset slice file missing: {split_path}")
                    if digest(split_path) != split["sha256"]:
                        raise SystemExit(f"Benchmark dataset slice fingerprint drift: {split_path}")
                    with split_path.open("r", encoding="utf-8") as handle:
                        actual_count = sum(1 for line in handle if line.strip())
                    if actual_count != expected_count:
                        raise SystemExit(
                            f"Benchmark dataset slice line count mismatch: {split_key} "
                            f"{actual_count} != {expected_count}"
                        )
                    verified += 1
    return verified


def main() -> None:
    expected = json.loads(os.environ["BENCHMARK_EXPECTED_FINGERPRINT_JSON"])
    servebench_root = Path(os.environ["SERVEBENCH_ROOT"])
    actual_version = subprocess.check_output(
        [str(servebench_root / "servebench"), "--version"], text=True
    ).strip().split()[-1]
    if actual_version != expected["servebench_version"]:
        raise SystemExit(
            f"ServeBench version drift: {actual_version} != {expected['servebench_version']}"
        )
    actual_commit = (servebench_root / "PINNED_COMMIT").read_text(encoding="utf-8").strip()
    if actual_commit != expected["servebench_commit"]:
        raise SystemExit(
            f"ServeBench commit drift: {actual_commit} != {expected['servebench_commit']}"
        )

    spec_root = Path(os.environ["BENCHMARK_SPEC_ROOT"])
    suite = spec_root / "suites" / expected["suite_file"]
    if digest(suite) != expected["suite_sha256"]:
        raise SystemExit(f"Suite fingerprint drift: {suite}")
    for relative, wanted in expected["schema_files_sha256"].items():
        path = spec_root / relative
        if digest(path) != wanted:
            raise SystemExit(f"Schema fingerprint drift: {path}")

    tokenizer = Path(os.environ["BENCHMARK_TOKENIZER"])
    for relative, wanted in expected["tokenizer_files_sha256"].items():
        path = tokenizer / relative
        if digest(path) != wanted:
            raise SystemExit(f"Tokenizer fingerprint drift: {path}")

    datasets = Path(os.environ["BENCHMARK_DATASET_ROOT"])
    for relative, wanted in expected["dataset_manifests_sha256"].items():
        path = datasets / relative
        if digest(path) != wanted:
            raise SystemExit(f"Dataset manifest fingerprint drift: {path}")

    verified_slices = validate_suite_dataset_contract(suite, datasets)
    print(
        "Aligned L1 immutable-input fingerprint and suite/dataset contract verified "
        f"({verified_slices} slices)."
    )


if __name__ == "__main__":
    main()
