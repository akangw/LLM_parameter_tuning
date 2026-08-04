#!/usr/bin/env python3
"""Fail fast on frozen Benchmark dependency drift before loading vLLM."""

import hashlib
import json
import os
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


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

print("Aligned L1 immutable-input fingerprint verified.")
