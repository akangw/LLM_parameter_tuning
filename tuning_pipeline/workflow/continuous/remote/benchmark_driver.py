#!/usr/bin/env python3
"""Run a public vLLM benchmark or a project-allowlisted custom adapter."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_METRICS = (
    "successful_requests",
    "failed_requests",
    "output_token_throughput",
    "mean_ttft",
    "mean_tpot",
)


def _definition() -> dict[str, Any]:
    raw = base64.b64decode(os.environ["BENCHMARK_DEFINITION_B64"])
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("benchmark definition must be a JSON object")
    return value


def _identity() -> dict[str, Any]:
    value = json.loads(os.environ["BENCHMARK_IDENTITY_JSON"])
    if not isinstance(value, dict) or not re.fullmatch(
        r"[0-9a-f]{64}", str(value.get("sha256", ""))
    ):
        raise ValueError("missing or invalid frozen benchmark identity")
    return value


def _normalize_key(label: str) -> str:
    key = re.sub(r"\([^)]*\)", "", label.strip().lower())
    return re.sub(r"[^a-z0-9]+", "_", key).strip("_")


def _parse_number(raw: str) -> int | float | str:
    value = raw.strip().replace(",", "")
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", value):
        return float(value)
    return raw.strip()


def parse_vllm_output(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*([^:]{2,80}):\s*(.+?)\s*$", line)
        if not match:
            continue
        key = _normalize_key(match.group(1))
        if any(
            token in key
            for token in (
                "throughput",
                "ttft",
                "tpot",
                "itl",
                "latency",
                "duration",
                "successful_requests",
                "failed_requests",
                "input_tokens",
                "generated_tokens",
            )
        ):
            metrics[key] = _parse_number(match.group(2))
    return metrics


def validate_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        raise ValueError("adapter result.metrics must be an object")
    missing = [name for name in REQUIRED_METRICS if name not in metrics]
    if missing:
        raise ValueError(f"benchmark result is missing canonical metrics: {missing}")
    for name in REQUIRED_METRICS:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"canonical metric {name} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"canonical metric {name} must be finite and non-negative")
    return metrics


def _run_logged(command: list[str], log_path: Path, timeout: int) -> str:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="")
    if completed.returncode:
        raise RuntimeError(
            f"benchmark command exited with status {completed.returncode}"
        )
    return completed.stdout


def _vllm_command(definition: dict[str, Any], seed: int, prompts: int) -> list[str]:
    command = [
        "vllm",
        "bench",
        "serve",
        "--served-model-name",
        os.environ["SERVED_MODEL_NAME"],
        "--port",
        os.environ["SERVICE_PORT"],
        "--backend",
        str(definition.get("backend", "vllm")),
        "--dataset-name",
        "random",
        "--random-input-len",
        str(definition["input_tokens"]),
        "--random-output-len",
        str(definition["output_tokens"]),
        "--random-range-ratio",
        str(definition.get("random_range_ratio", 1.0)),
        "--request-rate",
        str(definition["request_rate"]),
        "--num-prompts",
        str(prompts),
        "--temperature",
        str(definition.get("temperature", 0)),
        "--seed",
        str(seed),
    ]
    if definition.get("ignore_eos", True):
        command.append("--ignore-eos")
    if definition.get("trust_remote_code", True):
        command.append("--trust-remote-code")
    return command


def run_public(run_dir: Path, definition: dict[str, Any]) -> dict[str, Any]:
    timeout = int(definition.get("timeout_seconds", 3600))
    warmup_prompts = int(definition.get("warmup_prompts", 0))
    if warmup_prompts:
        _run_logged(
            _vllm_command(definition, int(definition.get("warmup_seed", 24)), warmup_prompts),
            run_dir / "public_warmup.log",
            timeout,
        )
    text = _run_logged(
        _vllm_command(
            definition,
            int(definition.get("seed", 42)),
            int(definition["num_prompts"]),
        ),
        run_dir / "public_formal.log",
        timeout,
    )
    return {
        "metrics": validate_metrics(parse_vllm_output(text)),
        "raw_artifacts": ["public_formal.log"],
    }


def _safe_adapter(project_dir: Path, definition: dict[str, Any]) -> Path:
    relative = Path(str(definition["adapter_path"])); project = project_dir.resolve()
    adapter = (project / relative).resolve()
    roots = [(project / str(root)).resolve() for root in definition["allowlisted_roots"]]
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or adapter.suffix != ".py"
        or not adapter.is_file()
        or not any(adapter == root or adapter.is_relative_to(root) for root in roots)
    ):
        raise ValueError("custom adapter is absent or outside its frozen allowlist")
    return adapter


def run_custom(
    run_dir: Path, project_dir: Path, definition: dict[str, Any]
) -> dict[str, Any]:
    adapter = _safe_adapter(project_dir, definition)
    request = {
        "schema_version": "vllmtkb-benchmark-request/v1",
        "endpoint": f"http://127.0.0.1:{os.environ['SERVICE_PORT']}/v1",
        "served_model": os.environ["SERVED_MODEL_NAME"],
        "run_dir": str(run_dir),
        "config": definition.get("config", {}),
    }
    request_path = run_dir / "custom_benchmark_request.json"
    output_path = run_dir / "custom_adapter_output.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(adapter),
            "--request",
            str(request_path),
            "--output",
            str(output_path),
        ],
        cwd=project_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(definition.get("timeout_seconds", 3600)),
        check=False,
    )
    (run_dir / "custom_adapter.log").write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="")
    if completed.returncode:
        raise RuntimeError(f"custom adapter exited with status {completed.returncode}")
    result = json.loads(output_path.read_text(encoding="utf-8"))
    result["metrics"] = validate_metrics(result.get("metrics"))
    result.setdefault("raw_artifacts", ["custom_adapter.log", output_path.name])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("vllm_bench_serve", "custom_adapter"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    args = parser.parse_args()
    definition = _definition()
    result = (
        run_public(args.run_dir, definition)
        if args.mode == "vllm_bench_serve"
        else run_custom(args.run_dir, args.project_dir, definition)
    )
    payload = {
        **result,
        "schema_version": "vllmtkb-benchmark-result/v1",
        "benchmark_mode": args.mode,
        "benchmark_profile": os.environ["BENCHMARK_PROFILE"],
        "benchmark_identity": _identity(),
        "parse_status": "ok",
    }
    validate_metrics(payload["metrics"])
    destination = args.run_dir / "metrics.json"
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
