#!/usr/bin/env python3
"""Minimal stdlib reference adapter; replace it with a real benchmark suite."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    config = request.get("config", {})
    count = int(config.get("num_prompts", 16))
    concurrency = int(config.get("concurrency", 4))
    max_tokens = int(config.get("max_tokens", 64))
    url = request["endpoint"].rstrip("/") + "/completions"

    def invoke(index: int) -> tuple[bool, int, float]:
        body = json.dumps(
            {
                "model": request["served_model"],
                "prompt": f"Request {index}: explain throughput tuning briefly.",
                "max_tokens": max_tokens,
                "temperature": 0,
            }
        ).encode("utf-8")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(
                    url, data=body, headers={"Content-Type": "application/json"}
                ),
                timeout=float(config.get("request_timeout_seconds", 600)),
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            elapsed = time.perf_counter() - started
            tokens = int(payload.get("usage", {}).get("completion_tokens", 0))
            return True, tokens, elapsed
        except Exception as exc:  # The result contract records failures explicitly.
            print(f"request {index} failed: {exc}")
            return False, 0, time.perf_counter() - started

    wall_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(invoke, range(count)))
    wall = time.perf_counter() - wall_started
    successful = [item for item in results if item[0]]
    total_tokens = sum(item[1] for item in successful)
    latencies_ms = [item[2] * 1000 for item in successful] or [0.0]
    tpot_ms = [item[2] * 1000 / item[1] for item in successful if item[1] > 0] or [0.0]
    output = {
        "metrics": {
            "successful_requests": len(successful),
            "failed_requests": count - len(successful),
            "output_token_throughput": total_tokens / wall if wall else 0.0,
            # Non-streaming reference: full response latency is a conservative TTFT.
            "mean_ttft": statistics.fmean(latencies_ms),
            "mean_tpot": statistics.fmean(tpot_ms),
        },
        "adapter_metadata": {
            "name": "example_http_adapter",
            "warning": "Reference only; use vllm_bench_public_v1 for standard measurements.",
        },
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
