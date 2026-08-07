#!/usr/bin/env python3
"""Fail-closed template for a vLLMTKB Executor Adapter v1 bridge.

Copy this file and replace the handlers with calls to Kubernetes, Slurm,
ordinary SSH, or another scheduler. Do not select this unimplemented template
in a real Session.
"""

from __future__ import annotations

import json
import sys
from typing import Any


API_VERSION = "vllmtkb-executor-adapter/v1"


def response(*, ok: bool, **fields: Any) -> dict[str, Any]:
    return {"api_version": API_VERSION, "ok": ok, **fields}


def handle(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("api_version") != API_VERSION:
        return response(ok=False, error="unsupported api_version")
    action = str(request.get("action", ""))
    context = request.get("context", {})
    payload = request.get("payload", {})
    adapter_config = request.get("adapter_config", {})
    if not all(
        isinstance(value, dict) for value in (context, payload, adapter_config)
    ):
        return response(ok=False, error="context, payload and adapter_config must be objects")

    # Implement the target scheduler here. The examples below show only the
    # response shapes; returning success without performing the stated check is
    # forbidden because it would bypass Controller safety gates.
    if action == "prepare":
        return response(ok=False, error="implement scheduler resource preparation")
    if action == "check_ready":
        return response(ok=False, error="implement read-only scheduler preflight")
    if action == "submit":
        # Required on success:
        # return response(ok=True, task_id="job-123", run_id="a0_20260807_120000")
        return response(ok=False, error="implement scheduler submission")
    if action == "snapshot":
        # Required on success:
        # return response(ok=True, snapshot={
        #     "status": "Running", "active_pods": 2,
        #     "terminal": False, "partial_failure": False,
        # })
        return response(ok=False, error="implement scheduler status query")
    if action == "stop":
        return response(ok=False, error="implement scheduler stop")
    if action == "wait_for_release":
        # Block/poll up to payload["timeout_seconds"] and return released=true
        # only after all resources belonging to payload["task_id"] are gone.
        return response(ok=False, error="implement release wait")
    if action == "start_benchmark":
        return response(ok=False, error="implement detached benchmark start")
    if action == "stop_partial":
        return response(ok=False, error="implement partial-task cleanup")
    return response(ok=False, error=f"unknown action: {action}")


def main() -> int:
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        result = handle(request)
    except Exception as exc:  # The Controller still treats ok=false as failure.
        result = response(ok=False, error=f"{type(exc).__name__}: {exc}")
    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    # Structured scheduler rejections use ok=false with exit zero so the
    # Controller can surface the returned error. Process-level failures may
    # still exit non-zero and are also treated as fatal.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
