"""Repository-local deterministic, high-recall relevance filter."""

from __future__ import annotations

import re


PERFORMANCE_SIGNALS = re.compile(
    r"parallel|world[_-]?size|rank|batch|num[_-]?seq|prefill|decode|schedul|"
    r"token|cache|memory|offload|swap|compile|cudagraph|graph|fusion|kernel|"
    r"eager|quant|dtype|expert|moe|all2all|all[_-]?reduce|hccl|rdma|"
    r"communicat|attention|chunked|prefix[_-]?cach|speculat|load[_-]?(?:format|strategy)|"
    r"cpu[_-]?bind|throughput|latency|kv[_-]?|model[_-]?len|gpu[_-]?memory|"
    r"elastic|eplb|lora|multimodal|executor|worker|numa|inductor|autotune|"
    r"stream|buffer|timeout|backend|optimization|matmul|gemm|precision|layout|"
    r"prefetch|warmup|shm",
    re.IGNORECASE,
)
HARD_SKIP = re.compile(
    r"(?:^|[._-])(?:test|dummy|mock|credential|password|api[_-]?key|tokenizer|"
    r"download[_-]?dir|served[_-]?model|chat[_-]?template|request[_-]?id)(?:$|[._-])|"
    r"(?:^|[._-])(?:path|dir|host|port|endpoint)(?:$|[._-])",
    re.IGNORECASE,
)
NON_ASCEND = re.compile(
    r"(?:^|_)(?:ROCM|XPU|TPU|CUDA|NCCL|CUTLASS|MARLIN|AITER|FLASHINFER|XLA)(?:_|$)",
    re.IGNORECASE,
)
ASCEND_KEEP = re.compile(r"^(?:VLLM_ASCEND_|HCCL_|ASCEND_)|^additional_config\.", re.IGNORECASE)


def filter_parameters(
    params: list[dict], *, extra_keep_patterns: list[str] | None = None
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Return retained params, skipped audit records, and reason counts.

    The filter intentionally keeps public CLI options after hard exclusions so
    Stage 2, not a keyword-only heuristic, makes the final relevance decision.
    """
    passed, skipped, counts = [], [], {}
    extra_keep = (
        re.compile("|".join(f"(?:{item})" for item in extra_keep_patterns), re.IGNORECASE)
        if extra_keep_patterns
        else None
    )
    for param in params:
        name = str(param.get("name", ""))
        desc = str(param.get("description") or "")
        text = f"{name} {desc}"
        if extra_keep and extra_keep.search(name):
            keep, reason = True, "explicit_high_impact_keep"
        elif ASCEND_KEEP.search(name):
            keep, reason = True, "ascend_runtime_tuning_surface"
        elif param.get("type") == "env" and NON_ASCEND.search(name):
            keep, reason = False, "non_ascend_backend"
        elif HARD_SKIP.search(name):
            keep, reason = False, "non_tuning_control_surface"
        elif PERFORMANCE_SIGNALS.search(text):
            keep, reason = True, "performance_signal"
        elif param.get("type") == "cli":
            keep, reason = True, "user_facing_cli_review"
        else:
            keep, reason = False, "no_performance_signal"
        counts[reason] = counts.get(reason, 0) + 1
        if keep:
            passed.append(param)
        else:
            skipped.append({
                "name": name, "type": param.get("type"),
                "scope": param.get("scope"), "reason": reason,
            })
    return passed, skipped, dict(sorted(counts.items()))
