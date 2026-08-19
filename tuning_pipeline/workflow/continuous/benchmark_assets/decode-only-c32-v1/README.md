# Decode-only C32 V1

This is an isolated ServeBench suite for the fixed `decode-256-2048`, C32
workload. It reuses the immutable tokenizer, schema and decode dataset from the
validated Fast-C32 V2 asset, but it does not run or score the chat, prefill or
balanced workloads.

The primary optimization metric is aggregate output-token throughput. Every
formal report must also preserve TTFT P50/P90 and TPOT P50/P90, request counts,
and the zero-error/zero-incomplete gate.
