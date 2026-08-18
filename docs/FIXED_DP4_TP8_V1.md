# Fixed DP4/TP8 A8-Derived V1

## Scope

This package validates one frozen two-node DP4/TP8 topology without enabling
the outer topology Campaign. Each 16-NPU node owns two local DP ranks; every
rank is one TP8 replica. The total geometry is therefore DP=4, TP=8 over the
same 32 NPU allocation used by DP2/TP16.

The package is independent from the fixed DP2/TP16 route:

- Runtime Adapter: `glm52_w8a8_a3_dp4_tp8_a8_fixed_v1`;
- topology: `a3_dp4_tp8`;
- executor: `ktp_two_role_local_dp` / `distributed_local_dp_v1`;
- baseline: `a8_glm52_w8a8_dp4_tp8_fixed_v1`;
- runtime root: `runtime_fixed_dp4_tp8_v1`;
- Lease: `vllmtkb-auto-fixed-dp4tp8-v1-20260814-2x16npu`;
- output root: `lab_runs_fixed_dp4_tp8_v1`.

DP2 state, history, failures and best anchors are not eligible for this
Session. Both fixed Lease names block each other from owning the 32-NPU
allocation concurrently.

## A8-derived first baseline

The first DP4 measurement preserves the A8 model, 64K service length,
`max_num_batched_tokens=2048`, chunked prefill, prefix caching, async
scheduling, MTP enabled, MLAPO and the same compilation mode. Four parameters
are conservatively adapted for the larger TP8 weight shard:

| Parameter | DP2 A8 | DP4 first baseline |
| --- | ---: | ---: |
| `max_num_seqs` | 256 | 64 |
| `gpu_memory_utilization` | 0.92 | 0.90 |
| `num_speculative_tokens` | 3 | 1 |
| `max_cudagraph_capture_size` | 192 | 64 |

The DP4 capture list is `[16, 32, 48, 64]`. These reductions are startup
headroom, not final Search Limits. After model readiness and Fast-C32 pass, the
same 28 Active / 75 Reserve Agent strategy may raise capacity, MTP and graph
sizes inside the DP4-specific Session.

## Operator flow

From the server-autonomous repository root:

```bash
AUTO=tuning_pipeline/workflow/continuous/server_autonomous
bash "$AUTO/dp4_tp8.sh" dry-run
bash "$AUTO/dp4_tp8.sh" prepare-lease
bash "$AUTO/dp4_tp8.sh" preflight
bash "$AUTO/dp4_tp8.sh" start new
bash "$AUTO/dp4_tp8.sh" status
```

`dry-run` is zero-NPU. `prepare-lease` is the first command that creates the
isolated persistent Lease, so it must be run only when the operator explicitly
authorizes the live experiment. The first real A0 round is both the baseline
measurement and the DP4 live model-fit gate. No DP4 performance claim exists
until service readiness and all four Fast-C32 cases pass with throughput, TTFT
and TPOT evidence.
