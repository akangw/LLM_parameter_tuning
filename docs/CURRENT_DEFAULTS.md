# Current defaults

This file is the repository-level source of truth for new Sessions. Runtime
progress is deliberately excluded; a running Session always uses its frozen
`session_config.yaml` and `00_search_space/` bundle.

| Axis | New-Session default |
|---|---|
| Runtime Adapter | `glm52_w8a8_a3_dp4_tp8_search_v4` |
| Topology | `a3_dp4_tp8` (2 nodes × 16 NPU, DP4, local-DP2, TP8) |
| Baseline | `incumbent_glm52_w8a8_dp4_tp8_search_v4.yaml` (Guided-V4 a13 incumbent, 591.187 output tok/s) |
| Search Space | `automatic_registry_a8_frontier_v4` (30 Active + 73 Reserve) |
| Agent strategy | `hierarchical_agentic_guided_v5` |
| Benchmark | `aligned_fast_c32_v2` |
| Topology Campaign | disabled |
| Recovery | autonomous; pause only for a proven immutable external blocker |

Fast-C32-v2 is a fixed four-workload C32 measurement and reports output-token
throughput, TTFT P50/P90 and TPOT P50/P90. `target_benchmark_seconds: 600` is
planning metadata, not a hard timeout. The only outer stuck-run safety boundary
is `round_timeout_minutes`.

The default server lifecycle can be invoked directly or through
`server_autonomous/dp4_tp8_search_v4.sh`; both resolve to the same configuration and
runtime root. DP2/TP16, DP4 v1-v3, frontier-v3 and Topology Campaign V4 remain
integrated historical/explicit profiles. Their documentation describes those
frozen identities and does not override this file.

Git contains code, schemas, versioned assets and configuration. It excludes
API keys, PIDs, `state.json`, `state.json.previous`, logs and experiment output.
