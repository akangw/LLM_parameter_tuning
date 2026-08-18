# Fixed DP2/TP16 Parameter-Only V4

## Current production-validation route

Topology selection is deliberately removed from the main autonomous loop while
the serving-parameter chain is validated end to end. New Sessions select the
integrated `glm52_w8a8_a3_a8_frontier_v3` Runtime Adapter, which admits only
`a3_dp2_tp16` and freezes:

- two physical nodes with 16 NPU each;
- DP=2 and TP=16, one DP replica per node;
- the proven `ktp_two_role` executor;
- the A8 expert baseline;
- `automatic_registry_a8_frontier_v3` with 28 Active and 75 Reserve axes;
- `hierarchical_agentic_frontier_v3` and Fast-C32.

The Agent sees `fixed_topology_session` and may not propose, compare or budget a
topology change. Mutable state uses `runtime_fixed_dp2_tp16_v4`, and the Lease
output uses `lab_runs_fixed_dp2_tp16_v4`; the completed legacy Session remains
untouched and cannot be resumed accidentally.

## Why DP2/TP16

DP2/TP16 is the only production-proven geometry for this exact GLM-5.2 W8A8
image and two-node A3 allocation. Each node hosts one TP16 replica, so tensor
parallel communication remains within a node and each NPU carries the smaller
TP16 weight shard. The existing A8 baseline, model-loading path, rank contract,
memory envelope and Benchmark evidence were all measured on this identity.

DP4/TP8 may eventually improve high-concurrency aggregate throughput, but each
replica uses only eight NPU and therefore roughly doubles the per-NPU model
shard before KV cache, graph capture and MTP memory are considered. It has
static executor support but no live model-fit evidence yet. DP1/TP32 requires a
cross-node TP executor that is not implemented.

## Dormant topology work

`topology_campaign.py`, its schema/tests, DP4/TP8 executor, topology-keyed
history and the DP4 probe baseline are retained unchanged. To restore topology
search later, an operator must select `glm52_w8a8_a3_topology_campaign_v4`, set
`topology_campaign.enabled: true`, choose a new Lease/output/runtime identity,
and pass DP4 live preflight. Fixed DP2 history must not seed the DP4 Session.

## Parameters that must be revisited after a topology change

Changing DP/TP alters weight-shard memory, KV-cache headroom, per-replica
concurrency, collective communication, graph shapes and MoE traffic. At minimum
recalibrate `gpu_memory_utilization`, `max_model_len`, `max_num_seqs`,
`max_num_batched_tokens`, long-prefill thresholds, graph capture sizes, MTP
depth/batching, prefix/chunked-prefill policy, expert parallel/balancing, and
Ascend communication/fusion switches. A topology change therefore always
starts a new Session with a topology-specific baseline and history identity.
