# Topology Campaign V4

> Status since 2026-08-14: integrated but dormant. The production validation
> route currently fixes DP2/TP16 and runs the parameter-only Controller. No
> Campaign budget is allocated unless an operator explicitly selects the V4
> runtime, enables `topology_campaign`, and supplies a new isolated runtime root.

## Outcome

V4 makes DP/TP a real Agent decision without mixing topology changes into a
serving-parameter Session. The global search is joint, while execution is nested:

1. The Controller hard-filters impossible model/resource/executor contracts.
2. The topology Agent orders feasible startup probes.
3. Every surviving topology receives an equal three-measurement Fast-C32 screen.
4. The Agent selects the incumbent/challenger and allocates each four-round
   competitive slice. The Controller requires at least one challenger round.
5. Before completion, the Agent must request a two-round final challenger
   verification and then select the winner.

Every topology owns an isolated Controller runtime root, Session, compiled
Active/Reserve set, history, conditional failures and best anchor. Only comparable
Benchmark summaries cross the topology boundary.

## Current topology set

| Profile | Geometry | Status | First baseline |
| --- | --- | --- | --- |
| `a3_dp2_tp16` | 2 nodes, DP2, local-DP1, TP16 | production incumbent | measured A8 expert |
| `a3_dp4_tp8` | 2 nodes, DP4, local-DP2, TP8 | experimental eligible | conservative A8-derived probe |
| `a3_dp1_tp32` | 2 nodes, DP1, TP32 | blocked | none |

DP4/TP8 is no longer rejected merely because it lacks prior measurements. Its
`distributed_local_dp_v1` executor is wired and statically validated, including
worker start rank 2. It receives exactly one frozen-baseline startup plus Fast-C32
probe. Model-fit, startup OOM or executor-contract failure marks that topology
infeasible without asking the inner Agent to spend recovery rounds.

DP1/TP32 remains blocked because the current two-role launcher distributes DP
ranks; it cannot create one tensor-parallel group spanning both nodes.

## Baselines

The production topology starts from
`a8_glm52_w8a8_expert_fast_v1.yaml`. DP4/TP8 never reuses that candidate blindly;
it starts from `a8_glm52_w8a8_dp4_tp8_probe_v1.yaml`, which preserves 64K context
and MTP but initially reduces `max_num_seqs` to 64, memory utilization to 0.90,
speculative depth to 1 and graph capture to 64.

This DP4 definition is a probe, not claimed performance evidence. It becomes an
inner tuning anchor only after service readiness and all Fast-C32 gates pass.

## Search Limits inside each topology

The current generated registry contains 142 parameters:

- 103 eligible tunable axes;
- 28 Active axes supplied to the Agent;
- 75 Reserve axes available for topology-keyed Session-boundary rotation;
- 39 fixed axes.

The effective candidate schema has 33 runtime parameters: 28 Active plus derived
or fixed baseline fields. `max_model_len` remains a normal Active grid
(`16K/32K/48K/64K`) as requested. Capacity interactions may jointly change
`max_model_len`, `gpu_memory_utilization`, `max_num_seqs` and
`max_num_batched_tokens` under exact-combination failure memory.

`speculative_config__disable_padded_drafter_batch` is now an executable high-risk
axis instead of a baseline-only value that was silently filtered. The Controller
allows its true mode only with `async_scheduling=false`; the normal MTP-on path
remains dominant.

## Agent and Controller authority

The topology Agent owns:

- topology probe order;
- incumbent and challenger identity;
- competitive budget allocation;
- whether evidence is mature enough for final verification;
- the final topology winner.

The Controller owns only hard boundaries:

- physical rank arithmetic, executor/model compatibility and static baseline identity;
- one topology per frozen Session;
- no overlapping Lease allocations;
- equal initial screening, challenger floor and total caps;
- Fast-C32 success, throughput, TTFT and TPOT evidence gates;
- crash-safe pending-decision recovery;
- mandatory final challenger verification.

The Controller deliberately does not replace a missing Agent decision with a
throughput argmax.

## Benchmark contract

Every measurement uses the immutable `aligned_fast_c32_v1` asset. The Benchmark
phase has a hard 600-second limit and writes output-token throughput, TTFT and
TPOT to `metrics.json`. Model startup and Lease readiness are accounted separately;
they cannot silently expand the Benchmark matrix.

## Autonomous operation

When explicitly enabled, the server-autonomous config runs V4. Existing service commands
detect this and execute `topology_campaign.py`; the prior single-Session path stays
available when `topology_campaign.enabled` is false.

Run zero-NPU/static plus live control-plane preflight before formal execution:

```bash
bash tuning_pipeline/workflow/continuous/server_autonomous/preflight.sh
```

Then use the existing managed service entrypoint. Campaign state is stored at:

```text
tuning_pipeline/workflow/continuous/server_autonomous/runtime/topology_campaign/campaign_state.json
```

Inner topology states are stored below `topology_campaign/sessions/<profile>/`.
The outer decision is persisted before a slice starts; a process restart resumes
that exact slice. A non-terminal legacy single-Session state blocks Campaign start
instead of being silently abandoned.

No formal NPU experiment is started by deployment, static validation or
`topology_campaign.py --check-only`.
