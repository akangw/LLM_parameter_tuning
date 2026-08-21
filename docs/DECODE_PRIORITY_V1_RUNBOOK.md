# Decode Priority V1

## Frozen experiment identity

- Runtime: `glm52_w8a8_a3_dp4_tp8_decode_priority_v1`
- Topology: DP4/TP8, 2×16 A3 NPU
- Baseline: `expert_decode_glm52_w8a8_dp4_tp8_v1.yaml`
- Search Limits: `automatic_registry_decode_priority_v2`
- Agent strategy: `decode_priority_agentic_v1`
- Benchmark: `decode_only_c32_v1`
- Workload: `decode-256-2048`, concurrency 32, 64 formal requests
- Primary score: aggregate output-token throughput
- Required reference metrics: TTFT P50/P90 and TPOT P50/P90

The package is isolated from Search V4 and does not resume or mutate the paused
Session. Its history seed is intentionally empty; prior decode evidence is
encoded in the expert baseline and strategy priors without mixing old
four-workload scores into the new benchmark regime.

The only supported operational entrypoint is
`server_autonomous/decode_priority_v1.sh`. It freezes the dedicated config and
runtime root so the generic Search-V4 defaults cannot be selected accidentally.
Code, benchmark assets, Controller state and experiment output are contained by
`/mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-decode-priority-v1`.

## Autonomous flow

1. Submit the expert baseline unchanged.
2. If startup or benchmark fails, collect evidence and run the existing
   deterministic/Agent recovery chain. Recovery may change any validated Active
   parameter or an evidence-gated Recovery Registry field;
   normal List 2/List 1 phase limits do not apply to failure repair.
3. After a healthy baseline, spend the main budget on List 2:
   capacity geometry, MTP/graph shapes, then scheduler-capacity/KV refinement.
4. Necessarily explore List 1.3 with two to three compact successful
   measurements. The Agent chooses the parameters, values and whether to use
   List 2 companions; coupling hints are not a whitelist.
5. Enter Agent-owned cross-layer refinement. List 2 remains the main budget and
   promising List 1.3 interactions may continue. List 1.2 and then List 1.1 are
   optional: the Agent decides whether evidence justifies them, without a
   Controller-enforced Session quota. A reasoned secondary parameter may also
   accompany an ordered-layer experiment; it cannot replace the required
   in-layer change.
6. Every high-risk proposal must state a mechanism hypothesis, history or
   portrait evidence, passed constraints and expected payoff. Random novelty
   without a causal argument is not accepted.
7. Normal completion requires all ordered List 2/List 1.3 stages plus eight
   successful cross-layer measurements. This is a minimum evidence gate, not an
   automatic stop: the Agent continues while meaningful untested hypotheses or
   frontier uncertainty remain. Failed or incomplete rounds do not satisfy it.

## Start gate

The package is intentionally not running. Before launch:

1. Synchronize the reviewed local payload to the isolated server project
   `vllmtkb-decode-priority-v1`.
2. Create a fresh Lease identity without editing tracked configuration. Write
   the real name into the Git-ignored server-local overlay
   `config.dp4_tp8.decode_priority_v1.local.yaml`, which must extend
   `config.dp4_tp8.decode_priority_v1.yaml`. The tracked file permanently keeps
   its fail-closed `-pending` value.
3. Run `prepare_decode_only_benchmark.py` to copy the validated Fast-C32 V2
   schema/tokenizer spec tree into a new decode-only spec root and overlay the
   new suite. The operation is additive and does not delete or edit Fast-C32 V2.
4. Run offline preparation/preflight and verify the frozen suite hash.
5. Start a new Session only after an explicit user instruction.

All lifecycle commands must use the dedicated dispatcher, for example:

```bash
bash tuning_pipeline/workflow/continuous/server_autonomous/decode_priority_v1.sh preflight
bash tuning_pipeline/workflow/continuous/server_autonomous/decode_priority_v1.sh service systemd-start
bash tuning_pipeline/workflow/continuous/server_autonomous/decode_priority_v1.sh status
```

正式无人值守运行只允许使用 systemd user service；如果主机不支持 user
systemd，可先安装并使用 `service supervisor-start`。Decode 专用入口会拒绝
旧的 `start` 后台方式，避免 Controller 自身异常后无人拉起。

The server-local Lease overlay has this minimal shape:

```yaml
base_config: config.dp4_tp8.decode_priority_v1.yaml
lab:
  lease_name: REAL_FRESH_LEASE_NAME
```

## V2 continuation after A15

The reviewed continuation uses `decode_priority_v2.sh` and the isolated
`runtime_decode_priority_v2_live` root. It starts from measured best anchor
`round_012_a10f1` rather than the inferior final A15 branch, and freezes all 28
compatible attempted-history entries in `decode_priority_history_seed_v2.json`.
The V1 Session remains immutable. V2 remeasures A10F1 as its own baseline, then
lets the updated Agent strategy explore List 2 and evidence-backed List 1
companions without the obsolete four-measurement secondary quota.
