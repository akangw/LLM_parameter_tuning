# GLM-5.2 continuous tuning

This directory is the local control plane. The default execution mode uses one
persistent `ktp-lab` lease with two 16-NPU nodes. Individual rounds restart the
declared vLLM processes inside that lease without releasing and rescheduling
the resources.

Each session is archived as:

```text
experiments/glm52_continuous_<timestamp>/
  session_config.yaml
  round_000_a0/
    00_context/          fixed scenario and round manifest
    01_query/            query command and tag-filtered knowledge output
    02_parameters/       complete candidate, effective config and exact vLLM command
    03_submission/       submitted task YAML, task ID and submit output
    04_runtime/          Master/Worker/benchmark-runner logs and status
    05_results/          metrics.json or failure.json
    06_agent_analysis/   Agent prompt, events/stderr, analysis decision and next candidate
  round_001_a1/
  ...
```

The next round is started only after `metrics.json` exists and the Agent
decision passes strict whitelist, grid-distance, evidence, and constraint
validation. The Agent chooses the smallest defensible set allowed by the frozen strategy.
Multi-parameter proposals must identify a real interaction, provide evidence
for every changed parameter, and document the relevant constraint checks.
Independent guesses must remain separate experiments.
The Agent may return `stop_complete` when no useful, safe, untested change remains;
the controller then archives the decision without submitting another task.

## Search Limits modes

New Sessions default to `search_limits_mode: automated`. Before creating the
Session, the controller invokes the independent compiler in
`workflow/search_space_compiler`, reads the newest available completed-history
snapshot, applies bounded history-aware rotation, and freezes the result.

The effective candidate contains:

- 12 active tunable parameters selected by the compiler;
- one coupled derived parameter whose allowed values maintain graph invariants;
- four single-value runtime-contract parameters as fixed fields;
- a Session-specific Agent output schema generated from those exact fields;
- explicit runtime injection contracts for every active parameter.

The frozen evidence is written below the new Session:

```text
00_search_space/
  manual_search_limits.yaml
  search_space.compiled.yaml
  rotation_report.yaml
  agent_decision.schema.json
  failure_decision.schema.json
  parameter_portraits.full.yaml
  parameter_portraits.agent.yaml
  runtime_rules.yaml
```

The manual fallback pool remains in `config.yaml`. Set
`search_limits_mode: manual` to use it. Automated mode also copies that pool
into `manual_search_limits` in the frozen Session configuration.

`--resume` and `--reanalyze-current` always load the archived
`session_config.yaml`, never the current global defaults. Therefore changing
the default mode cannot change the schema or limits of an already-running
Session.

Planned parameters require an explicit entry in
`automated_search_limits.approved_planned_parameters`. If history-aware
rotation selects an unapproved Reserve parameter, Session creation fails
closed instead of silently executing it.

Dynamic EPLB is deliberately outside the current search space. The pinned
Ascend runtime does not expose the required upstream CLI contract, so
`enable_eplb=false` and `eplb_num_redundant_experts=0` remain fixed runtime
fields. A candidate that attempts to enable EPLB is rejected before submission.

Each lease node requests 80 CPU, 800Gi memory, and 16 NPU. The two-node
TP16/DP2 topology remains fixed across all tuning rounds.

The model checkpoint is on DTFS. The pinned vLLM build auto-detects only
NFS/Lustre for Safetensors prefetch, so the workflow explicitly freezes
`--safetensors-load-strategy=prefetch`, 8 prefetch threads, and a 16 MiB read
block. This replaces the observed 16-worker lazy-mmap access pattern without
changing any post-startup serving or benchmark parameter. Each run archives a
`startup_timeline.jsonl` so process-start-to-API-ready time can be compared.

This project uses the isolated persistent lease
`vllmtkb-418bd627-32c8cf190-glm52-a3-32npu`; it does not share the historical
0706 lease or its process slots.

## Benchmark modes

`config.yaml` keeps two benchmark implementations:

- `aligned_l1` (default): the read-only central
  `01_调优_固定矩阵-v3.yaml` / `tuning-fixed` standard, frozen JSONL datasets,
  C1/C16/C32, four workloads, and one complete repetition per tuning round.
- `legacy_random_32k1k`: the former `vllm bench serve` random 16K-48K input /
  500-1500 output, 8-prompt, 0.2-RPS path. It remains available only for
  historical reproduction.

The aligned runner reads the central ServeBench standard and datasets without
modifying them. Every repetition and all generated evidence are written below:

```text
/mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-418bd627-32c8cf190/
  workflow/auto/runs/<run-id>/servebench/
    config-rep1/
    results/l1-rep1/
```

The original ServeBench reports remain unchanged. `aligned_l1_metrics.py`
supports both the archived root-level layout and ServeBench 1.0's
`result.json` plus `artifacts/manifest.json` layout. It validates all 12 formal
cases, exact token shapes, zero errors/incomplete requests and cache evidence.
Because GuideLLM can close a 64-request concurrent case at 63 successful
requests while ServeBench reports it complete, the request gate requires at
least 98% completion (64 -> 63; 32 and 2 remain exact) and records the ratio.
It additionally computes strict aggregate output TPS as successful output
tokens divided by the formal measurement window.

Candidate acceptance requires all repetitions, a noise-adjusted primary gain,
TTFT/TPOT P50/P90 limits, and no more than 5% C32 throughput regression in any
single workload. Small-sample P99 is not used.

The one-repetition policy is a continuous-search policy, not a reduced matrix:
each round still executes all 12 formal cases and their 12 warmups. It is based
on the archived A0 calibration whose first two complete repetitions produced
535.26 and 527.32 output tok/s (1.06% primary-score CV). The 3% minimum gain,
zero-error gate, per-workload throughput floor, and latency guardrails remain.

If a round ends without metrics, Codex performs a separate failure analysis:

- proven parameter validation/OOM failure: generate a corrected full candidate;
- transient platform/network/HCCL failure: retry the same candidate;
- image, dependency, runtime bug, benchmark bug, or unknown cause: pause for
  human review instead of making an unsafe change.

Failure recovery prompts, evidence, JSONL events, decisions, and retry/adjusted
parameters are retained in the failed round. Three repeated infrastructure
failures trigger a safety pause.

The controller also detects terminal KTP states when `MASTER_DONE` is missing,
enforces `round_timeout_minutes`, prevents concurrent controller instances and
overlap with a task referenced by the previous state, verifies explicit remote
return-code markers, and size-checks downloaded artifacts.

Agent analysis receives a controller-built read-only evidence bundle containing
the candidate, verified image digest and source commits, submitted task,
effective command, metrics, comparison, deterministic measurement assessment,
current-session history, relevant logs, tag knowledge, the compact natural-
language portraits for every active Search Limits parameter, and the frozen
runtime-rule state. Every actual Agent change archives a fresh full portrait
recall for the changed parameters and their one-hop relations.

The default strategy is `best_anchor_coverage_v2`. Each new proposal is anchored
to the highest-scoring configuration that passed the deterministic baseline-relative
throughput and latency gate. The evidence bundle includes per-parameter tested and
untested values, causal single-parameter observations, and separately classified
multi-parameter observations. Exploration prefers 2-3 independent changes; after a
trusted improvement it narrows to 1-2. Rejected branches therefore remain evidence
but do not silently become the base configuration for subsequent rounds.

`best_anchor_coverage_v3` is also integrated. It deterministically requires 2-3
independent changes during exploration and narrows local refinement to one grid
step. Both strategies still run the complete aligned-L1 matrix for every candidate;
the screening helpers in `hierarchical_strategy.py` are reserved for a future
reviewed Screen-to-Full state machine and cannot accept an improvement today.

The provider and strategy are selected in `config.yaml` and frozen into each new
Session. `codex` is the default provider; `anthropic`, `openai_compatible`, and a
structured-stdout `command` adapter are available through `agent.providers`.
Credentials are referenced only by environment-variable name. New Sessions may
override the selection with `-AgentProvider` and `-StrategyProfile`; resume/retry
reject overrides so an existing experiment cannot drift.

Benchmark selection is an independent frozen axis. `benchmark_profiles.yaml`
maps a stable profile name to one complete definition in `config.yaml`.
`aligned_l1_v4` is the formal default and `legacy_random_32k1k` is retained for
historical reproduction. A new Session may use `-BenchmarkProfile`; resume and
retry reject Benchmark overrides so measurements from different contracts are
never silently mixed.

Before every submission, the candidate must pass the frozen runtime rule store
in addition to the existing Controller checks. Completed history is fed back
conservatively: a single-parameter `parameter_invalid`/`parameter_oom` failure
quarantines that exact scoped value; generalized or multi-parameter failures
remain non-blocking proposals until explicitly activated.
The Codex subprocess runs in a read-only sandbox. It may use local read-only
commands for extra inspection but cannot edit files, use the network, execute
remote commands, or submit/stop jobs.

## Persistent lease and sliced MTP checkpoint

Set the exact sliced MTP checkpoint supplied by the deployment owner:

```yaml
# config.yaml
execution_mode: ktp_lab
mtp_draft_model: /exact/server/path/to/the/sliced-mtp-checkpoint
```

The controller fails closed when speculative decoding is enabled and this path
is empty. It passes the path explicitly as `speculative_config.model`, avoiding
a second scan of all main-model checkpoint shards.

The lease name is versioned with the runtime image. The repository YAML files
are templates: before synchronization, the Controller renders their lease name,
runtime image, and command paths from `config.yaml` and the verified image
manifest. Updating the template does not mutate Pods in an already-created
lease, so every image change requires a new lease name and a newly prepared
lease. Controller state also records the verified image digest plus the vLLM
and vLLM Ascend commits; `--resume` is rejected when that identity is missing
or differs. All submitting entry points also require `activation.approved.yaml`
to match the exact manifest identity.

Synchronize the managed scripts and create the persistent two-node lease once:

```powershell
.\prepare_lab.ps1
```

`prepare_lab.ps1` is the only workflow command that allocates the lease. Normal
round transitions use `ktp-lab run --lease ...`; the resource-admission
protocol waits only for both lease workers to acknowledge the start request and
then keeps the lease alive. The lease definition is `remote/lease_loop.yaml`.

## Validate without using NPU

```powershell
python .\continuous_tuning.py --dry-run
```

## Start

```powershell
.\start_continuous.ps1

# Optional new-Session selections
.\start_continuous.ps1 -NewSession -StrategyProfile best_anchor_coverage_v3
.\start_continuous.ps1 -NewSession -AgentProvider anthropic
.\start_continuous.ps1 -NewSession -BenchmarkProfile legacy_random_32k1k
```

Use `-Foreground` if you want the controller attached to the current terminal.

Resume the task/run recorded in `state.json` after a local controller restart:

```powershell
.\start_continuous.ps1 -Resume
```

Re-run Agent analysis from the archived current-round metrics without launching
a new experiment:

```powershell
.\reanalyze_current.ps1
```

The validated decision is saved and reused by the next `-Resume`.

## Status

```powershell
.\status_continuous.ps1
```

## Stop

Graceful stop keeps collecting the active experiment until it finishes,
archives the resulting success or failure artifacts, and prevents submission
of the next round:

```powershell
.\stop_continuous.ps1
```

When immediate termination is explicitly required, the task recorded in the
frozen Session can also be stopped. The command uses that Session's server,
project path, execution mode, and Lease identity:

```powershell
.\stop_continuous.ps1 -StopActiveTask
```

No stop operation deletes task records, logs, metrics, or archived files.

## Bounded benchmark recovery

Aligned-L1 rounds use layered, fail-closed recovery while the same vLLM service
is alive:

1. Frozen ServeBench, suite, schema, tokenizer, and dataset fingerprints are
   checked before task submission and again after `SERVICE_READY`.
2. A remote watchdog and the local Controller share an atomic
   `BENCHMARK_START_LOCK`, so a local Controller outage cannot strand a ready
   service or start the benchmark twice.
3. A non-zero ServeBench run may retry the complete repetition in a fresh
   result directory. A clean single-case one-request shortfall may retry only
   that case. A metrics-compilation failure may rerun all repetitions in fresh
   directories. Every result must still pass the strict metrics gate.
4. Retry budgets are frozen in `candidate.env`; runtime and full-matrix
   recovery also share a total full-rerun cap, preventing independent layers
   from multiplying into an unexpectedly large benchmark bill. Exhausted recovery always
   writes `BENCHMARK_FAILED`, allowing the service and Controller to terminate
   or classify the round instead of waiting until the global timeout.
5. Controller-level same-candidate retries remain capped at two. Parameter
   changes are allowed only for proven parameter-invalid or parameter-OOM
   failures; unknown or unsafe failures pause for human review.
