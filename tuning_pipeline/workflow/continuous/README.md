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

## Executor adapters

The production default remains the built-in `ktp_lab` implementation. A new
Session may explicitly select `execution_mode: executor_adapter` and a
project-allowlisted JSON bridge. The bridge implements resource preparation,
readiness, submission, status, stop, and release for another scheduler while
the Controller retains candidate, Session, Benchmark, and failure authority.

The adapter source hash, non-secret configuration, capabilities, and API
version are frozen as `executor_identity`. Resume fails closed if that identity
changes. Existing Sessions and the current default never load an external
bridge. See `workflow/executor_adapters/README.md` and its separate overlay
example. Replacing the scheduler does not automatically validate a different
rank layout; that still requires an integrated Topology/Executor Profile and
Runtime Adapter.

## Search-Space profiles

The production V3 dispatcher selects `automatic_registry_decode_priority_v2`.
It freezes 25 Active axes, 75 Reserve axes and 42 Fixed contracts from tagged
portraits, pinned source evidence and deterministic Decode compatibility policy,
then imports only identity-matched A1-A15 history. The generic Search-V4 config
still selects `automatic_registry_a8_frontier_v4` (30 Active, 73 Reserve), and
`curated_registry_v1` remains an explicit alternative. A profile may be selected
only when creating a Session.

The effective candidate contains:

- 25 active tunable parameters and 75 auditable reserve parameters in the
  production Decode V3 compilation;
- `async_scheduling` as a coupled derived companion when the curated MTP token
  axis is enabled after B0;
- the remaining single-value runtime-contract parameters as fixed fields;
- a Session-specific Agent output schema generated from those exact fields;
- explicit runtime injection contracts for every active parameter.

The frozen evidence is written below the new Session:

```text
00_search_space/
  manual_search_limits.yaml
  search_space_profile.yaml
  registry.generated.yaml          # automatic profile only
  registry.audit.yaml              # automatic profile only
  search_space.compiled.yaml
  rotation_report.yaml
  agent_decision.schema.json
  failure_decision.schema.json
  parameter_portraits.full.yaml
  parameter_portraits.agent.yaml
  runtime_rules.yaml
```

The old `search_limits_mode` settings remain only for archived Session
compatibility. New Sessions use `search_space.profile`. The manual pool in
`config.yaml` is still frozen as runtime-contract values and audit evidence.

`--resume` and `--reanalyze-current` always load the archived
`session_config.yaml`, never the current global defaults. Therefore changing
the default mode cannot change the schema or limits of an already-running
Session.

Curated planned parameters require explicit entries in
`search_space.approved_planned_parameters`. Automatic parameters instead require
exact source identity, compatibility acceptance, a validated generic injection
contract, and a second deterministic candidate check before every submission.

Dynamic EPLB is deliberately outside the current search space. The pinned
Ascend runtime does not expose the required upstream CLI contract, so
`enable_eplb=false` and `eplb_num_redundant_experts=0` remain fixed runtime
fields. A candidate that attempts to enable EPLB is rejected before submission.

Each lease node requests 80 CPU, 800Gi memory, and 16 NPU. The current production
runtime is `glm52_w8a8_a3_dp4_tp8_decode_priority_v3`: DP4/TP8 is frozen before
Session creation, and the Agent receives a fixed-topology identity rather than
a list of topology candidates. The generic `config.yaml` retains Search V4 as
an explicit framework route; it is not the active production Session.

The completed `glm52_w8a8_a3_topology_campaign_v4`, `topology_campaign.py`,
DP2/TP16 and the topology Campaign remain integrated but dormant behind
explicit profiles and `topology_campaign.enabled: false`. Re-enabling the
Campaign requires a new isolated runtime root and Session; topology histories
are never mixed. See
`docs/FIXED_DP2_TP16_V4.md` and `docs/TOPOLOGY_CAMPAIGN_V4.md`.

The model checkpoint is on DTFS. The pinned vLLM background prefetcher divides
files by global DP*TP rank, which leaves each physical DP node with only part
of the page cache it needs. The workflow therefore freezes a node-blocking
transport: one process per node reads the complete checkpoint with 8 threads
and 16 MiB blocks, waits for completion, and then launches vLLM with lazy mmap
so the broken background prefetch cannot race model loading. Requested and
effective strategies, files, bytes, seconds, shards/s, and GiB/s are archived.

The production server-autonomous route uses `decode_priority_v3.sh`, a private
ignored lease overlay, and the isolated `runtime_decode_priority_v3_live` root.
It must not share a process slot with another dispatcher.

## Benchmark modes

`config.yaml` keeps multiple benchmark implementations:

- `decode_only_c32_v2` (production): one fixed `decode-256-2048` C32 workload,
  with output throughput as primary and TTFT/TPOT P50/P90 reporting;
- `aligned_fast_c32_v2` (generic Search-V4 default): the copied and immutable
  `tuning-fast-c32-v2` suite, four frozen C32 workloads with complete output
  throughput, TTFT and TPOT reporting. Its 600-second target is planning
  metadata and never truncates the fixed suite;
- `aligned_fast_c32_v1` (archived/opt-in): the prior fixed C32 suite;
- `aligned_l1_v4` (opt-in): the preserved 12-case C1/C16/C32 matrix.
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
`result.json` plus `artifacts/manifest.json` layout. It validates the frozen
profile-specific formal-case count, exact token shapes, zero errors/incomplete
requests and cache evidence.
Because GuideLLM can close a 64-request concurrent case at 63 successful
requests while ServeBench reports it complete, the request gate requires at
least 98% completion (64 -> 63; 32 and 2 remain exact) and records the ratio.
It additionally computes strict aggregate output TPS as successful output
tokens divided by the formal measurement window.

Under the default strategy, candidate acceptance requires all repetitions, a
noise-adjusted primary gain, TTFT/TPOT P50/P90 limits, and no more than 5% C32
throughput regression in any single workload. Small-sample P99 is not used.

The fast profile deliberately removes C1/C16 while preserving all four primary
C32 workload shapes. Reports expose output throughput, TTFT P50/P90, TPOT P50/P90
and benchmark wall time at the top level. The measured workload is allowed to
finish; only the outer round safety timeout can terminate a genuinely stuck run.

If a round ends without metrics, Codex performs a separate failure analysis:

- proven parameter validation/OOM failure: generate a corrected full candidate;
- transient platform/network/HCCL failure: retry the same candidate;
- image, dependency, runtime bug, benchmark bug, or unknown cause: collect full
  evidence and request a schema-constrained Agent recovery decision. Retry or a
  Search-Limits/Recovery-Registry correction remains automatic when evidence
  supports it; pause only for a proven immutable external dependency.

Failure recovery prompts, evidence, JSONL events, decisions, and retry/adjusted
parameters are retained in the failed round. Retry counters remain audit and
resource-safety signals; the production hard-terminal-only contract does not
turn an otherwise recoverable failure into a human pause merely because an old
soft counter was reached.

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

The production strategy is `decode_priority_agentic_v2`. The Controller selects
only the semantic layer during ordered List 2 and List 1.3 coverage, while the
Agent owns exact parameters, values, parameter count and justified companions;
cross-layer refinement is fully Agent-owned. It makes `max_model_len` a normal
Active axis, permits evidence-backed one-to-four parameter experiments, imports
the compatible A1-A15 history, and remembers hard failures as exact conditional
combinations. Optional List 1.2/1.1 changes have no Session quota but still need
mechanism/history evidence. `hierarchical_agentic_guided_v5` remains the generic
Search-V4 strategy for a separate Session.
`best_anchor_coverage_v2` remains an integrated opt-in strategy. It anchors each
proposal to the highest-scoring configuration that passed the deterministic
baseline-relative throughput and latency gate. Its evidence bundle includes
per-parameter tested and untested values, causal single-parameter observations,
and separately classified multi-parameter observations. Exploration prefers 2-3
independent changes; after a trusted improvement it narrows to 1-2. Rejected
branches remain evidence but do not silently become the base configuration.

`best_anchor_coverage_v3` is also integrated. It deterministically requires 2-3
independent changes during exploration and narrows local refinement to one grid
step. Both strategies still run the complete aligned-L1 matrix for every candidate;
the screening helpers in `hierarchical_strategy.py` are reserved for a future
reviewed Screen-to-Full state machine and cannot accept an improvement today.

`hierarchical_throughput_v1` remains the legacy W8A8 strategy for Sessions whose
primary objective is output-token throughput. It does **not** read A0 or any other Session's
candidates or metrics. Instead, it applies a general ordered curriculum: MTP and
its required scheduler/graph companions, expert-parallel communication, scheduler
capacity, compilation/graph capture, then Ascend communication refinement. This
prevents early rounds from being consumed by isolated low-impact tweaks.

The curriculum is a two-stage state machine, not a one-round-per-layer list. During
layered search, each family has an independent minimum/maximum successful-measurement
budget and an early-exit rule. MTP, MoE/communication, compilation, and Ascend
communication receive 1-2 measurements; scheduler/capacity receives 2-3. Failed or
incomplete runs do not consume those budgets. Once the minimum is met, a family exits
early when its best incremental output-throughput gain against the accepted entry
anchor is below -3%; otherwise it continues until its maximum budget. After all
families have been visited, cross-layer refinement ranks the observed families by
their best incremental gain, revisits the most promising family, and tests remaining
values or interactions with 1-2 independent parameter changes. Derived companion
fields required by a valid configuration do not count against that change budget.

In this profile TTFT/TPOT are still measured, archived, compared with the baseline,
and shown to the Agent, but threshold violations are advisory rather than an
acceptance veto. Output-throughput gain, the per-workload output-throughput floor,
benchmark completeness, exact token evidence, zero-error requirements, and the
noise/repetition checks remain hard gates. Because the profile and effective
measurement policy are frozen into `session_config.yaml`, selecting it for a new
Session cannot change an already-running Session or the default V2 behavior.

The provider and strategy are selected in `config.yaml` and frozen into each new
Session. `codex` is the default provider; `anthropic`, `openai_compatible`,
`deepseek`, and a structured-stdout `command` adapter are available through `agent.providers`.
Credentials are referenced only by environment-variable name. New Sessions may
override the selection with `-AgentProvider` and `-StrategyProfile`; resume/retry
reject overrides so an existing experiment cannot drift.

Benchmark selection is an independent frozen axis. `benchmark_profiles.yaml`
maps a stable profile name to one complete definition in `config.yaml`.
`decode_only_c32_v2` is the production profile selected by the V3 dispatcher.
`aligned_fast_c32_v2` remains the generic Search-V4 default; `aligned_fast_c32_v1`
and `aligned_l1_v4` remain opt-in.
A new Session may use `-BenchmarkProfile`; resume and
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
.\start_continuous.ps1 -NewSession -StrategyProfile hierarchical_throughput_v1
.\start_continuous.ps1 -NewSession -AgentProvider anthropic
.\start_continuous.ps1 -NewSession -AgentProvider deepseek `
  -BenchmarkProfile vllm_bench_public_v1 -SearchSpaceProfile automatic_registry_v1
```

To start a new Session from a previously completed B0 without rerunning the
baseline, pass `--reuse-baseline-session <session-directory>` to the Python
Controller. The import is accepted only when the Benchmark identity, image
manifest, topology, deployment contract, baseline definition, candidate schema,
and benchmark mode match. It copies round-000 parameters/runtime/metrics evidence,
but deliberately excludes the source Session's Agent analysis. The new Session
therefore analyzes the measured B0 with its newly frozen Strategy/Agent and submits
A1 directly. The source Session is never modified.

For a portable installation, create the Git-ignored `config.local.yaml` from
`config.local.example.yaml`; the start and prepare scripts auto-detect it. An
explicit alternative can be selected with `-Config <path>`.

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
