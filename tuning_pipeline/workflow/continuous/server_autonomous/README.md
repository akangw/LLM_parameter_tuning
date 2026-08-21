# Server-autonomous mode

This mode runs the knowledge query, deterministic Controller, DeepSeek Agent,
ktp-lab submission, Benchmark collection, and Session archive on the Linux
server.  It is isolated from the default Windows-to-server chain.

The current production decode-only A10F1 continuation uses `decode_priority_v2.sh`,
`config.dp4_tp8.decode_priority_v2.yaml`, and the isolated
`runtime_decode_priority_v2_live` root. It imports the complete compatible V1
attempted history and never mutates the frozen V1 Session.

The generic `config.yaml` reference still defines one fixed DP4/TP8 Session with the measured
Guided-V4 incumbent baseline, `automatic_registry_a8_frontier_v4`,
`hierarchical_agentic_guided_v5`, and Fast-C32-v2 benchmark. The topology
Campaign remains installed but dormant. That generic route is not the current
production Session and must not be used to manage the same Lease.

`dp4_tp8_search_v4.sh` is the explicit dispatcher for the generic V4 identity.
The older DP4 v1-v3 dispatchers and fixed DP2 package remain reproducible
historical entrypoints. None shares V2 runtime state.

## Isolation contract

- Existing defaults remain `windows_remote + paramiko + codex`.
- This mode must explicitly load `server_autonomous/config.yaml` and its own
  runtime root.
- Its repository, mutable state, output root, and Lease name are distinct.
- Local transport is rejected unless `operation_mode=server_autonomous` and all
  writable runtime paths remain under `autonomous.allowed_write_root`.
- The declared main-chain Leases are checked before Lease creation and every
  submission.  If one is active, autonomous execution fails closed.
- Existing model, image, and Benchmark inputs are read-only. `seed_assets.sh`
  copies immutable Benchmark assets into the new workspace without deleting or
  modifying the source tree.

## Server path

```text
/mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/
└── vllmtkb-decode-priority-v1/
```

## First deployment

For a different server or writable root, create the private overlay before any
preflight or service command.  It is ignored by Git and auto-detected by all
server-autonomous entrypoints:

```bash
AUTO=tuning_pipeline/workflow/continuous/server_autonomous
cp "$AUTO/config.local.example.yaml" "$AUTO/config.local.yaml"
vi "$AUTO/config.local.yaml"
```

An absolute alternate file may instead be selected with
`VLLMTKB_CONFIG=/absolute/path/config.yaml`.  The checked-in `config.yaml`
remains the verified reference environment and should not be edited just to
store an operator's paths.

From Windows, deploy the repository snapshot with:

```powershell
.\scripts\deploy-server-autonomous.ps1
```

The command above preserves the verified server defaults.  For another Linux
controller, pass both its SSH alias and the exact writable boundary; the
deployment refuses a target outside that boundary:

```powershell
.\scripts\deploy-server-autonomous.ps1 `
  -RemoteHost my-controller `
  -AllowedWriteRoot /srv/my-user `
  -RemoteRoot /srv/my-user/LLM_parameter_tuning
```

On the server:

```bash
cd /mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-decode-priority-v1
export DEEPSEEK_API_KEY='...'

python3 -m pip install -r tuning_pipeline/requirements-server-autonomous.txt
# Required only by the verified private aligned_l1_v4 profile.  Public
# vllm_bench_public_v1 and custom Benchmark adapters do not use this source.
AUTO=tuning_pipeline/workflow/continuous/server_autonomous
bash "$AUTO/decode_priority_v2.sh" seed-assets
bash "$AUTO/decode_priority_v2.sh" dry-run
bash tuning_pipeline/workflow/continuous/server_autonomous/decode_priority_v2.sh service authorize-new-session
```

`dry_run.sh` performs local generation and validation only. It does not query,
create, or submit work to any Lease.  It deliberately leaves terminal audit
state, so `authorize-new-session` archives that evidence before a real Session.

After the operator has confirmed that every `blocked_lease_names` entry is no
longer active, create the isolated persistent Lease:

```bash
bash "$AUTO/decode_priority_v2.sh" prepare-lease
bash "$AUTO/decode_priority_v2.sh" preflight
```

`preflight.sh` validates the fixed DP4/TP8 runtime and idle Lease without
submitting a candidate. Mutable state lives in
`runtime_decode_priority_v2_live/`, so legacy runtime roots
cannot cause an accidental resume. When the dormant Campaign is explicitly
re-enabled later, it must receive another isolated runtime root.

Start or resume the current Decode Priority V2 Controller:

```bash
AUTO=tuning_pipeline/workflow/continuous/server_autonomous
bash "$AUTO/decode_priority_v2.sh" service supervisor-start
bash "$AUTO/decode_priority_v2.sh" service supervisor-status
bash "$AUTO/decode_priority_v2.sh" status
```

The generic V4 dispatcher remains available only for a separate Session and Lease:

```bash
AUTO=tuning_pipeline/workflow/continuous/server_autonomous
bash "$AUTO/dp4_tp8_search_v4.sh" dry-run
bash "$AUTO/dp4_tp8_search_v4.sh" prepare-lease
bash "$AUTO/dp4_tp8_search_v4.sh" preflight
bash "$AUTO/dp4_tp8_search_v4.sh" start new
bash "$AUTO/dp4_tp8_search_v4.sh" status
```

The production V2 baseline exact values are owned by
`expert_decode_glm52_w8a8_dp4_tp8_a10f1_v2.yaml`; documentation does not duplicate
them. `prepare-lease` is intentionally not part of deployment or offline validation;
run it only when a new live experiment is explicitly authorized.

## Persistent service (recommended)

`start.sh` remains available for compatibility. For unattended operation, use
the foreground runner through either systemd or Supervisor. These service files
belong only to server-autonomous mode and do not modify the Windows-to-server
main chain.

First create the private environment file and replace its placeholder key:

```bash
AUTO=tuning_pipeline/workflow/continuous/server_autonomous
bash "$AUTO/decode_priority_v2.sh" service prepare-env
vi "$AUTO/.secrets/controller.env"
chmod 600 "$AUTO/.secrets/controller.env"
```

### systemd user service (preferred)

This provides crash recovery and, with user lingering enabled, reboot recovery:

```bash
bash "$AUTO/decode_priority_v2.sh" service systemd-install
bash "$AUTO/decode_priority_v2.sh" service systemd-start
bash "$AUTO/decode_priority_v2.sh" service systemd-status
bash "$AUTO/decode_priority_v2.sh" service systemd-logs 100
```

`systemd-install` deliberately does not start the experiment. On hosts where
user services must survive logout and boot without an interactive login, an
administrator must run this once:

```bash
sudo loginctl enable-linger demo1
```

### User-local Supervisor fallback

Supervisor needs no system configuration in this mode; its socket, PID, config,
and logs all stay below `runtime/service`:

```bash
bash "$AUTO/decode_priority_v2.sh" service supervisor-install
bash "$AUTO/decode_priority_v2.sh" service supervisor-start
bash "$AUTO/decode_priority_v2.sh" service supervisor-status
```

`supervisor-install` pins Supervisor 4.3.0 in the selected dispatcher's
`<runtime-root>/service/supervisor-venv`; it does not write to the system Python or the
operator's home directory.

This Supervisor fallback recovers a crashed Controller while `supervisord` is
alive. To recover automatically after a server reboot, the Supervisor daemon
itself must be registered with the host boot system; systemd user mode is the
simpler supported route.

For a legacy round paused by an older policy, run
`bash "$AUTO/decode_priority_v2.sh" service auto-retry-paused` and then start the managed service.
The one-shot request forces fresh Agent analysis against the current recovery
contract. With `failure_recovery.hard_terminal_only=true`, retry counters are
audit signals rather than pause gates; only a proven immutable external block
may preserve `paused_for_human`. A replacement round's new task/run identity is
persisted before monitoring resumes.

Both managers apply the same recovery policy:

- an active task/run is resumed after a process crash or reboot;
- Agent transport/structured-output failures use the frozen
  `agent_protocol_retries` budget within the same
  experiment round; schema-forbidden explanatory metadata is pruned with an
  adjacent audit, while parameter values, required fields and Search Limits
  remain strict;
- after four in-round protocol retries are exhausted, a completed round may be
  resumed by the service manager up to six times without rerunning its Benchmark;
- experiment failures are passed to the frozen Codex+DeepSeek Agent with the
  current logs plus unresolved signatures from earlier rounds. The Agent owns
  the recovery candidate and may request unchanged diagnostic reruns or a
  Search-Limits/Recovery-Registry-constrained single or coupled correction;
  deterministic rules are provider-availability fallbacks, not preselected
  Agent candidates. The Controller revalidates every action and never gives the
  Agent shell, image, topology, benchmark, path, or Lease mutation authority;
- retry, diagnostic and correction counts remain visible audit/budget signals.
  With `failure_recovery.hard_terminal_only=true`, exhausting those counters
  does not stop autonomy; each failed round returns to fresh Agent analysis;
- critical JSON/YAML artifacts use atomic replacement. `state.json.previous`
  mirrors the latest committed state, and submission intent/task/run identity
  form a recovery ledger so a crash around submission resumes existing work
  instead of blindly submitting a duplicate;
- artifact collection retries three times in place; persistent read-only
  control-plane failure is promoted to a bounded Supervisor recovery without
  changing the candidate or rerunning a completed Benchmark;
- a healthy-service GuideLLM zero-measurement failure stays in the Benchmark
  recovery path: expected nonzero shell commands cannot bypass their retry
  branch, and each failed attempt remains archived in a fresh result directory;
- a completed Session exits cleanly and stays stopped;
- `pause_for_human` is accepted only for a proven immutable block such as an
  unavailable/unsupported model or image, identity mismatch, missing permission
  or credential, resource/topology contract mismatch, or corrupt state. The
  round writes `human_intervention_required.json` with operator steps. Other
  Agent pauses are converted to continued recovery, and controller/provider
  errors are restarted with bounded backoff by the foreground service wrapper;
- TERM from either manager is converted into `STOP_REQUESTED`, then the wrapper
  waits up to 24 hours for the active round to archive before forced shutdown.

The Controller treats a missing Lease heartbeat as a transient platform
condition, not proof of an old worker protocol. Before each real submission it
waits up to `lab.readiness_wait_seconds` for the declared topology to return to
`2/2 Ready`. If readiness changes between the status check and `ktp-lab run`,
the specific pre-admission protocol-v2 error is retried with the same pending
candidate and run ID. It never overlaps a running slot. In hard-terminal-only
mode, an unavailable Lease enters controller recovery/backoff and is retried;
it does not become a human pause merely because a readiness window elapsed.

After an intentional stop, explicitly authorize a later start by archiving the
retained marker (the command moves it; it does not delete evidence):

```bash
bash "$AUTO/decode_priority_v2.sh" service authorize-resume
bash "$AUTO/decode_priority_v2.sh" service systemd-start   # or supervisor-start
```

The provided `systemd-restart` and `supervisor-restart` commands perform this
stop-marker archival automatically because invoking restart is itself explicit
authorization to continue the same Session.

An offline dry-run intentionally leaves a terminal `state.json`. The same
explicit command is used when a graceful operator stop must become a new
Session boundary after a rules/strategy migration. It accepts
`stopped_after_current_round` or `stopped_after_failed_round` only after the
Controller has cleared the active task; the old run id and state are archived
as audit evidence. Before the new service start, run:

```bash
bash "$AUTO/decode_priority_v2.sh" service authorize-new-session
```

This command refuses active, paused, failed, or otherwise non-terminal state;
it accepts only `dry_run_complete`, `completed_by_agent`, or `tuning_complete`.

Render both generated configurations without installing or starting anything:

```bash
bash "$AUTO/decode_priority_v2.sh" service render
```

Generated service files and logs live under `runtime/service` and are excluded
from Git. The API key remains only in the mode-600 `.secrets/controller.env`,
which is also excluded from Git.

Request a graceful stop:

```bash
bash tuning_pipeline/workflow/continuous/server_autonomous/decode_priority_v2.sh stop
```

The stop marker is retained. To authorize a later restart, move it to a dated
archive name inside the same runtime directory; do not delete it silently.

## Secret and network contract

`DEEPSEEK_API_KEY` is read only from the process environment. It is never
written to config, Session, candidate files, or logs. The server needs outbound
HTTPS access to `api.deepseek.com`; internal ktp-lab, registry, model storage,
and multi-node traffic remain on the internal network.

DeepSeek can only return schema-constrained decisions. It cannot execute shell
commands or submit ktp-lab tasks. The same Search Limits, compatibility rules,
candidate invariants, retry budgets, and measurement gates used by the existing
Controller remain authoritative.

## Codex Agent over DeepSeek (current default)

The autonomous default runs the schema-constrained decision prompt through
Codex while its server-managed configuration routes model inference to DeepSeek
V4 Flash. This is one Agent path (Codex execution framework plus DeepSeek model),
not two independent Agents. A named-profile installation may express the same
route as:

```yaml
agent:
  provider: codex
  providers:
    codex:
      command: auto
      profile: deepseek-v4-flash
      ephemeral: true
```

The currently verified server installation uses an explicit binary and a
server-managed base config instead of a named profile:

```yaml
agent:
  provider: codex
  providers:
    codex:
      command: /mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/tools/codex/releases/0.147.0/codex
      codex_home: /mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/tools/codex/home
      tmp_dir: /mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/tools/codex/tmp
      use_user_config: true
      ephemeral: true
```

This opt-in is required: without `profile` or `use_user_config: true`, the
Controller continues to pass `--ignore-user-config`. The server-managed config
was smoke-tested with Codex CLI 0.147.0 and `deepseek-v4-flash` on 2026-08-07.
The versioned, secret-free template is
`server_autonomous/codex_home.example/config.toml`; install it as
`$CODEX_HOME/config.toml`. It references `DEEPSEEK_API_KEY` by environment
variable name and never stores the credential value.

For a named-profile deployment, the service account must expose the configured
Codex command and provide `$CODEX_HOME/deepseek-v4-flash.config.toml`. For the
verified server-managed-base deployment above, `codex_home` supplies the base
config explicitly and the binary need not be in `PATH`. In both cases credentials
stay outside YAML. The Controller invokes Codex with a read-only sandbox, the
existing JSON output schema, and `--ephemeral`.

Protocol compatibility must be proven before use. Current upstream Codex custom
providers use the OpenAI Responses wire protocol, while DeepSeek's public V4
Flash endpoint documents OpenAI Chat Completions and Anthropic compatibility.
Therefore every installation requires a zero-NPU smoke test rather than relying
on documentation alone. This server's existing `https://api.deepseek.com`
Responses configuration passed that test with Codex CLI 0.147.0 on 2026-08-07.

The direct `agent.provider: deepseek` HTTP adapter remains available as a
lighter fallback for a **new Session**. It receives the same Controller-built
evidence and is subject to the same output schema and candidate validation, but
does not provide the Codex execution loop.

Run `preflight.sh` before creating the new Session. It verifies the DeepSeek key
and model, then Controller startup validation verifies the configured Codex
executable and CODEX_HOME. Before allocating a Lease, also run a zero-NPU
structured-output smoke test using the selected profile or server-managed base
config. A missing executable, missing CODEX_HOME, or invalid profile name fails
before any NPU experiment is submitted. Existing Sessions freeze their provider
and cannot be switched or resumed under this alternate route.
