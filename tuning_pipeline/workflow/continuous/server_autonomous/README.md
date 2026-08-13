# Server-autonomous mode

This mode runs the knowledge query, deterministic Controller, DeepSeek Agent,
ktp-lab submission, Benchmark collection, and Session archive on the Linux
server.  It is isolated from the default Windows-to-server chain.

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
└── vllmtkb-server-autonomous-418bd627-32c8cf190/
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

On the server:

```bash
cd /mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-server-autonomous-418bd627-32c8cf190
export DEEPSEEK_API_KEY='...'

python3 -m pip install -r tuning_pipeline/requirements-server-autonomous.txt
# Required only by the verified private aligned_l1_v4 profile.  Public
# vllm_bench_public_v1 and custom Benchmark adapters do not use this source.
bash tuning_pipeline/workflow/continuous/server_autonomous/seed_assets.sh
bash tuning_pipeline/workflow/continuous/server_autonomous/dry_run.sh
bash tuning_pipeline/workflow/continuous/server_autonomous/service.sh authorize-new-session
```

`dry_run.sh` performs local generation and validation only. It does not query,
create, or submit work to any Lease.  It deliberately leaves terminal audit
state, so `authorize-new-session` archives that evidence before a real Session.

After the operator has confirmed that every `blocked_lease_names` entry is no
longer active, create the isolated persistent Lease:

```bash
bash tuning_pipeline/workflow/continuous/server_autonomous/prepare_lease.sh
bash tuning_pipeline/workflow/continuous/server_autonomous/preflight.sh
```

Start or resume the Controller:

```bash
bash tuning_pipeline/workflow/continuous/server_autonomous/start.sh auto
bash tuning_pipeline/workflow/continuous/server_autonomous/status.sh
```

## Persistent service (recommended)

`start.sh` remains available for compatibility. For unattended operation, use
the foreground runner through either systemd or Supervisor. These service files
belong only to server-autonomous mode and do not modify the Windows-to-server
main chain.

First create the private environment file and replace its placeholder key:

```bash
AUTO=tuning_pipeline/workflow/continuous/server_autonomous
bash "$AUTO/service.sh" prepare-env
vi "$AUTO/.secrets/controller.env"
chmod 600 "$AUTO/.secrets/controller.env"
```

### systemd user service (preferred)

This provides crash recovery and, with user lingering enabled, reboot recovery:

```bash
bash "$AUTO/service.sh" systemd-install
bash "$AUTO/service.sh" systemd-start
bash "$AUTO/service.sh" systemd-status
bash "$AUTO/service.sh" systemd-logs 100
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
bash "$AUTO/service.sh" supervisor-install
bash "$AUTO/service.sh" supervisor-start
bash "$AUTO/service.sh" supervisor-status
```

`supervisor-install` pins Supervisor 4.3.0 in
`runtime/service/supervisor-venv`; it does not write to the system Python or the
operator's home directory.

This Supervisor fallback recovers a crashed Controller while `supervisord` is
alive. To recover automatically after a server reboot, the Supervisor daemon
itself must be registered with the host boot system; systemd user mode is the
simpler supported route.

Both managers apply the same recovery policy:

- an active task/run is resumed after a process crash or reboot;
- Agent transport/structured-output failures are retried twice within the same
  experiment round; schema-forbidden explanatory metadata is pruned with an
  adjacent audit, while parameter values, required fields and Search Limits
  remain strict;
- after protocol retries are exhausted, a completed round may be resumed by
  the service manager up to three times without rerunning its Benchmark;
- experiment failures not solved by deterministic rules are passed to the
  frozen Codex+DeepSeek Agent with complete archived evidence. It may request
  one unchanged diagnostic rerun or a Search-Limits-constrained parameter
  correction; the Controller revalidates every action and never gives the
  Agent shell, image, topology, benchmark, path, or Lease mutation authority;
- one failure chain is capped at four transient retries, one Agent diagnostic
  retry, three parameter corrections, and six total recovery rounds;
- a completed Session exits cleanly and stays stopped;
- unknown invariant failures, exhausted recovery, other `paused_*`, inconsistent
  state, an already running Controller, or a retained `STOP_REQUESTED` marker
  exits with code 78 and is not restarted;
- TERM from either manager is converted into `STOP_REQUESTED`, then the wrapper
  waits up to 24 hours for the active round to archive before forced shutdown.

The Controller treats a missing Lease heartbeat as a transient platform
condition, not proof of an old worker protocol. Before each real submission it
waits up to `lab.readiness_wait_seconds` for the declared topology to return to
`2/2 Ready`. If readiness changes between the status check and `ktp-lab run`,
the specific pre-admission protocol-v2 error is retried with the same pending
candidate and run ID. It never retries arbitrary submission errors, never
overlaps a running slot, and safely pauses after the bounded deadline expires.

After an intentional stop, explicitly authorize a later start by archiving the
retained marker (the command moves it; it does not delete evidence):

```bash
bash "$AUTO/service.sh" authorize-resume
bash "$AUTO/service.sh" systemd-start   # or supervisor-start
```

The provided `systemd-restart` and `supervisor-restart` commands perform this
stop-marker archival automatically because invoking restart is itself explicit
authorization to continue the same Session.

An offline dry-run intentionally leaves a terminal `state.json`. Before the
first real service start, archive that state without deleting its evidence:

```bash
bash "$AUTO/service.sh" authorize-new-session
```

This command refuses active, paused, failed, or otherwise non-terminal state;
it accepts only `dry_run_complete`, `completed_by_agent`, or `tuning_complete`.

Render both generated configurations without installing or starting anything:

```bash
bash "$AUTO/service.sh" render
```

Generated service files and logs live under `runtime/service` and are excluded
from Git. The API key remains only in the mode-600 `.secrets/controller.env`,
which is also excluded from Git.

Request a graceful stop:

```bash
bash tuning_pipeline/workflow/continuous/server_autonomous/stop.sh
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
