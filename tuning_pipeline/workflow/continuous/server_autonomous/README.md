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

From Windows, deploy the repository snapshot with:

```powershell
.\scripts\deploy-server-autonomous.ps1
```

On the server:

```bash
cd /mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-server-autonomous-418bd627-32c8cf190
export DEEPSEEK_API_KEY='...'

python3 -m pip install -r tuning_pipeline/requirements-server-autonomous.txt
bash tuning_pipeline/workflow/continuous/server_autonomous/seed_assets.sh
bash tuning_pipeline/workflow/continuous/server_autonomous/dry_run.sh
```

`dry_run.sh` performs local generation and validation only. It does not query,
create, or submit work to any Lease.

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
python3 -m pip install supervisor
bash "$AUTO/service.sh" supervisor-start
bash "$AUTO/service.sh" supervisor-status
```

This Supervisor fallback recovers a crashed Controller while `supervisord` is
alive. To recover automatically after a server reboot, the Supervisor daemon
itself must be registered with the host boot system; systemd user mode is the
simpler supported route.

Both managers apply the same recovery policy:

- an active task/run is resumed after a process crash or reboot;
- a completed Session exits cleanly and stays stopped;
- `paused_*`, inconsistent state, an already running Controller, or a retained
  `STOP_REQUESTED` marker exits with code 78 and is not restarted;
- TERM from either manager is converted into `STOP_REQUESTED`, then the wrapper
  waits up to 24 hours for the active round to archive before forced shutdown.

After an intentional stop, explicitly authorize a later start by archiving the
retained marker (the command moves it; it does not delete evidence):

```bash
bash "$AUTO/service.sh" authorize-resume
bash "$AUTO/service.sh" systemd-start   # or supervisor-start
```

The provided `systemd-restart` and `supervisor-restart` commands perform this
stop-marker archival automatically because invoking restart is itself explicit
authorization to continue the same Session.

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
