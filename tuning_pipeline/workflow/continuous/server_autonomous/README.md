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
