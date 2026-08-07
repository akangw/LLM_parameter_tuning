# Executor Adapter v1

This directory is the allowlisted extension point for schedulers other than
the built-in `ktp` / `ktp_lab` execution paths.  The default configuration does
not select an adapter, so adding files here cannot affect an existing Session.

## Boundary

The adapter owns only resource-manager operations:

```text
prepare -> check_ready -> submit -> snapshot -> stop / wait_for_release
                                      `-> optional start_benchmark / stop_partial
```

The Controller remains authoritative for:

- B0 and candidate generation;
- Search Limits and runtime-rule validation;
- candidate environment rendering;
- Session/Round state and failure policy;
- artifact collection and metric comparison;
- Agent decisions and acceptance gates.

Every scheduler must preserve the remote artifact contract:

```text
<remote_auto>/runs/<run_id>/
```

The service scripts write `SERVICE_READY`, logs, `MASTER_DONE`, and
`metrics.json` below that directory.  `submit` must arrange for the selected
scheduler to launch the frozen model command and return a stable `task_id` plus
a filesystem-safe `run_id`.

## Create an adapter

1. Copy `template_executor_bridge.py` to a new file in this directory.
2. Implement its action handlers with the target scheduler CLI/API.
3. Keep credentials in environment variables.  Adapter configuration is
   archived in `session_config.yaml` and must never contain a secret value.
4. Copy `executor_adapter.example.yaml` to a private file in this directory,
   fill the normal operator-specific remote/model fields there, and keep it out
   of Git.
5. Run a new-Session check before any submission.

```powershell
.\一键启动.ps1 -Config .\path\to\scheduler.local.yaml -CheckOnly -NewSession
```

Only after the read-only check succeeds should the operator prepare resources
or start a new Session. Existing Sessions retain their frozen execution mode
and cannot switch adapters during resume.

## JSON protocol

The bridge reads one JSON object from stdin and writes exactly one JSON object
to stdout. Diagnostics belong on stderr.

Request:

```json
{
  "api_version": "vllmtkb-executor-adapter/v1",
  "action": "snapshot",
  "context": {},
  "payload": {"task_id": "scheduler-job-id"},
  "adapter_config": {}
}
```

Every success response contains:

```json
{"api_version": "vllmtkb-executor-adapter/v1", "ok": true}
```

Required action-specific fields:

| Action | Response fields |
|---|---|
| `prepare` | optional `message` |
| `check_ready` | optional `message` |
| `submit` | `run_id`, and `task_id` unless dry-run; optional `task` mapping |
| `snapshot` | `snapshot.status`, `active_pods`, `terminal`, `partial_failure` |
| `stop` | optional `message` |
| `wait_for_release` | `released: true` only after all resources for the round stop |
| `start_benchmark` | optional; required by `aligned_l1` |
| `stop_partial` | optional; otherwise Controller falls back to `stop` |

An error response should use `ok: false` and a concise `error`. A non-zero
process exit, malformed JSON, API-version mismatch, missing task/run identity,
or malformed snapshot fails closed.

## Topology versus scheduler

`workflow/continuous/executor_profiles.yaml` validates the process/rank layout.
This adapter controls the resource manager. A scheduler replacement that keeps
the same master/worker contract can reuse an existing topology profile. A new
rank layout still needs its own integrated Executor Profile and Runtime Adapter;
the generic Controller does not need to be rewritten.
