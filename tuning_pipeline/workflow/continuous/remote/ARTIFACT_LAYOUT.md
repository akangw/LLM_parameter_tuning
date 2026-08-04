# Server artifact and log layout

The server is the authoritative store for complete runtime and ServeBench logs.
Run IDs always use `<candidate-label>_<YYYYMMDD_HHMMSS>`, for example
`a1f1_20260804_133046`.

```text
workflow/auto/
|-- runs/<run-id>/                       project-owned authoritative run
|   |-- server_run_manifest.yaml         self-description and log index
|   |-- candidate.env                    frozen injected values
|   |-- effective_config.yaml            effective service/benchmark config
|   |-- vllm_common_command.txt           exact vLLM command
|   |-- task.yaml                         submitted lease task snapshot
|   |-- run_status.json                   current/final phase
|   |-- startup_timeline.jsonl            process-start and API-ready timestamps
|   |-- master.log                        DP0/TP16 vLLM log
|   |-- worker.log                        DP1/TP16 vLLM log
|   |-- benchmark_runner.log              Aligned-L1 orchestration
|   |-- benchmark_watchdog.log            detached recovery watchdog
|   |-- metrics.json                      consolidated valid metrics, when successful
|   |-- SERVICE_READY / BENCHMARK_* / MASTER_DONE
|   `-- servebench/
|       |-- config-rep<N>/                resolved input for repetition N
|       `-- results/l1-rep<N>/
|           |-- result.json               formal ServeBench result
|           |-- logs/run.log              repetition-level log
|           `-- artifacts/cases/...       per-workload/per-concurrency evidence
`-- lab_runs/<run-id>/                    ktp-lab outer process capture
    |-- request.json
    |-- requests/service.json
    `-- service/
        |-- rank-000.log                  master Pod outer log
        `-- rank-001.log                  worker Pod outer log
```

Reading order:

1. `server_run_manifest.yaml` and `run_status.json` identify the run.
2. `startup_timeline.jsonl`, `master.log`, and `worker.log` diagnose service startup.
3. `benchmark_runner.log` and `benchmark_watchdog.log` diagnose measurement control.
4. `servebench/results/l1-rep<N>/logs/run.log` diagnoses individual cases.
5. Only `metrics.json` is a valid consolidated performance result.

The local Controller mirrors only the core logs, status markers, effective
configuration, and consolidated metrics needed by the Agent. Complete per-case
ServeBench logs stay here on the server. Server artifacts are retained and are
never removed by the workflow.
