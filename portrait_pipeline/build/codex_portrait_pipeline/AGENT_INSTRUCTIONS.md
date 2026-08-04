# Codex portrait worker instructions

For each claimed task:

1. Read `build/codex_portrait_pipeline/run/tasks/<task_id>.json` and
   `build/codex_portrait_pipeline/run/contexts/<task_id>.json`.
2. Inspect the exact pinned repositories recorded in the queue `index.json`
   under `inputs.source_roots`; Ascend behavior has precedence. The default
   production queue uses `sources/`, while version-migration queues use their
   isolated source checkouts.
   Every `source_file` and `usage_locations.file` value must be relative to
   its repository root. Ascend package code starts with `vllm_ascend/`, while
   Ascend documentation starts with `docs/` (never `vllm_ascend/docs/`).
3. Treat class A legacy profiles as hypotheses that still require source
   verification, class B profiles as navigation hints, and CURRENT_ONLY tasks
   as clean analyses from current source.
4. Author exactly one YAML file using `parse_params.schema.ParameterYAML`.
   If the parameter has no latency/throughput/memory effect, use the minimal
   `SkippedParamYAML` shape.
5. Write the draft to
   `build/codex_portrait_pipeline/run/drafts/<task_id>.yaml`, then run
   `python -m build.codex_portrait_pipeline accept <task_id> <draft>` and fix any
   validation error before moving to the next task.
6. Never edit `build/parse_params/output`, `../tuning_pipeline`, or the pinned
   source trees.
