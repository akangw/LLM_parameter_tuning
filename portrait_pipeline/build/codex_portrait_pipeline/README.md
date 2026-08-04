# Codex Agent 参数画像流水线

该目录不调用 Claude/OpenAI API。它将当前版本候选参数制作为可续跑任务，
由 Codex Agent 阅读固定源码和迁移证据后生成师兄原格式 `ParameterYAML`。

```text
build/extracted_parameters/parameters.structured.json
build/migration_candidates/reports/migration-manifest.json
  -> prepare
  -> run/tasks/*.json（415 个任务）
  -> claim/context（源码证据）
  -> Codex 写 run/drafts/*.yaml
  -> accept（Pydantic schema 校验）
  -> run/params/*.yaml
```

常用命令：

```powershell
python -m build.codex_portrait_pipeline prepare
python -m build.codex_portrait_pipeline list --limit 10
python -m build.codex_portrait_pipeline claim param.xxxxxxxxxxxxxxxx
python -m build.codex_portrait_pipeline accept param.xxxxxxxxxxxxxxxx .\build\codex_portrait_pipeline\run\drafts\param.xxxxxxxxxxxxxxxx.yaml
python -m build.codex_portrait_pipeline audit
```

`run/params/` 是后续 `tag_params` 的输入。`run/skipped/` 保存经源码复核后
确认没有运行时性能影响的候选，不进入标签流程。
