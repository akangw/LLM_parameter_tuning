# vLLM 参数画像迁移（隔离前半段）

这个目录是对原有 `extract_params -> parse_params` 前半段的隔离改造。

它不修改师兄项目的源码和既有产物。迁移完成后仍输出与原
`parse_params` 完全相同的 `ParameterYAML` 文件，因而可以继续交给原有
`../tuning_pipeline/tag_params`、`../tuning_pipeline/query.py` 和
`../tuning_pipeline/SKILL-with-tag.md`。

## 工作流

```text
固定的新版本源码
  -> 原 extract_params 的静态提取器
  -> 新版 parameters.json + provenance.json
  -> 确定性 Stage 1 初筛
  -> 与旧 ParameterYAML 的 A/B/CURRENT_ONLY/D 审计
  -> 原 parse_params 的 ParameterYAML 写入器（带迁移提示）
```

迁移等级：

- `A`：名称、类型、scope、默认值一致；旧画像是高可信提示，仍需当前源码复核。
- `B`：同一参数但默认值/类型/scope/名称映射发生变化；旧画像只作导航提示。
- `CURRENT_ONLY`：新版本初筛后保留、但没有可迁移旧画像；从当前源码完整画像。
- `C`：旧画像仍有对应参数，但新版 Stage 1 排除；只写入审计报告。
- `D`：旧画像参数在新版提取结果中不存在；只写入审计报告。

## 使用

先以 `--dry-run` 验证当前源码提取、初筛和迁移分类：

```powershell
python -m build.migration_pipeline `
  --vllm-src C:\path\to\vllm `
  --vllm-ascend-src C:\path\to\vllm-ascend `
  --legacy-dir .\build\parse_params\output\params `
  --extract-output .\build\extracted_parameters `
  --output .\build\migration_candidates `
  --dry-run
```

确认 `reports/migration-manifest.json` 后，移除 `--dry-run` 执行 LLM 画像。
迁移候选位于 `build/migration_candidates/`；最终完整画像由 Codex 构建器写入 `outputs/ParameterYAML/`。

```powershell
python ..\tuning_pipeline\query.py --params-dir ..\tuning_pipeline\tag_params\output\params --list-tags
```

`target-context.snapshot.yaml` 用于记录目标 commit；源码目录必须由调用方明确传入。
这样不会把本地临时源码路径或旧画像路径硬编码到师兄项目中。
