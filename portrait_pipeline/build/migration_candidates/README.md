# 当前版本 ParameterYAML 产物

由 `migration_pipeline` 在当前固定源码、旧画像迁移提示和 LLM 复核后写入。

- `params/`：与师兄 `parse_params/output/params/` 同构的 ParameterYAML；
- `reports/migration-manifest.json`：A/B/CURRENT_ONLY 与旧 C/D 审计；
- `reports/stage1-summary.json`：确定性初筛统计。

此目录不覆盖旧 `parse_params/output/`。
