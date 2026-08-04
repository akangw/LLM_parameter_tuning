# 产物目录

## 正式成品

| 产物 | 位置 | 当前数量/状态 |
|---|---|---:|
| 参数画像 | `portrait_pipeline/outputs/ParameterYAML/` | 340 |
| 画像跳过证据 | `portrait_pipeline/outputs/skipped/` | 105 |
| 五维标签 | `tuning_pipeline/tag_params/output/params/` | 340 |
| 标签审计 | `tuning_pipeline/tag_params/output/audit.json` | 0 error |
| 最新 Search Limits | `tuning_pipeline/search_limits/` | 12 Active |
| 在线 Controller | `tuning_pipeline/workflow/continuous/continuous_tuning.py` | 已接入 |

## 可复现构建证据

- `portrait_pipeline/build/extracted_parameters/`：1540 个结构化参数表面和 provenance。
- `portrait_pipeline/build/migration_candidates/`：Stage-1 候选、排除项和迁移报告。
- `portrait_pipeline/build/parse_params/`：Schema、解析和校验工具。
- `portrait_pipeline/build/codex_portrait_pipeline/`：画像队列程序；大体积运行日志不进入 Git。
- `tuning_pipeline/workflow/search_space_compiler/`：注册表、场景、策略和编译器。
- `tuning_pipeline/workflow/sidecars/`：画像检索、默认规则和运行规则历史。

## 运行时产物

`tuning_pipeline/workflow/continuous/experiments/`、`state.json`、Controller 日志和 PID 属于机器本地运行状态，默认不提交 Git。每个 Session 内部仍按以下层级归档：

```text
round_NNN_label/
├─ 00_context/
├─ 01_query/
├─ 02_parameters/
├─ 03_submission/
├─ 04_runtime/
├─ 05_results/
└─ 06_agent_analysis/
```

交接时如需继续当前 Session，应通过加密文件传输单独复制完整 `experiments/<session>/` 和 `state.json`，不要把运行日志当作源码提交。
