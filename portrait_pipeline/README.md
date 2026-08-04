# 当前版本参数画像

## 成品

- `outputs/ParameterYAML/`：340 份当前版本完整画像，是下游 Tags 的唯一输入。
- `outputs/skipped/`：105 份有源码或语义依据的画像阶段跳过记录。
- `sources/`：固定提交的 vLLM 与 vllm-ascend 源码。

## 可复现构建区

- `build/extracted_parameters/`：1540 个结构化源码表面及独立覆盖审计。
- `build/migration_candidates/`：445 个 Stage-1 候选、1095 个排除证据及 A/B/CURRENT_ONLY 迁移报告。
- `build/migration_pipeline/`：结构化提取、覆盖审计、Stage-1 与迁移程序。
- `build/parse_params/`：ParameterYAML Schema 和验证工具。
- `build/codex_portrait_pipeline/`：Codex 画像队列、上下文与审计日志。
- `build/version_migrations/`：按两份源码 commit 隔离的新版本迁移运行区（可复现、本地生成、不入 Git）。
- `build/target-context.snapshot.yaml`：目标版本和场景快照。

## 新版本迁移接口

从仓库根目录运行：

```powershell
.\scripts\migrate-versions.ps1 -Vllm <tag-or-commit> -VllmAscend <tag-or-commit> -PrepareOnly
.\scripts\migrate-versions.ps1 -Vllm <tag-or-commit> -VllmAscend <tag-or-commit>
```

第一条只完成源码抓取、结构化提取、覆盖审计、Stage-1 和 Codex 队列准备；第二条继续用 Codex 生成并审计画像。所有结果进入 commit-qualified 隔离目录，不会直接替换 `outputs/ParameterYAML/`。迁移提取器、Schema 和队列程序均在本仓库内，克隆后无需旧项目目录。只有人工审计并明确激活新画像后，才重新生成 Tags 和 Search Limits。

固定源码版本：

- vLLM：`418bd6273c03bf48d5066733769e0a74bdc51694`
- vllm-ascend：`32c8cf190f596b47f0d0b965e64aea9f2b789ad4`

画像审计：

```powershell
# 从项目根目录执行
cd .\portrait_pipeline
python -m build.codex_portrait_pipeline audit
```
