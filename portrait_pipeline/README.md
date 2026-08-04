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
.\scripts\migrate-versions.ps1 -Vllm <tag-or-commit> -VllmAscend <tag-or-commit> -PortraitMode rebuild
```

第一条只完成源码抓取、结构化提取、覆盖审计、Stage-1 和确定性画像计划；Codex 路线还会准备待执行队列。第二条默认走 `migrate`，要求现有 ParameterYAML 并将其作为待源码复核的迁移提示；第三条走 `rebuild`，完全不读取旧画像。两条路线随后都生成并审计画像、生成 Tags、按场景召回并编译 Search Limits。可用 `-Scenario <yaml>` 替换场景模板。所有源码与结果进入 commit-qualified 隔离目录，不会直接替换 `outputs/ParameterYAML/`。

```text
build/version_migrations/<commit-pair>/
├─ 00_sources/          该运行独占的固定提交源码
├─ 01_extract/          结构化参数表面与覆盖证据
├─ 02_portrait_plan/    Stage-1、迁移/重建计划和 Anthropic 画像产物
├─ 03_portrait_queue/   Codex 画像队列及画像产物
├─ 04_tags/             五维 Tags、进度、日志和审计
└─ 05_search_limits/    指定场景的召回与 Search Limits
```

固定源码版本：

- vLLM：`418bd6273c03bf48d5066733769e0a74bdc51694`
- vllm-ascend：`32c8cf190f596b47f0d0b965e64aea9f2b789ad4`

画像审计：

```powershell
# 从项目根目录执行
cd .\portrait_pipeline
python -m build.codex_portrait_pipeline audit
```
