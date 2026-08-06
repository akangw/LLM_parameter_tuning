# 01 参数知识：从源码到 Search Limits

这一层只负责回答：当前源码有什么参数，当前场景能让 Agent 调哪些参数。

```text
1 参数画像
  ↓
2 画像迁移或重建
  ↓
3 Tags
  ↓
4 场景召回
  ↓
5 Search Limits
```

| 阶段 | 输入 | 做什么 | 输出/实现位置 |
|---|---|---|---|
| 1. 参数画像 | 固定 commit 的 vLLM、vllm-ascend 源码 | 整理入口、默认值、合法值、影响、风险和关联参数 | `portrait_pipeline/outputs/ParameterYAML/` |
| 2. 迁移/重建 | 新旧源码与已有画像 | `migrate` 复核迁移，或 `rebuild` 从新源码重建 | `portrait_pipeline/build/migration_pipeline/` |
| 3. Tags | ParameterYAML | 添加模型、目标、拓扑、硬件、场景五维标签 | `tuning_pipeline/tag_params/output/params/` |
| 4. 召回 | Tags + 场景定义 | 找到当前模型/拓扑可能相关的画像 | Session `00_search_space/parameter_portraits.full.yaml` |
| 5. Search Limits | 召回画像 + 注册表 + 约束 | 分类 Active、Reserve、Fixed、Rejected | Session `00_search_space/search_space.compiled.yaml` |

## 必须区分的四个概念

- 画像：参数说明书，不代表允许调整。
- Tags：分类索引，不代表进入候选池。
- 召回：与场景相关的知识集合，可以偏宽。
- Search Limits：Controller 最终允许 Agent 使用的边界，必须严格。

因此 `340 份画像 → 109 份召回 → 12 个 Active` 是逐层收紧，不是画像丢失。

## 场景隔离

同一源码版本可以共享 340 份画像和 Tags；W8A8、W4A8C8 分别使用自己的 Scenario、
模型/拓扑约束和基线重新召回与编译。不同场景不能共享历史最优值或 Session Search Limits。

## 最终看哪里

离线知识成品看：

```text
portrait_pipeline/outputs/ParameterYAML/
tuning_pipeline/tag_params/output/params/
```

某次实验真正采用的知识边界只看：

```text
tuning_pipeline/workflow/continuous/scenario_runs/<scenario-id>/experiments/<session-id>/00_search_space/
```
