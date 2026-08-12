# 当前实验摘要

## 当前目标

- 场景：`glm52-w8a8-a3-2n-dp2-tp16`
- 模型：GLM-5.2 W8A8
- 拓扑：2 节点 × 16 NPU，DP=2、TP=16
- Benchmark：`aligned_l1_v4`
- Agent 策略：`best_anchor_coverage_v2`
- 项目统一默认 Search Limits：`automatic_registry_v1`；人工审计路线需在新 Session 显式选择 `curated_registry_v1`
- 项目统一默认 Agent Strategy：`hierarchical_throughput_v1`

W4A8C8 场景仍为 `planned`，当前参数画像对应的运行镜像不支持该量化方式，因此不得启动。

## B0 状态

B0 的参数定义和一次真实启动后解析出的有效参数已经留存，但当前 B0 **尚未通过 Benchmark 验收，不能称为正式基线锚点**。

- 定义：`tuning_pipeline/workflow/baselines/b0_deployable_64k.yaml`
- 最近有效参数：`tuning_pipeline/workflow/continuous/experiments/glm52_continuous_20260806_140358/round_000_b0_deployable/02_parameters/effective_config.yaml`
- 最近 Session：`glm52_continuous_20260806_140358`
- Session 状态：`stopped_after_failed_round`

最近三次 B0 尝试均未生成可验收的 `metrics.json`：

| 尝试 | 结果 |
|---|---|
| `round_000_b0_deployable` | 启动阶段发生端口占用，未形成指标 |
| `a0r1` | 节点驱逐/部分失败，未形成指标 |
| `a0r2` | 服务成功 Ready，Benchmark 已开始；在 `balanced-1024-1024/c16-warmup` 阶段服务连接被拒绝，Benchmark 未完成 |

因此准确结论是：B0 参数已经固定并有真实 Ready 证据；当前官方 B0 分数仍为空。

## 历史数据边界

旧 Session `glm52_continuous_20260804_112130` 的 A0 分数 `602.5576` 只能作为历史参考。它与当前 B0 的验收合同、运行身份和重复次数不完全相同，不能直接升级成当前官方 B0。

## 下一步

当前 Lease `vllmtkb-418bd627-32c8cf190-glm52-a3-32npu` 已终止（0/2 Ready）。开始新实验前应先重新准备 W8A8 2×16 Lease，再创建全新 Session，从 B0 完整 Benchmark 开始。B0 通过指标门禁后，流程才会进入 Agent 提案和后续参数轮次。
