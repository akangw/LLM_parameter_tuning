# 文档导航

## 当前生产交接必读

- `CURRENT_DEFAULTS.md`：Decode Priority V2 生产身份。
- `CURRENT_SESSION.md`：活动 Session 状态入口。
- `HANDOFF.md`：完整交接清单。
- `DECODE_ONLY_STRATEGY_ANALYSIS.md`：List 1/List 2 与 Agent 策略。
- `DECODE_ONLY_HARD_RULES.md`：确定性强约束。
- `DECODE_PRIORITY_V1_RUNBOOK.md`：V1 历史与 V2 续跑说明。
- `OPERATIONS.md`：运行和恢复。
- `ARTIFACTS.md`：产物布局。

## 通用框架参考

- `ARCHITECTURE.md`、根目录 `框架.md`：离线知识、Controller、Agent、Benchmark 总体架构。
- `PROJECT_STRUCTURE.md`：目录边界。
- `ASCEND_RUNTIME_ADAPTERS.md`：更换模型、镜像、拓扑时的适配边界。
- `MODEL_LOADING.md`：DTFS 和模型加载。
- `DEPENDENCIES.md`、`PORTABLE_QUICKSTART.md`、`LINUX_DOCKER_CONTROLLER.md`：迁移环境时使用。

通用 `config.yaml` 的 Guided-V4/Fast-C32 组合是可复用框架入口，不等于当前生产
Decode Priority V2。交接和恢复活动任务时始终使用 `decode_priority_v2.sh`。

## 历史或未来显式路线

- `A8_FAST_AGENTIC_V2.md`
- `A8_FRONTIER_AGENTIC_V3.md`
- `FIXED_DP2_TP16_V4.md`
- `FIXED_DP4_TP8_V1.md`
- `TOPOLOGY_CAMPAIGN_V4.md`
- `GLM52_W4A8C8_A0.md`

这些文件为旧 Session 复现、历史解释或未来规划保留。它们不是当前默认，不应复制
其中的参数、Lease、Benchmark 或启动命令来恢复 V2。为遵守审计和禁止删除约束，
历史文件保留在仓库中，但已从交接主阅读路径移除。
