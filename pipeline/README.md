# 业务链路总览

这里是理解项目的唯一业务入口。先不要进入 `portrait_pipeline/` 或
`tuning_pipeline/workflow/continuous/`；它们是实现目录，不是阅读顺序。

```text
01 参数知识
源码 → 参数画像迁移/重建 → Tags → 场景召回 → Search Limits
                                      ↓
02 Agent 调参
B0/A0 + Search Limits + 历史结果 → Agent 候选 → Controller 安全校验
                                      ↓
03 Benchmark
启动服务 → 固定负载测量 → 指标门禁 → 结果回填 → 下一轮
```

## 三个一级模块

| 顺序 | 模块 | 回答的问题 | 主要输出 |
|---:|---|---|---|
| 1 | [`01_parameter_knowledge/`](01_parameter_knowledge/README.md) | 哪些参数存在、与场景相关、允许怎么改？ | ParameterYAML、Tags、召回结果、Search Limits |
| 2 | [`02_agent_tuning/`](02_agent_tuning/README.md) | Agent 依据什么选择下一组参数？ | candidate、完整启动参数、Agent 决策证据 |
| 3 | [`03_benchmark/`](03_benchmark/README.md) | 候选是否比基线更好？ | metrics、comparison、通过/拒绝结论 |

## 配置和运行入口

这三个模块之外只需要记住两个入口：

- `scenarios/`：选择模型、量化、镜像、拓扑、基线和默认 Profile。
- `scripts/scenario.ps1`：初始化、预检、启动、恢复、查看和停止。

```powershell
.\scripts\scenario.ps1 -Action list
.\scripts\scenario.ps1 -Action artifacts -Name glm52-w8a8-a3-2n-dp2-tp16
```

## 阅读深度

1. 只想理解项目：只读本目录的四份 README。
2. 想运行项目：再读 `scenarios/README.md`。
3. 想新增模型/拓扑：再读 `docs/ASCEND_RUNTIME_ADAPTERS.md`。
4. 想修改框架实现：最后才进入 `portrait_pipeline/` 和 `tuning_pipeline/`。
