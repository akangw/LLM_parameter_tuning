# 当前生产默认

本文只定义交接后应使用的 **Decode Priority V3 服务器自治入口**。运行中的
Session 始终以自己的 `session_config.yaml`、`00_search_space/` 和镜像身份为准，
不得用后来修改的全局配置覆盖。

| 维度 | 当前生产值 |
|---|---|
| 入口 | `server_autonomous/decode_priority_v3.sh` |
| Runtime root | `runtime_decode_priority_v3_live/` |
| Runtime Adapter | `glm52_w8a8_a3_dp4_tp8_decode_priority_v3` |
| 拓扑 | Atlas A3，2 节点 × 16 NPU，DP=4、local DP=2、TP=8 |
| 基线 | `expert_decode_glm52_w8a8_dp4_tp8_a10f1_v3.yaml`；用新口径重新测量 A10F1 |
| Search Limits | `automatic_registry_decode_priority_v2`；冻结结果为 25 Active、75 Reserve、42 Fixed |
| Agent 策略 | `decode_priority_agentic_v2` |
| Benchmark | `decode_only_c32_v2`：仅 `decode-256-2048`、C32；无串行长预热，同服务3次正式测量取中位数 |
| 主目标 | 输出 token 吞吐；TTFT P50/P90、TPOT P50/P90 同时测量并报告，作为诊断参考 |
| 历史 | `decode_priority_history_seed_v3.json`；保留V1/V2候选用于去重和定性经验，不与V3数值基线混算 |
| 拓扑搜索 | 关闭；改变 DP/TP 必须创建独立 Session |
| 恢复 | 服务器自治；只有已证明需要修改镜像、身份、权限、资源契约或损坏状态时才等待人工 |

策略顺序是：成功建立 A10F1 基线 → List 2 三个有序层 → List 1.3 条件层 →
Agent 自主跨层精调。List 1.2/1.1 不强制覆盖，也没有 Controller 总次数上限；
它们可以在有明确耦合依据时作为有序层 companion，或由 Agent 在跨层阶段选择。

仓库根 `tuning_pipeline/workflow/continuous/config.yaml` 和
`server_autonomous/config.yaml` 仍保留通用 Guided-V4/Fast-C32 框架入口，用于历史
兼容和显式实验，不是当前交接任务的生产入口。不要用通用 `start.sh`、
`status.sh` 或 `dp4_tp8_search_v4.sh` 管理 Decode Priority V3。

当前生产生命周期命令：

```bash
AUTO=tuning_pipeline/workflow/continuous/server_autonomous
bash "$AUTO/decode_priority_v3.sh" status
bash "$AUTO/decode_priority_v3.sh" service supervisor-status
```

Git 跟踪代码、Schema、版本化配置和文档；API Key、服务器本地 overlay、
`state.json`、日志、Supervisor 环境和实验目录均不进入 Git。
