# Decode Priority V2 项目交接

## 1. 交接结论

当前生产路线是固定 DP4/TP8、Decode-only C32、A10F1 专家锚点、
`automatic_registry_decode_priority_v2` 和 `decode_priority_agentic_v1` 的服务器
自治闭环。旧 DP2/TP16、A8 Fast/Frontier、Guided-V4/Fast-C32 和 Topology Campaign
均保留为历史或显式路线，不能作为当前任务入口。

接手者阅读顺序：

1. `docs/CURRENT_DEFAULTS.md`：生产身份和唯一入口。
2. `docs/CURRENT_SESSION.md`：查看活动任务的方法。
3. `docs/DECODE_ONLY_STRATEGY_ANALYSIS.md`：List 1/List 2 参数与策略。
4. `docs/DECODE_ONLY_HARD_RULES.md`：Controller 强制规则。
5. `docs/DECODE_PRIORITY_V1_RUNBOOK.md`：V1 历史和 V2 续跑包。
6. `docs/OPERATIONS.md`：运行、恢复、停止和故障边界。
7. `docs/ARTIFACTS.md`：每轮证据和产物位置。
8. `docs/README.md`：现行、参考和历史文档分类。

## 2. 当前生产身份

```text
服务器目录   /mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-decode-priority-v1
Dispatcher   tuning_pipeline/workflow/continuous/server_autonomous/decode_priority_v2.sh
Runtime root tuning_pipeline/workflow/continuous/server_autonomous/runtime_decode_priority_v2_live
Config       config.dp4_tp8.decode_priority_v2.yaml + Git 忽略的本地 overlay
Runtime      glm52_w8a8_a3_dp4_tp8_decode_priority_v2
Topology     2×16 NPU，DP4、local DP2、TP8
Baseline     expert_decode_glm52_w8a8_dp4_tp8_a10f1_v2.yaml
Search       automatic_registry_decode_priority_v2
Strategy     decode_priority_agentic_v1
Benchmark    decode_only_c32_v1
History      decode_priority_history_seed_v2.json
```

镜像、vLLM/vllm-ascend commit、模型路径、Benchmark 指纹和 Executor 身份已经冻结
进 Session。不要从文档手抄这些值替换运行状态；以
`session_config.yaml`、`image_version_manifest.yaml` 和 `00_search_space/` 为准。

## 3. 日常状态与自治服务

```bash
cd /mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-decode-priority-v1
AUTO=tuning_pipeline/workflow/continuous/server_autonomous

bash "$AUTO/decode_priority_v2.sh" service supervisor-status
bash "$AUTO/decode_priority_v2.sh" status
```

正常状态应同时满足：Supervisor 为 `RUNNING`；State 不含 `controller_error`；Lease
为 2/2 Ready、32/32 NPU；同一 Lease 只有一个受管 run。服务器自治不依赖本地电脑
开机。

不要同时启动 `decode_priority_v1.sh`、通用 `start.sh`、Windows Controller 或其他
dispatcher 管理同一 Lease。V2 运行期间可以同步纯文档和测试，但不能热改其冻结
Config、Search Limits、策略、Benchmark 或运行脚本；这些变化必须在轮次优雅结束后
创建新 Session。

## 4. 策略与历史

- V2 第 0 轮重新测量 A10F1，随后进入 List 2，而不是直接跳到跨层阶段。
- 冻结历史包含 28 条兼容记录，其中 24 条计为参数实验、23 个不同完整候选，覆盖
  A1–A15；重复项只是两个来源都含同一基线，不会触发重复提交。
- Agent 看到完整候选、输出吞吐、TTFT/TPOT、成功/失败和归因证据。
- Controller 对完整候选做确定性去重；恢复历史最佳只更新状态，不重跑 Benchmark。
- 基线、确认测量和无有效指标的基础设施重试允许同参数运行，这是实验完整性要求。
- List 1.2/1.1 没有四次总配额；它们仍需机制、历史或画像证据，不能盲扫。

## 5. 故障处理边界

Controller 依次使用：确定性规则 → Benchmark 内部重试 → 同候选基础设施重试 →
Agent 结构化诊断 → Search Limits/Recovery Registry 内的参数修复。只有已经证明必须
由人工修改镜像/模型支持、身份、权限、凭据、资源/拓扑契约或损坏状态时才停止等待。

接手人不要根据一条日志手工改参数，也不要把没有 `metrics.json` 的基础设施失败
计为负收益。完整硬规则见 `docs/DECODE_ONLY_HARD_RULES.md`。

## 6. Git 与运行产物

两个 GitHub `main`、本地和服务器源码应指向同一提交。检查：

```powershell
git status --short
git rev-parse HEAD
git ls-remote origin refs/heads/main
git ls-remote teacher refs/heads/main
```

```bash
git status --short
git rev-parse HEAD
```

不进入 Git 的内容包括：API Key、本地 overlay、运行状态、Session、日志、PID、
Supervisor venv、模型缓存和远端 Benchmark 原始目录。它们不是版本漂移；活动 Session
的运行目录是审计证据，禁止删除。

交接文档已通过 `docs/README.md` 收敛为“当前生产 / 通用参考 / 历史路线”三类。
历史文档没有物理删除，因为旧 Session 仍依赖其身份说明，也因为审计证据禁止删除；
它们已退出主阅读路径。测试缓存、PPT 临时目录、旧本地实验目录和 Python 缓存均由
Git 忽略，不属于源码交付。不要为了让工作区看起来更短而移动或清理服务器活动
`runtime_decode_priority_v2_live/`、当前 Lease 产物或任何轮次目录。

## 7. 安全约束

- 服务器写操作仅限 `cjx-workspace` 下授权项目目录。
- `/mnt/host-model/slai/user-1-wangakang/wangakang` 的其他内容只读。
- 不执行删除命令，不覆盖或改写旧轮次证据。
- 不在活动 Session 中切换镜像、拓扑、Benchmark、Search Limits 或策略。
- 不提交 Key、SSH 凭据、Lease 私有 overlay 和运行产物。
- 改变 DP/TP、模型、镜像或 Benchmark 身份时创建新 Session，不跨身份比较指标。
