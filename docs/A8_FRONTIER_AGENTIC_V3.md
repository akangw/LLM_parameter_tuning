# A8 Frontier Agentic V3

版本标识：`a8-frontier-agentic-v3-20260814`

这是已归档的固定 DP2/TP16 frontier-v3 方案；当前默认已切换为固定 DP4/TP8 的 guided-v4。本文保留用于解释旧 Session 的冻结身份，不能当作新 Session 默认。V3 的目标不是穷举，而是在确定性硬约束内允许 Agent 做高风险、可解释、可复现的组合探索。

## 固定身份

- Runtime：`glm52_w8a8_a3_a8_frontier_v3`
- Topology：`a3_dp2_tp16`（当前唯一已验证拓扑；拓扑选择发生在 Session 外）
- Baseline：`a8_expert_fast_v1`
- Search Space：`automatic_registry_a8_frontier_v3`
- Strategy：`hierarchical_agentic_frontier_v3`
- Benchmark：`aligned_fast_c32_v1`
- History：`latest_completed_session`，只接受 Benchmark、镜像和源码身份完全一致的 Session

上述身份在创建 Session 时冻结。旧 Session 继续使用自己的 `session_config.yaml`，不会被新默认值污染。

## A. 容量与显存几何

`max_model_len` 是普通可调 Active 参数，不再固定。四轴显式网格为：

| 参数 | 候选值 |
|---|---|
| `max_model_len` | 16384、32768、49152、64000 |
| `gpu_memory_utilization` | 0.85、0.90、0.92、0.93、0.95、0.97 |
| `max_num_seqs` | 32、64、128、192、256、384、512 |
| `max_num_batched_tokens` | 1024、2048、4096、8192、16384、32768 |

容量首层允许一次改变 2–4 个独立参数，按“上下文与显存”“序列与批 token”“四轴联合前沿”三类交互逐步建立证据。Controller 只执行白名单、组合约束、网格步长、运行时能力和测量门禁，不替 Agent 选具体组合。

## B. 高风险组合探索

探索阶段每轮允许 1–4 个独立参数、单参数最多跨 2 个网格、总步长最多 8。容量四轴是明确批准的高风险交互组，可在一次实验中联合改变四项。其他跨层高风险组合也不是一刀切禁止，但 Agent 必须为每个参数给出知识证据、约束检查，并为每个额外变化解释交互关系。

这使系统可以探索人工通常不会逐项尝试的联合空间，同时仍会在提交前拒绝不在域内、违反前置条件、运行时不支持或重复的候选。

## C. 条件失败记忆

`parameter_oom` 和 `parameter_invalid` 现在只排除发生失败的完整配置，不再全局删除某个参数值。例如，`max_num_batched_tokens=16384` 在一组显存/并发条件下 OOM，只会记录该完整组合；相同 batch 值仍可与不同 `max_model_len`、显存比例或序列数重新组合。

失败观测仍进入参数风险评分，因此系统不会忘记风险。Controller 只有在历史条件覆盖当前完整 Candidate Schema 且每个值都相同时才硬拒绝；部分重合只作为 Agent 的风险证据。

## D. Active / Reserve 轮换

当前自动编译结果是：

```text
340 ParameterYAML
→ 225 场景召回
→ 142 自动 Registry 参数
→ 103 Tunable：28 Active + 75 Reserve
→ 39 Fixed + 0 Compiler Rejected
```

新 Session 会读取最新的身份匹配历史与上一个 Active 选择。每次最多进行 3 个 Active/Reserve 交换，最小历史评分优势为 1，并至少保留 5 个核心参数。轮换比较的是历史证据调整分，而不是静态类别分；没有参数级差异证据时不会发生伪轮换。每个交换都会写入 `rotation_report.yaml`。

当前 28 个 Active 为：

```text
max_num_seqs
max_model_len
max_num_batched_tokens
gpu_memory_utilization
compilation_mode
num_speculative_tokens
async_scheduling
enable_expert_parallel
speculative_config__method
long_prefill_token_threshold
mlapo
fused_mc2
enable_balance_scheduling
enable_reduce_sample
speculative_config__enforce_eager
speculative_config__attention_backend
cudagraph_capture_sizes
VLLM_ASCEND_ENABLE_BATCH_MEMCPY
TASK_QUEUE_ENABLE
additional_config__ascend_compilation_config__fuse_allreduce_rms
additional_config__prefill_comm_compute_overlap
additional_config__ascend_compilation_config__enable_static_kernel
additional_config__ascend_compilation_config__enable_npugraph_ex
additional_config__ascend_compilation_config__fuse_norm_quant
compilation_enable_sp
enable_prefix_caching
flashcomm1
disable_hybrid_kv_cache_manager
```

`max_cudagraph_capture_size` 仍保留完整候选域，但作为 `max_num_seqs`、MTP 深度和显式 capture list 的派生运行时参数，不单独占用 Active 名额。

Ascend 推测解码后端只保留 `null`（能力自选）和 `FLASH_ATTN`；当前镜像无法证明可用的 CUDA/ROCm/XPU 后端在编译阶段被过滤。`disable_padded_drafter_batch=true` 在当前异步 MTP 契约下不兼容，因此不会浪费实验轮次。

## E. Agent 自主策略与 65/25/10

分层顺序如下：

1. 拓扑首层（Session 外）
2. 容量与显存几何
3. MTP / 推测解码
4. Prefill、KV Cache 与调度
5. 编译与图形状
6. MoE 路由与均衡
7. Ascend 通信与融合内核

有序覆盖结束后，Controller 不提供候选层或候选意图。Agent 必须为每次继续决策声明 `exploration_intent`：`exploitation`、`cross_layer_interaction`、`frontier_novelty`，或者严格受限的 `diagnostic_ablation`。

Controller 从已经归档的决策中计算实际计数、占比和相对 65% / 25% / 10% 目标的缺口，并把 `underrepresented_intents` 作为证据返回。它明确输出 `controller_preselected_intent: null`；最终选择仍由 Agent 完成，所以比例是可测量的探索导向，不是机械配额。

MTP 的正常策略保持开启。`true/false` 能力仍被保留，但从已开启 MTP 切到关闭只能使用 `diagnostic_ablation`，且每个 Session 最多一次。这样既保留诊断能力，又避免重复测试通常没有收益的关闭路径。

## Benchmark 与报告

旧 Session 的工作副本固定在项目自己的 `workflow/auto/vendor/benchmark-tuning-fast-c32-v1`，保留四类 C32 工作负载，每类 64 个正式请求。配置中的 600 秒仅是设计目标，不是执行截断；真正的外层安全边界由冻结的 round timeout 决定。

最终 `metrics.json` 和报告必须包含：

- 输出吞吐（顶层与每工作负载）
- TTFT P50 / P90
- TPOT P50 / P90
- Benchmark 实际墙钟时间
- 请求完整性、错误数和 token shape 证据

## 权限与发布规则

本地与授权服务器目录可以对齐，但默认不提交或推送任何 GitHub 仓库。只有操作者明确要求时才允许更新 GitHub。服务器写操作仅限 `cjx-workspace` 下的两个项目目录；流程不需要、也禁止使用删除命令。
