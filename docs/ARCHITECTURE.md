# 架构与数据流

> 2026-08-18 当前默认为固定 DP4/TP8、A8 DP4 派生基线、28/75 自动空间、
> guided-v4 Agent 自主选参与 fast-C32-v2 固定基准。Topology Campaign V4 已实现但
> 不介入当前主流程。本文后续出现的 B0、DP2、22/80、
> `hierarchical_throughput_v1` 或 `aligned_l1_v4` 默认描述属于保留的旧路线；
> 当前权威设计见 [A8_FRONTIER_AGENTIC_V3.md](A8_FRONTIER_AGENTIC_V3.md)。

## 系统边界

项目分为离线知识构建和在线连续调优两条链路。本地 Windows 机器是控制面，`hetao-npu` 是执行入口，ktp-lab 两节点 Lease 是计算面。

```mermaid
flowchart LR
    A["固定提交源码"] --> B["结构化参数提取"]
    B --> C["Stage-1 初筛与迁移"]
    C --> D["ParameterYAML 参数画像"]
    D --> E["五维 Tags"]
    E --> R["场景召回"]
    R --> P["可选：自动注册表 / 人工注册表"]
    P --> F["Search Limits 编译"]
    F --> G["本地 Controller"]
    G --> H["Codex 选参与确定性校验"]
    H --> I["独立 ktp-lab Lease"]
    I --> J["vLLM + Aligned-L1"]
    J --> K["指标、日志和失败证据"]
    K --> G
```

## 离线知识链路

1. 从两个固定提交识别 CLI、配置字段和环境变量等参数表面。
2. 独立覆盖审计确保结构化提取没有已知缺口。
3. Stage-1 按性能相关性、可调性和安全边界初筛。
4. 迁移程序只迁移仍适用于当前源码的旧知识，并记录 A/B/CURRENT_ONLY 分类证据。
5. Codex 为每个候选生成作用、合法值、约束、风险、关联参数和调优建议。
6. 五维 Tags 按硬件、模型、拓扑、场景和优化目标组织检索。

## Search Limits

当前 W8A8 新 Session 默认使用独立自动注册表：

```text
340 ParameterYAML
→ 225 场景召回
→ 自动 registry.generated.yaml
→ 142 自动 Registry 参数
→ 103 Tunable：28 Active + 75 Reserve
→ 39 Fixed + 0 Compiler Rejected
```

新 Session 会重新编译并把结果冻结到自身 `00_search_space/`。当前自动路径的
22 个 Active 是：

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

`automatic_registry_a8_frontier_v3` 是固定 DP4/TP8 Session 的统一默认，不读取人工注册表，而是从召回画像、固定源码和确定性兼容策略生成注册表；`curated_registry_v1` 保留为复用人工审计 `registry.yaml` 的显式可选路线。当前注册表为 28 Active + 75 Reserve；历史只有在 Benchmark、镜像、源码和完整拓扑身份均匹配时，才可在新 Session 边界最多轮换 3 个 Reserve 轴。休眠的 `glm52_w8a8_a3_topology_campaign_v4` 未来恢复时仍使用拓扑隔离的失败和 best anchor，不跨 DP/TP 初始化。

自动替代路径当前 Controller 编译结果为：103 Tunable（28 Active + 75 Reserve）→ 39 Fixed + 0 Compiler Rejected。硬失败按完整组合隔离，不再全局删除单值。

历史驱动轮换只接受 Benchmark 定义、镜像 Digest 和两个源码 commit 全部一致的 Session；不匹配的历史与上一版选择会失败关闭。

`enable_eplb=false` 与 `eplb_num_redundant_experts=0` 是固定运行契约，不是搜索轴。当前 pinned Ascend 平台不支持上游 `--enable-eplb` CLI。

## 在线状态机

- 当前默认从 A8 DP4/TP8 fixed-v3 建立新 Session 基线；B0 作为官方源码默认对照保留。
- Codex 从最佳已验收锚点出发提出候选。
- 第一层使用画像和关系证据判断候选是否合理。
- 第二层由 Python 强制执行白名单、参数组合、网格预算、重复候选和历史隔离。
- 通过后把完整候选转换为环境变量和 vLLM 命令。
- 远端成功启动服务后才允许 Benchmark。
- 指标必须通过请求完整性、Token 形状、吞吐和 TTFT/TPOT 门禁。
- 参数 OOM/非法值可以最小化修正；基础设施故障只允许同候选有限重试；未知问题暂停人工处理。

资源执行层通过独立边界接入。默认 `ktp_lab` 继续走已验证的内置路径；新 Session
可以显式加载 Executor Adapter v1，将 `prepare/check_ready/submit/snapshot/stop/
wait_for_release` 映射到其他调度器。外部适配器不能修改候选、Session 或指标判定，
其文件与配置身份随 Session 冻结。拓扑 Executor Profile 约束 rank/进程布局，
Executor Adapter 负责资源管理；两者不能互相冒充。

## 本地与远端隔离

新项目使用：

```text
Lease: vllmtkb-418bd627-32c8cf190-glm52-a3-32npu
Remote: /mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-418bd627-32c8cf190
```

Controller 的状态、锁、停止标记和实验目录都位于本项目内部，不与 `vllmTKB0706` 共用。
