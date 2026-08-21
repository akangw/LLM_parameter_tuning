# Decode-only Agent 策略：参数清单与约束

> 状态：decode-only V1 已作为独立配置落地、进入 runtime allowlist 并通过测试。它必须与固定 decode benchmark 一起创建新 Session，不能热迁移或改写旧 Session；当前尚未创建真实 Lease 或启动实验。
>
> 本文只关注固定场景 `decode-256-2048`、并发 32、DP4/TP8。首要指标为输出吞吐，TTFT/TPOT 为参考指标。

当前有效 Active 共 25 个：List 2 主搜索轴 9 个、List 1.3 条件轴 4 个、List 1.2/1.1 晚期可选轴 12 个。`speculative_config__attention_backend` 固定为 `null`，`fuse_norm_quant` 已进入晚期可选轴。
>
> 下文百分比均来自历史四场景报告中的 **decode case 单独结果**，不是四场景平均分或几何平均分。现有证据多数只有 1 次 repetition，因此标为“单次配对证据”；新策略落地前应复验关键开关。

# List 1：专家基线开关

## 1.1 默认开启

| 参数 | 默认值 | 机制原理及为何采用该值 | Decode-only 证据 |
|---|---:|---|---|
| `speculative_config__method` | `mtp` | 当前模型支持 MTP；List 2 要搜索 K，必须保留 MTP 路径。注意：历史 `+41.37%` 是 K1→K3 的收益，不能归因于 `method` 字段本身。 | K1→K3：`568.40→803.56 TPS`，单次配对。 |
| `async_scheduling` | `true` | 异步调度可减少 CPU 调度与设备执行之间的同步等待，是当前专家基线的高吞吐路径；但 MTP 本身并不强制要求它，未来若解冻仍可合法测试 `false`。 | 当前稳定高性能路径使用 `true`，暂无独立消融。 |
| `enable_expert_parallel` | `true` | 将 MoE 专家分散到不同 rank，降低单 rank 专家计算/存储压力，也是当前 FlashComm1 与 fused MC2 路径的前提。 | `501.90→555.47 TPS`，`+10.67%`，单次配对。 |
| `flashcomm1` | `true` | 减少 TP/MoE 通信及中间张量开销，但图形状必须满足 TP 整除。 | `555.52→568.40 TPS`，`+2.32%`，单次配对。 |
| `fused_mc2` | `2` | 融合 MoE dispatch、计算与 combine，减少 decode 每步通信和 kernel launch。 | K3：`803.56→846.36 TPS`，`+5.33%`，单次配对。 |
| `additional_config__ascend_compilation_config__fuse_allreduce_rms` | `true` | 目标是减少 AllReduce 后独立 RMSNorm 的 launch/访存；但在当前 `npugraph_ex=true` 路径，源码主要让它影响编译区间边界，不能把观测收益直接等同于“融合命中”。沿用 `true` 是经验基线选择，后续需用编译日志确认真实命中。 | K3+MC2：`846.36→856.58 TPS`，`+1.21%`，单次配对。 |
| `additional_config__ascend_compilation_config__enable_npugraph_ex` | `true` | decode 重复执行相近形状，图执行可减少 Python/调度与 kernel launch 开销。 | 当前高性能路径一直开启；暂无独立消融。 |
| `additional_config__ascend_compilation_config__fuse_norm_quant` | `true` | 仅在 `enable_npugraph_ex=false` 时注册 Norm→Quant 融合 pass，减少匹配图中的 launch 和中间张量流量；在默认 NPUGraph 路径中无效。 | 默认保留潜在融合收益，仅在晚期与 `enable_npugraph_ex=false` 成对探索 true/false，不做无效单扫。 |

## 1.2 默认关闭

| 参数 | 默认值 | 机制原理及为何采用该值 | Decode-only 证据 |
|---|---:|---|---|
| `enable_prefix_caching` | `false` | 当前请求前缀不复用，缓存命中收益接近零，反而增加哈希、block 与元数据管理。 | `774.68→807.39 TPS`，关闭后 `+4.22%`，单次配对。 |
| `additional_config__ascend_compilation_config__enable_static_kernel` | `false` | 动态 batch 与 MTP 接受率会改变运行形状；静态特化可能造成覆盖不足或额外编译。 | 开启：`803.56→741.81 TPS`，`-7.68%`，单次配对。 |
| `additional_config__prefill_comm_compute_overlap` | `false` | 输入只有 256 token，可重叠的 Prefill 工作很少，额外协调开销可能大于收益。 | K3 的最佳配对使用 `false`；尚无纯独立复验。 |
| `speculative_config__disable_padded_drafter_batch` | `false` | 开启后 pinned vLLM 会自动关闭 async scheduling；这不会必然启动失败，但会让实际执行路径偏离当前异步专家基线，因此默认关闭，仅在晚期有明确耦合假设时探索。 | 源码为自动降级关系，不作为通用硬非法规则。 |
| `long_prefill_token_threshold` | `0` | 只有 256-token 输入，不需要长 Prefill 特殊分支；错误阈值会增加不必要的路径切换。 | 设为 1024：`501.90→464.83 TPS`，`-7.39%`，单次配对。 |
| `TASK_QUEUE_ENABLE` | `1` | 值 2 只适合无图模式，与当前正常 NPUGraph 路径不兼容。 | 值 2 已产生图模式冲突。 |
| `compilation_enable_sp` | `false` | 图编译 SP 把 TP AllReduce+Norm 改写为 ReduceScatter/本地 Norm/AllGather，但当前 pinned Ascend 支持面主要是非量化 VL；本项目是量化 GLM 且已有 FlashComm1 路径，开启可能无效并改变图形状，还与当前 AllReduce/RMS 路径冲突，因此显式关闭。 | 当前无支持性独立收益证据；依据 pinned 支持矩阵与冲突关系固定。 |

## 1.3 允许 Agent 条件探索的开关

| 参数 | 默认值 | 机制原理 | 为什么仍是 Active | Decode-only 证据 |
|---|---:|---|---|---|
| `enable_chunked_prefill` | `true` | 把 Prefill 按 token budget 切块，使长 Prefill 不会一次占满调度批次；代价是多次调度和更复杂的图形状。 | 本场景只有 256-token Prefill，切块收益可能小；但 `false` 只有在 `max_num_batched_tokens >= max_model_len` 时合法，所以必须与 model length/token budget 联合探索。 | 无独立 true/false 消融，不能凭经验永久固定。 |
| `enable_balance_scheduling` | `false` | 每个 DP rank 交换运行请求数，并在某个 rank 饱和时限制新请求进入，可减小 DP 负载偏斜；同时增加每轮 all-gather 和排队，可能恶化 TTFT。 | DP4+C32 可能出现 rank 间不均衡，收益随 `max_num_seqs`、显存压力和 K 改变，因此保留条件联测。 | K1 近似中性；K3 关闭后 `774.68→797.50 TPS`（`+2.95%`），单次。 |
| `enable_reduce_sample` | `false` | 保持 TP 词表 logits 分片，只交换各 rank 的局部 top-k/最大值，而不是汇聚完整词表；可减少 TP8 采样通信和临时张量，但改变采样实现路径。 | TP8、长 Decode 有潜在通信收益；功能仍属实验性，必须同时校验输出 token/logprobs，并与 K、采样方式联测。 | K1 约 `-5.89%`，K3 约 `+1.77%`，方向随 K 反转。 |
| `speculative_config__enforce_eager` | `null/false` | 只让 draft/speculative 子路径使用 eager，绕开不稳定图形状或捕获限制；代价是恢复逐步 launch，通常降低吞吐。 | 不是常规提速开关，而是探索 K/图形状边界和故障归因的高风险轴；只有明确图不兼容假设时才值得测试 `true`。 | 当前最佳路径依赖图执行；暂无 `true` 的正收益证据。 |

## 1.4 已移出候选 schema 的无效轴

| 参数 | 为什么不应搜索 |
|---|---|
| `mlapo` | 它融合 MLA Q/K/V 预处理并以额外显存换取 Decode 性能，但 A3 上要求该进程是 PD 分离的 decode-only KV consumer；当前 `pd_disaggregated=false`、无 KV transfer，开关不会激活真实算子路径。 |
| `VLLM_ASCEND_ENABLE_BATCH_MEMCPY` | 它是构建 vllm-ascend 扩展时选择批量 H2D/D2H API 的编译期变量，不是服务启动后的运行时旋钮；当前也没有 KV CPU offload 流量，在线调参不会改变二进制或性能。 |
| `speculative_config__attention_backend` | 当前 profile 未启用 batch-invariant/FA3 前提；显式 `FLASH_ATTN` 通常回落到普通 Ascend 后端，与 `null` 自动选择没有独立实验意义，因此固定为 `null`。若未来建立满足 FA3 前提的新 profile，再单独解冻。 |

# List 2：Agent 重点搜索的多值域参数

| 优先级 | 参数 | 建议种子/范围 | 机制原理 | 为什么值得探索 / 当前证据 |
|---|---|---|---|---|
| P0 | `num_speculative_tokens` | `1,2,3,4` | 每步草拟 K 个 token，再由目标模型并行验证；K 大可减少目标模型迭代次数，但会增加草稿、验证和拒绝浪费。 | 接受率与成本非线性；K3 当前 `803.56 TPS`，K2/K4 未充分覆盖。 |
| P0 | `cudagraph_capture_sizes` | Agent 在冻结的高质量列表模板中自主选择并联调 | 实际 batch/token shape 会向上 padding 到最近已捕获图；密集列表减少 padding，过多图则增加编译时间和常驻显存。 | 与 K、TP8、seqs 形成离散组合，人工难穷举；低端 shape 调整单次约 `+6.01%`。Agent 可自主选模板，但不能在运行中的冻结 Search Limits 外凭空生成新值；新增模板需创建新版 profile/Session。 |
| P0 | `max_cudagraph_capture_size` | `32–256` | 决定图覆盖上界，同时影响 MoE dispatch/MC2 buffer 容量；过小导致 eager 回退，过大浪费图和通信缓冲显存。 | 显式给定时必须等于 capture list 最大值；与 K、seqs 的关系是覆盖效率问题，不应伪装成通用非法条件。暂无干净单参数消融。 |
| P0 | `max_num_seqs` | `32,48,64,80,96,128` | 限制同时调度的序列数，决定并行度、KV 占用、队列深度及图 shape 分布。 | 太小欠饱和，太大增加 KV/通信/排队；最优点随 K、显存和 capture list 改变，暂无完整孤立扫描。 |
| P0 | `max_num_batched_tokens` | `2048,2304,3072,4096,6144,8192,12288,16384` | 限制一次 scheduler batch 的 token 总量，控制设备填充率与请求合批等待。 | 过小喂不满设备，过大增大等待和临时张量；4096 是当前锚点而非理论最优，并且决定 chunked prefill=false 是否合法。 |
| P0 | `gpu_memory_utilization` | `0.88–0.97` | 决定可供 KV cache 使用的显存比例；提高可容纳更多序列，但挤压图、通信 buffer、临时激活和碎片余量。 | 与 model_len/seqs/capture 强耦合，接近 OOM 边界时收益和风险都大；当前无完整联合扫描。 |
| P0 | `max_model_len` | `2304,3072,4096,8192,16384,32768,64000` | 规定服务上下文上限，并参与 KV cache 容量几何；即使单请求只有 2304 token，更大上限也可能减少可并发 block。 | 固定 workload 下 2304 已满足能力，64k 是旧服务锚点；应直接测服务能力与吞吐的代价。 |
| P1 | `compilation_mode` | 以 `FULL_DECODE_ONLY` 为主，允许合法替代 | 决定哪些阶段/shape 使用完整图、分段图或 eager，影响 launch 开销、动态 shape 容忍度、编译时间和图内存。 | 当前最佳路径依赖 Decode 图，但 K/capture 改变后其他模式可能更稳；必须与 capture list 联测。 |
| P1 | `disable_hybrid_kv_cache_manager` | `null,false,true` | 控制是否使用 hybrid KV 管理策略；会改变不同层/注意力类型的 block 分配与回收方式。 | 长输出下可能改变 KV 碎片、容量与 TPOT，效果取决于 model_len 和显存压力；当前无 GLM decode 独立证据。 |

## 推荐探索组合（软先验，非白名单、非强制）

- `K × compilation_mode × capture_sizes × max_capture × max_num_seqs`：K 改变每轮 decode 的 token shape，图模式和捕获集合决定这些 shape 是否命中图；这是最直接的高收益离散耦合组。
- `max_model_len × gpu_memory_utilization × max_num_seqs × max_num_batched_tokens`：四者共同决定 KV 容量、可调度并发和临时显存余量，适合寻找接近 OOM 边界但仍稳定的高吞吐点。
- `enable_chunked_prefill × max_model_len × max_num_batched_tokens`：是否切块的合法性和收益都由上下文上限与 token budget 决定，不能脱离容量参数单扫。
- `balance_scheduling × max_num_seqs × max_num_batched_tokens × K`：DP 负载均衡是否值得额外通信，取决于队列饱和度与每轮 MTP 工作量。
- `hybrid_kv_manager × max_model_len × gpu_memory_utilization × max_num_seqs`：KV 分配策略只有在容量压力下才可能显著改变碎片和可并发数。
- `reduce_sample × K`：采样通信收益可能随 MTP 深度变化；当前 benchmark 的 greedy 采样方式固定，不应虚构一个并不存在的 Active“sampling mode”轴。

这些组合只提高 Agent 的关注优先级。Agent 可以只改其中一个或部分参数、跨组组合，也可以提出未列出的合法组合；参数数量由 Agent 根据历史和因果假设决定。Controller 不能把它们变成候选白名单。

候选默认只测一次以保持收敛速度；若一个候选可能成为新 best、但相对基线增益仅在 `1%–3%`，Controller 会自动追加一次相同候选复测。两次都通过确定性门槛后，才用两次输出吞吐的中位数更新 best anchor。增益达到 `3%` 及以上的候选不触发这项选择性复测。

# List 3：核心约束与实现状态

本节只保留核心摘要；完整、可审计的硬规则以 [`DECODE_ONLY_HARD_RULES.md`](DECODE_ONLY_HARD_RULES.md) 为唯一总表。

## 3.1 硬约束：明确会启动失败或不满足固定服务契约

1. 固定 `decode-256-2048` workload 要求 `max_model_len >= 2304`；低于该值无法承载本场景请求。
2. pinned vLLM 要求 `max_num_batched_tokens >= max_num_seqs`；若关闭 chunked prefill，还要求 `max_num_batched_tokens >= max_model_len`。
3. 图模式下若显式提供 capture list，它必须非空且只含正整数；显式 `max_cudagraph_capture_size` 必须等于列表最大值。
4. Full Graph 且 `K>1` 时，Controller 要求显式 capture size 都能被 `K+1` 整除。运行时本可向上取整，但这会静默改变 Agent 提交的实验画像；因此这是保证实验可解释性的显式列表规范，不应扩大成所有图模式或 K=1 的引擎通用禁令。
5. FlashComm1 要求 `TP>1`；当前模型是 MoE，因此还要求 EP=true。在 Full Decode Graph + MTP 下，显式 `compilation_enable_sp=true` 或 FlashComm1=true 都会让 pinned Ascend v1 runner 在上游图检查期间启用有效 SP，进而触发 vLLM 的 `max(K+1, TP)` 互整除检查；共同倍数列表不能绕过该检查。归一化并经过 Ascend TP 过滤后的图列表还必须非空。
6. fused MC2 与 MC2 hierarchy communication 不能同时开启；pinned Ascend 会直接抛出启动错误。
7. 当前已观测的 torch_npu 路径中，`TASK_QUEUE_ENABLE=2` 与图捕获冲突，因此值 2 只允许无图且无显式 capture list。
8. 上游 `enable_eplb/eplb_num_redundant_experts` 不是当前 Ascend 原生 dynamic EPLB 接口，已从候选 schema 移除，启动器不得误发对应 CLI。
9. `enable_static_kernel=true` 要求 `enable_npugraph_ex=true`；fallback fusion-pass compiler 不支持静态 kernel 配置，会在初始化阶段断言失败。

以上规则均已在 Controller、编译约束或候选 schema 边界实现；不再把 `batch_tokens >= seqs×(K+1)`、MTP 必须 async、`fused_mc2=2` 必须 K>0 等“经验路径”误写成引擎硬约束。

## 3.2 归一化与无效候选剪枝：不是非法，也不代表高风险

1. capture list 的乱序和重复会被 vLLM 去重排序。Full Decode Graph + MTP + 有效 SP 时，vLLM 先把 shape 向上取整到 `max(K+1, TP)` 的倍数；Ascend 随后还会过滤剩余的非 TP 倍数，并同步更新有效最大值。最终列表为空才是硬失败。Agent 应优先提交已经对齐的列表，避免候选画像与实际执行配置不一致。
2. `FULL_DECODE_ONLY` 会过滤低于 `K+1` 或高于 `max_num_seqs×(K+1)` 的不可达 shape。超界值不是启动错误，只是没有独立实验意义；推荐组合应避免浪费预算。
3. `disable_padded_drafter_batch=true` 会让 pinned vLLM 关闭 async scheduling，而不是必然报错；当前固定 `false` 是为了保持专家基线执行路径一致。
4. `fused_mc2=2` 在没有 speculative decode 时可能不激活相应 fused 路径，但这属于无效/退化配置，不是通用非法组合；当前 decode profile 已固定 MTP 且 K≥1。
5. `fuse_norm_quant` 只有在 `enable_npugraph_ex=false` 的 fallback fusion-pass compiler 中生效；它保留为晚期条件轴，Agent 不应在 `enable_npugraph_ex=true` 时单独消耗轮次。

## 3.3 保留给 Agent 的高风险合法空间

1. 高 `gpu_memory_utilization`、大 `max_model_len`、大 seqs/batch budget、稀疏或大型 capture 集合可以触碰 OOM、编译时间和延迟边界，但只要不违反上述硬条件就不得预先封禁；失败后只隔离精确组合。
2. Agent 可自由跨越推荐耦合组或提出未列出的多参数组合；推荐组仅是因果优先级，不是白名单，也不规定每轮必须改几个参数。
3. `reduce_sample`、draft eager、balance scheduling 等实验允许方向反转或失败；Controller 只做兼容性检查，收益由输出吞吐决定，TTFT/TPOT 与输出一致性进入报告。
4. 修改 K 后只需重新校验受影响的图约束，不要求无条件重生成全部参数；TP8+K4 的步长 40 只是静态合法示例，不等于实测有效。
5. 精确重复候选会拦截，已证实失败的完整组合可隔离；不得因一次联合失败就永久封禁其中任意单值。
