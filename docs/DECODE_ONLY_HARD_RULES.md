# Decode-only 自治调优硬规则总表

本文汇总当前 `decode-256-2048`、DP=4/TP=8、固定 Ascend 镜像自治链路中，会在提交前、进程启动前或状态迁移前**强制拦截**的规则。生产入口是 `server_autonomous/config.dp4_tp8.decode_priority_v1.yaml`：runtime=`glm52_w8a8_a3_dp4_tp8_decode_priority_v1`、Search Limits=`automatic_registry_decode_priority_v2`、策略=`decode_priority_agentic_v1`、Benchmark=`decode_only_c32_v1`。它不收录收益倾向、推荐组合和 Agent 软提示。

`K` 表示 MTP 每轮预生成的候选 token 数，`K+1` 表示一次验证 decode 实际处理的 token 位置数。

## 1. 候选结构与服务能力

| 编号 | 触发条件 | 强制规则 | 拦截原因 |
|---|---|---|---|
| C01 | 所有候选 | 参数字段必须与冻结 Session 的 candidate schema 完全一致，不能缺字段或增加未知字段。 | 防止新旧参数名、遗漏注入和版本漂移。 |
| C02 | 所有候选 | 每个值必须存在于该 Session 冻结的 Search Limits；Active 参数还必须存在于自动注册表的编译值域并能成功渲染注入。 | Agent 可以自主组合，但不能提交未经当前镜像和注入契约验证的任意值。 |
| C03 | 当前 benchmark | `max_model_len >= 最大输入 token + 最大输出 token`；当前固定场景即 `max_model_len >= 2304`。 | 否则服务能力无法覆盖 256 输入、2048 输出。 |
| C04 | 所有候选 | `max_num_batched_tokens >= max_num_seqs`。 | pinned vLLM 的 scheduler 基本容量约束。 |
| C05 | `enable_chunked_prefill=false` | `max_num_batched_tokens >= max_model_len`。 | 关闭分块预填充后，一个最长请求必须能由单次 token budget 承载。 |
| C06 | `K>0` | 必须配置有效的 MTP draft model 精确路径；启动任务只读取其 `config.json`，校验文件身份及 `n_predict`。若配置含 `n_predict` 且 `K>n_predict`，则必须满足 `K % n_predict == 0`。 | pinned vLLM 会复用 MTP 模块，不满足该整除关系会在服务初始化时失败。预检把它提前为可解释的启动错误。 |
| C07 | `K>0` | 当前 Ascend fused TND decode 路径要求 `K<=15`。 | 超出实现支持的推测深度会确定性失败。 |

## 2. 拓扑、并行与 MoE

| 编号 | 触发条件 | 强制规则 | 拦截原因 |
|---|---|---|---|
| T01 | 候选包含 decode context parallel | `TP % decode_context_parallel_size == 0`。 | DCP 复用 TP rank，必须整除 TP。 |
| T02 | 候选包含 prefill context parallel | `DP × TP × PP × prefill_context_parallel_size <= 总设备数`。 | 物理并行规模不能超过可用设备。 |
| T03 | `enable_balance_scheduling=true` | `DP > 1`。 | 单 DP 没有跨副本负载均衡对象。 |
| T04 | `flashcomm1=true` | `TP > 1`。 | FlashComm1 依赖张量并行通信。 |
| T05 | MoE 模型且 `flashcomm1=true` | `enable_expert_parallel=true`。 | 当前 MoE FlashComm1 路径依赖专家并行布局。 |
| T06 | `additional_config__enable_shared_expert_dp=true` | `enable_expert_parallel=true`。 | shared expert DP 依赖专家并行。 |
| T07 | 同一候选 | `additional_config__mix_placement` 与 `additional_config__enable_shared_expert_dp` 不能同时为 true。 | 两种专家放置方式冲突。 |

当前主流程的 DP4/TP8 是 Session 固定拓扑，不是 Agent 本轮搜索轴；切换拓扑必须新建 Session，不能复用原拓扑的状态和失败隔离记录。

## 3. MTP、序列并行与图捕获

| 编号 | 触发条件 | 强制规则 | 拦截原因 |
|---|---|---|---|
| G01 | 显式提供 capture list | 图模式开启时列表不能为空；任何非空列表都只能包含正整数。 | 空列表或非法 shape 无法形成有效图捕获配置。 |
| G02 | 显式提供非空 capture list 和最大值 | `max_cudagraph_capture_size == max(cudagraph_capture_sizes)`。 | 防止列表与声明上限不一致。 |
| G03 | Full Decode Graph 且 `K>1` | 每个显式 capture size 必须是 `K+1` 的倍数。 | 这是 Controller 的实验画像规范：运行时虽可向上取整，但会静默改变 Agent 提交的 shape，破坏实验归因。它不是所有图模式下的通用引擎禁令。 |
| G04 | Full Decode Graph、`K>0`、`TP>1`，且有效 SP 生效（显式 `compilation_enable_sp=true` 或 FlashComm1=true） | `(K+1) % TP == 0` 或 `TP % (K+1) == 0`。 | pinned Ascend v1 runner 会在上游图检查前把 FlashComm1 映射为有效 SP；pinned vLLM 使用 `max(K+1, TP)` 而不是二者最小公倍数作为图形状步长，互不整除会在处理 capture list 前确定性失败。 |
| G05 | 与 G04 相同 | `max_cudagraph_capture_size >= max(K+1, TP)`。 | 最大图尺寸必须至少容纳运行时图形状步长。 |
| G06 | 图模式未关闭、`TP>1`，且运行时有效 SP/FlashComm1 开启 | 经 MTP 图尺寸归一化后，capture list 必须至少保留一个能被 TP 整除的值。 | pinned Ascend 随后会过滤非 TP 倍数，并在结果为空时直接断言失败；不要求原列表的所有值都是 TP 倍数，但 G04 生效时仍须先满足互相整除。 |
| G07 | `TASK_QUEUE_ENABLE=2` | 必须同时满足 `compilation_mode=NONE` 且 `cudagraph_capture_sizes=null`。 | 当前 torch_npu 路径中任务队列模式 2 与图捕获冲突。 |
| G08 | `TORCH_COMPILE_DISABLE="1"` | compilation config mode 必须处于未启用状态。 | 禁用 torch compile 与请求编译互相冲突。 |
| G09 | target `enforce_eager=true` | `enable_npugraph_ex` 不能同时为 true。 | target eager 与 NPU 图执行互斥；draft `speculative_config__enforce_eager` 是另一条独立路径。 |

TP=8 且显式 `compilation_enable_sp=true` 或 FlashComm1=true 时，G04 示例：K=1（K+1=2）合法，K=3（K+1=4）合法，K=2（3）和 K=4（5）非法。即使 K=4 使用 40/80/120 等 5 与 8 的共同倍数作为图尺寸，当前实现仍会在读取列表前失败。

## 4. 功能依赖与确定性冲突

| 编号 | 触发条件 | 强制规则 | 拦截原因 |
|---|---|---|---|
| F01 | `speculative_config__method` 有效 | `num_speculative_tokens > 0`。 | 开启推测解码方法必须给出推测深度。 |
| F02 | `num_speculative_tokens > 0` | `speculative_config__method` 不能为 null。 | 有推测深度必须有对应方法。 |
| F03 | `enable_eplb=true` | 当前一律拒绝；同时 `eplb_num_redundant_experts>0` 必须先有 `enable_eplb=true`。 | pinned Ascend 不支持上游 `--enable-eplb` CLI；原生 dynamic EPLB 尚未接入该字段。因此当前有效契约等价于 false/0。 |
| F04 | `fused_mc2 != 0` | 不能同时开启 MC2 hierarchy communication。 | pinned Ascend 两条 MC2 通信路径冲突。 |
| F05 | `enable_static_kernel=true` | `enable_npugraph_ex=true`。 | 当前编译器的 static kernel 路径依赖 NPUGraph Ex。 |

## 5. Agent 提案与搜索流程硬门槛

这些规则是 Controller 的实验治理，不是 vLLM 引擎限制。

1. 新 Session 必须先成功测量冻结基线，之后才进入性能搜索；失败恢复不受正常分层优先级限制。
2. 分层阶段由 Controller 给出当前层，但层内参数和值由 Agent 决定。候选至少要改变当前层的一个参数，允许携带跨层 companion；当前未启用 `skip_layer`。
3. `decode_priority_agentic_v1` 正常探索每轮允许 1–4 个独立参数；单参数最多跨 3 个网格步，合计最多 10 个网格步。具体有序层可以进一步收紧参数数量。
4. List 1.2/1.1 可选开关不能在有序分层阶段消耗轮次，只能在自主跨层阶段由 Agent 基于证据选择；成功测量总额度为 4 次。失败恢复不消耗该额度。
5. Agent 声明的 changes 必须与候选实际差异完全一致；每项 before/after 必须正确且理由可审计。多参数候选必须给出耦合关系、参数证据和约束检查，不能只提交结论。
6. 关闭已启用的 MTP 只能标记为 `diagnostic_ablation`，且每个 Session 最多一次；不能伪装成常规性能优化。
7. 与历史中完整失败候选完全相同的组合会被隔离；只有部分字段相同不会全局封禁单参数或单值。
8. 正常结束前必须完成四个有序探测层，并至少获得 8 次成功的自主跨层测量。失败启动和不完整 benchmark 不计入完成门槛。
9. 相对基线提升在 1%–3% 的潜在新 best 必须成功复测两次后，才用两次吞吐中位数更新 best；更高增益不触发该选择性复测。

## 6. 失败恢复硬门槛

1. `retry_same`、`adjust_parameters`、回滚等自动动作必须标记 `safe_to_automate=true`；`pause_for_human` 必须为 false，且只能用于已经证明需要人工处理的终态类别。
2. `adjust_parameters` 至少包含一个有效 Active 参数变化或 Recovery Registry 变化。恢复时允许精确的一参数修复，不受正常探索“推荐多参数联调”的限制。
3. Recovery Registry 只能修改登记字段，并且只能使用登记值域；变更数量不能超过配置上限，before/after 和理由必须真实完整。
4. 失败修复不能再次提交一个已经失败且没有已知成功证据的完整候选；已知成功候选可以作为回滚目标。
5. 运行时规则库只隔离具有完整条件和证据的精确组合；Agent 提议的新规则在达到晋升条件前只是 proposal，不能直接变成全局硬禁令。
6. HCCL、节点、benchmark harness 等可恢复基础设施故障使用各自有界重试预算；预算耗尽后进入 Agent 诊断，而不是把相关调优参数永久移出 Search Limits。

## 7. 部署、Benchmark 与版本身份硬门槛

1. 服务镜像必须通过 activation approval，并与冻结 manifest 的仓库、tag、digest 一致。
2. Benchmark 镜像必须用 digest 固定；served model 名称和服务端口必须与部署配置一致。
3. 自治输出路径必须是服务器绝对路径，并保持在授权写目录下；不得把任务写到授权范围外。
4. Session 冻结 image、vLLM commit、vLLM-Ascend commit、benchmark、模型、拓扑和 executor/runtime adapter 身份。MTP 启动预检另外冻结 draft model 路径、`config.json` SHA-256 和 `n_predict`；权重不由预检读取。任一身份变化都必须新建 Session，禁止直接续用旧状态。
5. 历史结果只有在上述完整身份一致时才能用于初始化或避免重复；不匹配历史只能作为人工参考，不能直接作为 best anchor。
6. 基线复用要求 benchmark mode、完整 candidate schema 和唯一 `round_000` 均一致，并且该轮同时具备候选参数与有效指标。

## 8. 明确不是硬规则

以下内容不得用于预先封禁高风险探索：

- “只要 K 改变就必须同时改固定的一组参数”——不是；Agent 可单改或自由耦合，只需候选整体合法且有依据。
- “只要存在共同倍数就总能绕过 G04”——不对：显式 `compilation_enable_sp=true` 或 FlashComm1=true 都会在读取列表前触发互相整除检查；当前实现使用 `max(K+1, TP)`，不会改用最小公倍数。
- “所有原始 capture size 必须是 TP 的倍数”——不是硬规则。Full Decode Graph 会先按 `max(K+1, TP)` 向上取整，Ascend 再过滤仍不满足 TP 对齐的值并更新有效最大值；真正的硬规则只是最终列表不能为空。
- “MTP 必须开启 async scheduling”——当前 decode-only policy 已明确禁用这条旧偏好约束。
- “unpadded drafter 必须关闭 async scheduling”——当前不作为通用硬约束；实际行为和当前固定路径另行评估。
- “fused MC2 开启就必须 EP=true”——当前 decode-only policy 不把它作为引擎硬约束；但 fused MC2 与 hierarchy communication 的冲突仍是 F04。
- `max_num_batched_tokens >= max_num_seqs × (K+1)`——当前无 LoRA 场景不是通用硬约束；真正强制的是 C04，以及关闭 chunked prefill 时的 C05。
- TTFT、TPOT 的建议阈值——当前是报告指标和诊断参考，不是淘汰候选的硬门槛；主目标仍是输出吞吐。
- 推荐耦合组、List 优先级和“通常应打开”的开关——它们是搜索优先级，不是候选白名单。
- `fuse_norm_quant=true` 必须配 `enable_npugraph_ex=false`——这是避免无效单扫的软指导，不是服务启动硬约束；Controller 不应因此拒绝完整候选。

## 9. 实现位置

- 候选与图约束：`tuning_pipeline/workflow/continuous/continuous_tuning.py::validate_candidate_invariants`
- Agent 提案约束：`continuous_tuning.py::validate_candidate`、`validate_change_evidence`
- 失败恢复约束：`continuous_tuning.py::validate_failure_decision`
- 机器约束：`tuning_pipeline/workflow/search_space_compiler/compiler.py::machine_constraints`
- 组合兼容约束：`tuning_pipeline/workflow/registry_builder/compatibility_policy.decode_only_v1.yaml`
- 当前策略门槛：`tuning_pipeline/workflow/continuous/strategy_profiles.yaml::decode_priority_agentic_v1`
- MTP draft 配置预检：`tuning_pipeline/workflow/continuous/remote/validate_mtp_model.py`
- pinned vLLM 图形状实现：`portrait_pipeline/sources/vllm/vllm/config/compilation.py::adjust_cudagraph_sizes_for_spec_decode`
- pinned vLLM SP 尺寸过滤：`portrait_pipeline/sources/vllm/vllm/config/vllm.py::update_sizes_for_sequence_parallelism`
- pinned Ascend 非空断言：`portrait_pipeline/sources/vllm-ascend/vllm_ascend/platform.py`
- pinned vLLM MTP `n_predict` 校验：`portrait_pipeline/sources/vllm/vllm/config/speculative.py`

本表描述的是当前固定版本。未来升级镜像、源码或拓扑时，必须重新验证来源并以新 Session 固化，不能把本表无条件外推到其他版本。
