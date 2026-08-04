---
name: tune-vllm-ascend
description: Guide for performance tuning vLLM on Ascend NPU using the parameter knowledge base. Use when the user wants to optimize vLLM inference performance on Ascend hardware — deployment configuration, OOM troubleshooting, throughput/latency tuning, or parameter selection for a specific model and hardware setup.
---

# vLLM Ascend 性能调优

基于多维标签检索 + 迭代式规划的 vLLM 昇腾 NPU 推理性能调优。

## 核心原则

1. **标签驱动** — 部署画像 → 标签指纹 → 检索 → 分析 → 出方案，不预设固定的 phase 流程。
2. **约束优先** — 违反 `constraints` 会导致崩溃或未定义行为，必须在推荐值之前校验。
3. **以 KB 为准** — 每个推荐必须引用具体 YAML 文件中的 `impact_detail` / `suggested_values` / `quick_guide` 作为证据。
4. **Ascend 优先** — vllm-ascend 可能静默覆盖上游 vLLM 参数。当参数同时出现在两个 scope 中时，以 ascend 版本为准。
5. **区分 v1/v2 worker** — `usage_locations` 中的版本标签（v1-ascend / v2-ascend）必须关注。

## 标签体系

| 维度 | 含义 | 取值 |
|------|------|------|
| `model` | 模型架构 | dense, moe, mla, vlm, quantized |
| `optimize_target` | 优化目标 | ttft, tpot, throughput, memory |
| `deploy_topology` | 部署拓扑 | single_node, multi_node |
| `hardware` | 硬件平台 | a2, a3 |
| `deploy_scenario` | 部署场景 | long_input, long_output, high_concurrency |

通过标签交集锁定相关参数。全部维度均适用时全选，不用 `[]`。

## 工具

```bash
# 查看可用标签
python query.py --list-tags

# 标签检索
python query.py -t model=moe -t hardware=a3 -t optimize_target=throughput \
  --show name,performance_impact,tuning_advice.quick_guide

# 组合检索（标签 + 性能等级 + 全文搜索）
python query.py -t model=moe,mla -t deploy_scenario=long_input -w performance_impact=high -s "context" \
  --show name,category,impact_detail

# 组合检索（标签 + category 分类聚焦）
python query.py -t model=moe -t hardware=a3 -w category=compilation \
  --show name,default,tuning_advice.quick_guide

# 读取完整参数
python query.py -w name=--tensor-parallel-size -a --format yaml

# 列表格式（管道用）
python query.py -t model=moe -t optimize_target=memory --format list

# 查看所有可查字段
python query.py --list-fields
```

---

## 工作流

### Step 1 — 部署画像 → 标签指纹

从用户处收集信息。缺失的字段主动提问，不可猜测。然后将信息映射为标签。

#### 硬件检测

**在询问 NPU 型号之前，必须先尝试自动检测：**

```bash
python -c "from vllm_ascend.utils import get_ascend_device_type; print(get_ascend_device_type())"
```

失败时才向用户提问。

#### 模型配置检测（第一步，最高优先级）

**在映射任何标签之前，必须读取模型的 `config.json` 来确定准确的架构信息。** 模型名称具有误导性（如 GLM-4.7 实际是 MoE 架构），不可根据名称推断。

```bash
cat <model_path>/config.json
```

从 `config.json` 中提取以下关键字段，所有字段均可直接影响标签映射和后续参数推荐：

| config.json 字段 | 用途 | 判断逻辑 |
|------------------|------|----------|
| `architectures` | 模型架构类型 | 包含 `Moe`/`MoE` → MoE；包含 `ForCausalLM` 且无 MoE → dense；`DeepseekV3`/`DeepSeekV3` → mla |
| `model_type` | 辅助确认架构 | 含 `moe` → MoE；`deepseek_v3` → mla；`qwen2_vl` 等 → vlm |
| `quantize` / `quantization_config` | 量化方式 | 存在任一字段 → `quantized` 标签 |
| `num_nextn_predict_layers` | MTP/投机解码 | > 0 → 模型支持 MTP，后续需关注 speculative 类参数 |
| `n_routed_experts` / `num_experts` | MoE 专家数 | 存在且 > 0 → 确认 MoE 架构 |
| `num_attention_heads` / `num_key_value_heads` | 约束校验 | 用于验证 TP 整除约束（num_attention_heads % tp_size == 0） |
| `head_dim` / `hidden_size` / `num_hidden_layers` | 模型规模 | 辅助判断显存需求和并行策略 |

**检查清单**（每项都要输出到部署画像中）：
- [ ] 架构类型：dense / moe / mla / vlm
- [ ] 是否量化：W8A8 / FP8 / W4A8 / 无
- [ ] 是否支持 MTP：num_nextn_predict_layers > 0
- [ ] Attention heads：num_attention_heads / num_key_value_heads（GQA 比率）
- [ ] 约束预检：TP 能否整除 num_attention_heads

#### 收集信息 → 映射标签

**`model` 维度的标签是叠加关系而非替代关系**。例如 DeepSeek-V3 W8A8 应标为 `model=moe,mla,quantized`（既是 MoE + MLA 架构，又是量化模型）。不要因为选了 `quantized` 就丢掉架构标签。

| 架构检测结果 | → 标签 |
|-------------|--------|
| architectures 含 `Moe`/`MoE` 或 model_type 含 `moe` | `model=moe` |
| architectures 含 `DeepseekV3`/`DeepSeekV3` 或 model_type 含 `deepseek` | `model=mla` |
| architectures 为 `ForCausalLM`/`ForConditionalGeneration` 类且非 MoE/MLA | `model=dense` |
| 多模态模型（Qwen2-VL、InternVL 等） | `model=vlm` |
| config.json 中存在 `quantize` 或 `quantization_config` 字段 | `model=quantized`（叠加到架构标签上） |

| 部署信息 | → 标签 |
|---------|--------|
| DeepSeek-V3 | `model=moe,mla` |
| Qwen / Llama dense | `model=dense` |
| 多模态模型（Qwen2-VL 等） | `model=vlm` |
| W8A8 / W4A8 / FP8 / INT8 量化模型 | `model=quantized`（叠加到架构标签上） |
| NPU 910B / 910B2 | `hardware=a2` |
| NPU 910B3 / A3 系列 | `hardware=a3` |
| 单机 | `deploy_topology=single_node` |
| 多机 RDMA / RoCE | `deploy_topology=multi_node` |
| 优化吞吐 | `optimize_target=throughput` |
| 优化首 token 延迟 | `optimize_target=ttft` |
| 优化逐 token 延迟 | `optimize_target=tpot` |
| 优化显存 | `optimize_target=memory` |
| 输入 > 8K tokens | `deploy_scenario=long_input` |
| 输出 > 8K tokens | `deploy_scenario=long_output` |
| 高并发 / 大批量 | `deploy_scenario=high_concurrency` |

构建标签前先跑一次 `python query.py --list-tags`，确认当前 KB 中有哪些标签值可用及各值覆盖的参数数量。

#### 确认部署画像

向用户展示整理好的部署画像和对应的标签指纹，确认无误后进入 Step 2。

#### 知识库范围说明

KB 覆盖**影响推理运行时性能**的参数，以下类型**不在 KB 范围内**：
- 系统级基础设施变量（`TASK_QUEUE_ENABLE`、`OMP_NUM_THREADS`、`jemalloc` / `LD_PRELOAD` 等）
- 网络接口配置（`HCCL_IF_IP`、`GLOO_SOCKET_IFNAME` 等）
- 容器/进程管理参数（`--seed`、`--trust-remote-code`、`--port` 等）
- 厂商临时调优 knob（`pa_shape_list` 等）

这些参数仍需按官方文档配置，标签检索不会返回它们。

---

### Step 2 — 标签检索 → 分析 → 出方案

用标签指纹检索参数，分析结果，制定调优方案。检索前先确认 KB 中有哪些标签可用（`python query.py --list-tags`），避免使用无人用的标签值。

#### 检索策略

用全套标签 + 高影响参数进行首次检索：

```bash
python query.py -t model=<mapped> -t hardware=<mapped> -t deploy_topology=<mapped> \
  -t optimize_target=<mapped> -t deploy_scenario=<mapped> \
  -w performance_impact=high \
  --show name,category,type,tuning_advice.quick_guide
```

检索后，自行规划如何探索这些参数。
对感兴趣的具体参数读完整 YAML：

```bash
python query.py -w name=<param-name> -a --format yaml
```

#### 分析要点

1. **constraints** — 硬约束是否与部署画像冲突？（TP 能否整除 attention heads、显存是否够、硬件是否支持）
2. **suggested_values** — 有没有匹配当前部署场景的推荐值？
3. **caveats** — 注意事项中是否有与当前配置冲突的限制？
4. **usage_locations** — 参数在哪个 worker 版本（v1/v2）生效？
5. **deprecated** — 是否已弃用？

#### 出方案

根据分析结果自行规划方案结构，比如标签检索出来40个参数，则可以按优先级每次推荐 3-5 个参数实测根据反馈不断迭代。每个参数注明标签和 KB 来源。

---

### Step 3 — 迭代优化

用户实施方案后，根据反馈自行规划下一步：调整参数组合及取值、标签重新检索、或定位问题根因。迭代直到用户满意为止。

---

## 决策规则

### 约束校验（必做）
推荐的参数组合的constraints必须经过校验，保证不出现明确的互斥。

### 冲突处理

当参数建议相互矛盾时：
1. vllm-ascend 分析结论优先（Ascend 插件覆盖上游行为）
2. 以 `constraints` 更严格的为上限
3. 通过 `related_parameters` 确认语义关系
4. 仍无法判定时，向用户说明冲突点

### 不要做的事情

- **不要**一次性推荐 20+ 个参数，检索出来的参数很多的话，可以规划计划分优先级探索
- **不要**在没有 `suggested_values` 或 `impact_detail` 支撑的情况下给具体数值
- **不要**跳过约束校验
- **不要**在用户没有提供模型/硬件信息的情况下推荐配置

## 参数展示规范

```
### [领域] 参数名
- **推荐值**: xxx
- **理由**: [引自 YAML]
- **标签**: model=x, optimize_target=x, hardware=x, deploy_scenario=x
- **约束检查**: [通过 / 警告 + 内容]
- **适用版本**: [v1-ascend / v2-ascend / 通用]
- **KB 来源**: tag_params/output/params/<filename>.yaml
```
