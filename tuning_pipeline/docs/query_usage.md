# query.py — 参数知识库查询工具

对 vLLM / vllm-ascend 性能调优知识库进行筛选、搜索、标签检索、投影和统计的 CLI 工具。

## 快速上手

```bash
cd ./tuning_pipeline

# 统计知识库规模
python query.py --count

# 列出所有可查询字段及其类型
python query.py --list-fields

# 列出所有可用标签维度及取值分布
python query.py --list-tags

# 查看高影响参数（默认表格格式）
python query.py -w performance_impact=high
```

## 筛选参数

### `-w / --where` — 字段筛选

语法：`-w 字段名=值1,值2,值3`

```bash
# 按性能影响等级筛选
python query.py -w performance_impact=high
python query.py -w performance_impact=high,medium   # 同时查 high 和 medium（OR）

# 按功能分类筛选
python query.py -w category=memory
python query.py -w category=memory,compilation

# 按来源范围筛选
python query.py -w scope=vllm-ascend     # 只看昇腾插件参数
python query.py -w scope=vllm             # 只看上游 vLLM 参数

# 按参数类型筛选
python query.py -w type=cli               # 命令行参数
python query.py -w type=env               # 环境变量
python query.py -w type=nested            # 嵌套 JSON 配置

# 精确查找某个参数
python query.py -w name=--tensor-parallel-size
python query.py -w name=VLLM_ASCEND_ENABLE_FLASHCOMM1
```

多个 `-w` 之间是 **AND** 关系，单个 `-w` 内逗号分隔是 **OR** 关系：

```bash
# Ascend 编译类参数（AND）
python query.py -w scope=vllm-ascend -w category=compilation

# 高影响的 env 或 cli 参数
python query.py -w performance_impact=high -w type=env,cli
```

### 嵌套字段筛选

支持点号访问嵌套字段：

```bash
# 查找有特定推荐的参数
python query.py -w tuning_advice.quick_guide=TP

# 查找影响延迟的参数
python query.py -w performance_scope=latency
```

### 按弃用状态筛选

```bash
python query.py -w deprecated=true   # 已弃用的参数
```

### `-t / --tag` — 标签检索

按多维标签筛选参数。语法：`-t 维度名=取值1,取值2`

标签维度包括：`model`、`optimize_target`、`deploy_topology`、`hardware`、`deploy_scenario`。可用 `--list-tags` 查看所有可用取值。

```bash
# 查所有 MoE 相关的参数
python query.py -t model=moe

# 查长输入场景且涉及 TTFT 优化的参数（AND）
python query.py -t deploy_scenario=long_input -t optimize_target=ttft

# 同维度内 OR：TTFT 或 TPOT 相关
python query.py -t optimize_target=ttft,tpot

# 查 A3 硬件上的高影响参数（tag + where 组合）
python query.py -t hardware=a3 -w performance_impact=high

# 查多机部署 + 高并发 + MoE 的参数
python query.py -t deploy_topology=multi_node -t deploy_scenario=high_concurrency -t model=moe
```

多个 `-t` 之间是 **AND**，单个 `-t` 内逗号分隔是 **OR**，与 `-w`、`-s` 也是 **AND**。

标签检索的本质是检查标签值是否存在于参数 `tags` 字段的对应维度列表中，而非字符串模糊匹配。

### `-s / --search` — 全文搜索

不指定字段，在所有文本中搜索（不区分大小写）：

```bash
python query.py -s "OOM"              # 搜索包含 OOM 的参数
python query.py -s "FlashComm"        # 搜索 FlashComm 相关内容
python query.py -s "KV cache"         # 短语搜索

# 全文搜索可和字段筛选、标签检索组合（AND）
python query.py -w category=memory -s "FP8"
python query.py -t model=moe -s "EPLB"
```

## 显示控制

### `--show` — 选择展示字段

默认展示：`name,type,category,scope,performance_impact,tuning_advice.summary`

标签相关字段通过 `tags.<dimension>` 点号访问：

```bash
# 查看模型和硬件标签
python query.py -w performance_impact=high --show name,tags.model,tags.hardware

# 查看标签 + 调优建议
python query.py -t model=moe --show name,tags.optimize_target,tuning_advice.quick_guide
```

### 常用字段组合

| 场景 | `--show` 参数 |
|------|--------------|
| 快速调优参考 | `name,tuning_advice.quick_guide` |
| 评估影响面 | `name,performance_impact,impact_detail` |
| 标签检索结果 | `name,tags.model,tags.optimize_target,tags.deploy_scenario` |
| 了解约束 | `name,default,valid_choices,constraints` |
| 排障溯源 | `name,source_file,usage_locations` |
| 参数关系梳理 | `name,related_parameters` |
| 场景推荐 | `name,tuning_advice.suggested_values` |

### `--format` — 输出格式

所有格式默认只展示 `--show` 指定的字段。需要全量字段时加 `-a` / `--show-all`。

```bash
# table — 对齐表格（默认），适合浏览多项结果
python query.py -w category=compilation --format table

# yaml — YAML 文档，适合管道给其他工具
python query.py -w name=--tensor-parallel-size --format yaml

# summary — 分块展示，适合深读单个参数
python query.py -w name=VLLM_ASCEND_ENABLE_FLASHCOMM1 --format summary

# list — 纯参数名列表，适合管道给其他命令
python query.py -t model=moe --format list
python query.py -t deploy_scenario=long_input --format list | wc -l
```

### `-a / --show-all` — 展示所有字段

覆盖默认的 `--show` 投影，输出参数的全部字段（含嵌套子字段和 tags）。

```bash
# 查看某个参数的全部信息（含完整标签）
python query.py -w name=--tensor-parallel-size -a --format yaml

# 导出某分类的所有字段供其他程序消费
python query.py -w category=parallelism -a --format yaml > parallelism_params.yaml
```

### `--sort-by` — 排序

```bash
# 按名称排序
python query.py -w category=memory --sort-by name

# 按性能影响等级排序（high 在前）
python query.py -w scope=vllm-ascend --sort-by performance_impact --sort-reverse

# 按分类排序
python query.py -w performance_impact=high --sort-by category
```

### `--count / -c` — 仅计数

```bash
# 快速统计
python query.py -t model=moe --count
# → 361

python query.py -t model=moe -t deploy_scenario=long_input --count
# → 73

# 对比各部署场景的 MoE 参数数量
for scene in long_input long_output high_concurrency; do
  echo -n "$scene: "
  python query.py -t model=moe -t deploy_scenario=$scene --count
done
```

## 实用案例

### 场景一：部署 DeepSeek-V3（MoE + MLA），优化长文本场景下的 TTFT

```bash
# 找到全部相关参数
python query.py -t model=moe,mla -t deploy_scenario=long_input -t optimize_target=ttft \
  --show name,type,tags.model,tags.deploy_scenario,tuning_advice.quick_guide
```

### 场景二：A3 硬件上多机部署，优化吞吐和显存

```bash
# 组合硬件 + 拓扑 + 优化目标标签
python query.py -t hardware=a3 -t deploy_topology=multi_node -t optimize_target=throughput,memory \
  --show name,performance_impact,tags.optimize_target,tuning_advice.quick_guide \
  --sort-by performance_impact
```

### 场景三：新模型上线前，了解所有编译优化选项

```bash
python query.py -w category=compilation -w scope=vllm-ascend \
  --show name,performance_impact,tuning_advice.quick_guide \
  --sort-by performance_impact
```

### 场景四：排查 OOM 问题

```bash
python query.py -s "OOM" --show name,impact_detail --format summary
python query.py -w category=memory --show name,default,tuning_advice.quick_guide
python query.py -t optimize_target=memory --show name,tags.model,tuning_advice.quick_guide
```

### 场景五：了解某参数的完整上下文

```bash
# 查看全部字段（包括 tags、constraints、suggested_values、caveats 等）
python query.py -w name=VLLM_ASCEND_ENABLE_NZ -a --format yaml

# 只看特定关注点
python query.py -w name=VLLM_ASCEND_ENABLE_NZ --show name,tags,impact_detail,constraints --format summary
```

### 场景六：高并发场景下所有需要关注的参数

```bash
python query.py -t deploy_scenario=high_concurrency --show name,type,category,tags.optimize_target,tuning_advice.summary
```

### 场景七：检查哪些参数已弃用

```bash
python query.py -w deprecated=true --show name,type,scope
```

### 场景八：对比 vLLM 上游和 Ascend 的高影响参数

```bash
echo "=== vllm-ascend ===" && python query.py -w scope=vllm-ascend -w performance_impact=high --count
echo "=== vllm ===" && python query.py -w scope=vllm -w performance_impact=high --count
```

### 场景九：导出带标签的完整数据集

```bash
# 导出全部带标签的参数 YAML
python query.py -a --format yaml > all_tagged_params.yaml
```

## 按自定义参数目录查询

```bash
# 指定其他参数目录（如未打标签的原始目录）
python query.py -d tag_params/output/params/ --count

# 指定自定义输出目录
python query.py -d /path/to/custom/output/params/ --count
```

## 可用字段一览

### 顶层字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | str | 参数名 |
| `type` | cli / env / nested | 参数类型 |
| `category` | enum | 功能分类 |
| `scope` | vllm / vllm-ascend | 来源范围 |
| `source_file` | list[str] | 定义文件路径 |
| `value_type` | enum | 参数值 Python 类型 |
| `default` | any | 默认值 |
| `valid_choices` | any | 合法取值 |
| `cli_example` | str | CLI 使用示例 |
| `deprecated` | bool | 是否已弃用 |
| `performance_impact` | high / medium / low / none | 性能影响等级 |
| `performance_scope` | list[enum] | 影响范围 |
| `impact_detail` | str | 性能影响机制 |
| `usage_locations` | list[dict] | 源码使用点 |
| `related_parameters` | list[dict] | 关联参数 |
| `constraints` | list[str] | 硬约束 |
| `analysis_date` | str | 分析日期 |
| `skip_reason` | str | 跳过原因（仅 none 参数） |

### 嵌套字段（`--show` 和 `-w` 可用点号访问）

| 字段 | 类型 | 说明 |
|------|------|------|
| `tuning_advice.summary` | str | 一句话调优总结 |
| `tuning_advice.suggested_values` | list[dict] | 场景推荐值 |
| `tuning_advice.caveats` | list[str] | 注意事项 |
| `tuning_advice.quick_guide` | str | 快速指引 |

### 标签字段（`--show` 和 `-t` 可用）

| 字段 | 类型 | 说明 |
|------|------|------|
| `tags.model` | list[str] | 模型架构：dense, moe, mla, vlm, quantized |
| `tags.optimize_target` | list[str] | 优化目标：ttft, tpot, throughput, memory |
| `tags.deploy_topology` | list[str] | 部署拓扑：single_node, multi_node |
| `tags.hardware` | list[str] | 硬件：a2, a3 |
| `tags.deploy_scenario` | list[str] | 部署场景：long_input, long_output, high_concurrency |

> 注意：`performance_impact: none` 的参数只有 `name`、`type`、`performance_impact`、`skip_reason` 四个字段，其他字段不存在。`tags` 字段仅在 tag_params 打标后存在。
