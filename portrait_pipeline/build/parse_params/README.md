# parse_params

vLLM / vllm-ascend 参数性能分析工具。性能调优知识库的第二阶段。

## 做什么

从 `parameters.json`（由 extract_params 生成的 vLLM/vllm-ascend 参数列表）中筛选性能相关参数，用 LLM 深入分析源码，为每个参数生成结构化的 YAML 知识文件。

```
extract_params（第一阶段）              parse_params（本阶段）
─────────────────────────             ─────────────────────
vllm/vllm-ascend 源码                  parameters.json (所有提取到的参数)
        │                                      │
        ▼                                      ▼
   parameters.json          ──►     output/params/*.yaml (性能相关参数的 YAML 知识文件)
  (结构化参数列表)                  每个参数一个 YAML，含性能影响分析、
                                   调优建议、硬约束、参数间关系等
```

## 安装

```bash
pip install anthropic pyyaml pydantic
```

设置 Anthropic API key。key 通过环境变量传入，代码不硬编码：

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

`LLMClient` 在 `stage2_analyzer.py:341` 初始化时调用 `Anthropic()` 无参构造，SDK 自动从 `ANTHROPIC_API_KEY` 环境变量读取。

## 运行

```bash
cd ./portrait_pipeline
python -m parse_params --input ./parameters.json --output parse_params/output/
```

### 常用选项

| 选项 | 说明 |
|------|------|
| `--input / -i` | parameters.json 路径（默认 `./parameters.json`） |
| `--output / -o` | 输出目录（默认 `./output/`） |
| `--resume / -r` | 从中断处续跑（读取 progress.json） |
| `--dry-run-stage1` | 仅运行 Stage 1 粗筛，打印通过/跳过统计 |
| `--max-params / -n N` | 限制 Stage 2 最多分析 N 个参数（测试用） |
| `--concurrency / -c N` | LLM API 并发数（默认 15） |
| `--verbose / -v` | 开启 debug 日志 |

### 示例

```bash
# 预览 Stage 1 筛选结果（不调 LLM，零成本）
python -m parse_params --dry-run-stage1

# 只跑 10 个参数做测试
python -m parse_params --max-params 10 --concurrency 5

# 完整运行
python -m parse_params --concurrency 15 -v

# 中断后续跑
python -m parse_params --resume
```

## 输入

`parameters.json` — JSON 数组，每个元素包含：

| 字段 | 说明 | 示例 |
|------|------|------|
| `name` | 参数名 | `"--tensor-parallel-size"` |
| `type` | `"cli"` / `"env"` / `"nested"` | |
| `category` | 功能分类 | `"parallelism"`, `"memory"`, `"compilation"` |
| `description` | 从代码中提取的描述 | |
| `source_file` | 源文件路径 | |
| `scope` | `"vllm"` / `"vllm-ascend"` | |

（由第一阶段 `extract_params` 模块生成）

## 输出

```
output/
├── params/              # 性能相关参数的 YAML 文件
│   ├── tensor_parallel_size.yaml
│   ├── cudagraph_mode.yaml
│   └── ...
├── logs/
│   ├── skipped_params.json     # 被跳过的参数及原因
│   └── analysis_errors.json    # 分析失败的参数
├── schema.yaml                 # 输出 schema（副本）
├── manifest.yaml               # 构建清单
└── progress.json               # 进度文件（支持断点续跑）
```

### YAML 文件结构

```yaml
name: --tensor-parallel-size
type: cli
category: parallelism
scope: vllm
source_file:
  - vllm/engine/arg_utils.py:825

value_type: int
default: 1
valid_choices: null

performance_impact: high           # high | medium | low
performance_scope:                 # latency | throughput | memory
  - latency
  - throughput
  - memory
impact_detail: >
  控制张量并行度，直接影响每张卡上的模型分片大小、
  通信量和 KV cache 分布。

usage_locations:
  - file: vllm_ascend/worker/worker.py:447
    context: 编译预热时检查 cudagraph_mode

related_parameters:
  - name: --pipeline-parallel-size
    relation: 与 TP 共同决定总并行度

constraints:                       # 硬约束，违反会崩溃/异常
  - tp_size 必须整除 attention heads 总数
  - tp_size × dp_size × pp_size 不能超过可用 NPU 数量

tuning_advice:
  summary: 根据模型大小选择最小的可行 TP 值以减少通信
  suggested_values:
    - scenario: 7B 模型，8×64GB NPU
      value: 1
      reason: 单卡可容纳完整模型
  caveats:                         # 软建议/性能陷阱
    - 增大 TP 会增加 all-reduce 通信量
  quick_guide: TP=1 性能最优；显存不足时优先增大 TP 而非 PP
```

## 流水线

```
parameters.json (所有提取到的参数)
        │
        ▼
┌──────────────────────────┐
│ Stage 1: 粗筛（本地规则）  │
│ - 按 category 黑/白名单    │
│ - 名称关键词匹配           │
│ - 描述文本正则             │
│ → 跳过明确无关参数（认证凭证/日志/编译工具链等）│
└──────────┬───────────────┘
           │ 通过的参数进入 Stage 2
           ▼
┌──────────────────────────┐
│ Stage 2: LLM 深度分析      │
│ a. 搜索双仓库源码          │
│ b. AST 提取函数/类作用域   │
│ c. 构建 prompt             │
│ d. LLM 判断性能影响+生成    │
│    YAML（1 次 API 调用）   │
│ e. Pydantic 校验           │
│ f. 写入 output/params/     │
└──────────┬───────────────┘
           │
           ▼
      Post: manifest.yaml + 统计
```

## 架构

```
parse_params/
├── resources/                  # 可独立修改的配置资源
│   ├── schema.yaml             # 输出 YAML schema
│   ├── system_prompt.txt       # LLM 系统提示词
│   ├── user_prompt_template.txt
│   └── stage1_rules.yaml       # Stage 1 过滤规则
├── config.py                   # 常量配置
├── schema.py                   # Pydantic 模型 + schema 文本生成
├── utils.py                    # AST 作用域/文件名安全化/YAML 清洗
├── stage1_filter.py            # Stage 1 粗筛
├── stage2_analyzer.py          # Stage 2 核心
│   ├── PromptBuilder           #   prompt 模板渲染
│   ├── ContextReader           #   双仓库源码搜索 + AST 提取
│   ├── LLMClient               #   Anthropic API 封装（重试/并发）
│   ├── YAMLWriter              #   校验 + 写入
│   └── Stage2Analyzer          #   编排器
├── progress.py                 # 断点续跑
├── manifest.py                 # manifest.yaml 生成
└── __main__.py                 # 入口
```

## 配置

核心配置在 `config.py`：

```python
LLM_MODEL = "claude-sonnet-4-6"    # 分析用模型
LLM_CONCURRENCY = 15               # 并发数
LLM_MAX_RETRIES = 2                 # API 重试次数
MAX_USAGE_LOCATIONS = 10            # 每个参数最多展示的使用点
MAX_USAGE_SCOPE_LINES = 40          # 每个使用点最多展示行数
```

## 成本

- Stage 2 分析的参数数量取决于 Stage 1 筛选结果
- 每个参数平均 ~2.5K input tokens（含源码上下文）+ ~500 output tokens
- Sonnet 4.6 定价 $3/$15 per 1M tokens
- 预估总成本与参数数量成正比，典型规模下约 $5-15
