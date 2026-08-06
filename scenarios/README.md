# 场景目录：选择、迁移与启动

`scenarios/` 是面向使用者的第一入口。每个目录代表一个独立实验身份，W8A8 与
W4A8C8 在同一层级展示；Controller、执行器和注册表仍集中放在 `tuning_pipeline/`
中复用，不复制算法代码。

```text
scenarios/
├─ glm52-w8a8-a3-2n-dp2-tp16/    # integrated，可运行
│  ├─ scenario.yaml               # 模型/拓扑/B0/Benchmark/产物索引
│  ├─ operator.example.yaml       # 别人只需要填写的机器配置
│  ├─ ARTIFACTS.md                # 固定定义与运行产物位置
│  └─ README.md
└─ glm52-w4a8c8-a3-1n-dp2-tp8/   # planned，独立验证中
   ├─ scenario.yaml
   ├─ operator.example.yaml
   ├─ ARTIFACTS.md
   └─ README.md
```

运行时文件不进入源码树中的上述目录，而统一进入同级身份的忽略目录：

```text
.runtime/scenarios/<scenario-id>/
├─ state.json
├─ experiments/<session-id>/
├─ logs/
└─ process/
```

因此两个场景不会共享 Session、B0 回填值、候选历史、最佳 Anchor、失败规则或指标。

## 别人拿到项目后必须提供什么

| 类别 | 必填内容 | 为什么需要 |
|---|---|---|
| 调度入口 | SSH alias、`ktp-lab`、用户自己的远端可写目录 | 同步脚本、创建 Lease、收集结果 |
| 模型 | 模型路径、served-model-name；启用 MTP 时提供切片模型路径 | 生成实际 vLLM 命令 |
| Ascend 环境 | CANN/初始化脚本、网卡、服务端口 | 多节点通信与服务启动 |
| 镜像身份 | 镜像 reference/digest、vLLM commit、vllm-ascend commit | 防止旧画像和新运行时误配 |
| 资源身份 | 唯一 Lease 名、节点数、每节点 NPU、DP/TP | 防止场景争抢和拓扑漂移 |
| Agent | Codex 登录，或 API Key 环境变量 | 只负责结构化选参 |
| Benchmark | 内部 Aligned-L1、公开 vLLM bench 或自定义适配器 | 冻结本 Session 的评价标准 |

API Key 永远只放环境变量或服务器 `.secrets`，不写入 `scenario.yaml`、个人 YAML、
Session 或 Git。

## 五步启动

```powershell
# 1. 查看两个同级场景及状态
.\scripts\scenario.ps1 -Action list

# 查看一个场景引用的全部固定文件、共享画像数量和已有 Session
.\scripts\scenario.ps1 -Action artifacts -Name glm52-w8a8-a3-2n-dp2-tp16

# 2. 生成 Git 忽略的个人配置
.\scripts\scenario.ps1 -Action init -Name glm52-w8a8-a3-2n-dp2-tp16
# 编辑 scenarios/glm52-w8a8-a3-2n-dp2-tp16/operator.local.yaml

# 3. 只读预检，不提交任务
.\scripts\scenario.ps1 -Action check -Name glm52-w8a8-a3-2n-dp2-tp16

# 4. 首次创建/准备该场景自己的 Lease
.\scripts\scenario.ps1 -Action prepare -Name glm52-w8a8-a3-2n-dp2-tp16

# 5. 新 Session 从自己的 B0/A0 开始，之后自动进入 Agent 闭环
.\scripts\scenario.ps1 -Action start -Name glm52-w8a8-a3-2n-dp2-tp16
```

日常操作：

```powershell
.\scripts\scenario.ps1 -Action status -Name glm52-w8a8-a3-2n-dp2-tp16
.\scripts\scenario.ps1 -Action resume -Name glm52-w8a8-a3-2n-dp2-tp16
.\scripts\scenario.ps1 -Action stop -Name glm52-w8a8-a3-2n-dp2-tp16
```

## 什么可以直接换，什么必须新建场景

| 变化 | 操作 | 是否需要新 Session | 是否需要新场景包 |
|---|---|---:|---:|
| SSH alias、远端可写目录、端口 | 修改 `operator.local.yaml` | 是 | 否 |
| Codex 换 DeepSeek/Claude/OpenAI-compatible | 改 Provider 并设置 Key | 是 | 否 |
| 已集成 Benchmark Profile | 启动时传 `-BenchmarkProfile` | 是 | 通常否 |
| 自定义 Benchmark | 实现 result-v1，选择 `custom_adapter_v1` | 是 | 外部 Adapter 场景需派生包 |
| 人工/自动 Search Limits | 选择 Search-Space Profile | 是 | 外部 Adapter 场景需派生包 |
| V2/V3 Agent 策略 | 选择 Strategy Profile | 是 | 外部 Adapter 场景需派生包 |
| 同模型路径但镜像 digest/commit 改变 | 重新身份核验和画像迁移 | 是 | 是 |
| W8A8 换 W4A8C8/其他量化 | 新 B0/A0、Scenario、Search Limits | 是 | 是 |
| 节点数、NPU、DP/TP 改变 | 新 Topology + Executor 验证 | 是 | 是 |

已经创建的 Session 不能中途切换上述身份。`session_config.yaml` 会冻结最终适配包、
Search Limits、策略、Benchmark 和 Agent Provider。

### 换 Agent

例如 DeepSeek：

```powershell
$env:DEEPSEEK_API_KEY = "..."
.\scripts\scenario.ps1 -Action check -Name glm52-w8a8-a3-2n-dp2-tp16 -AgentProvider deepseek
```

Provider 只改变结构化决策模型；证据包、Search Limits、策略和确定性校验不变。

### 换 Benchmark

没有内部 ServeBench 权限时：

```powershell
.\scripts\scenario.ps1 -Action check -Name glm52-w8a8-a3-2n-dp2-tp16 `
  -BenchmarkProfile vllm_bench_public_v1
.\scripts\scenario.ps1 -Action prepare -Name glm52-w8a8-a3-2n-dp2-tp16 `
  -BenchmarkProfile vllm_bench_public_v1
.\scripts\scenario.ps1 -Action start -Name glm52-w8a8-a3-2n-dp2-tp16 `
  -BenchmarkProfile vllm_bench_public_v1
```

接入自己的 Benchmark 时，复制
`tuning_pipeline/workflow/benchmark_adapters/example_http_adapter.py`，实现
`benchmark_result.schema.json` 的 result-v1 指标，然后在个人配置里补充
`benchmark.custom_benchmark` 并选择 `custom_adapter_v1`。同一 Session 绝不能把不同
Benchmark 的分数横向比较。

### 换模型、镜像或拓扑

复制一个现有场景目录作为新同级目录，先保持 `status: planned`，再用
`scripts/new-runtime-adapter.ps1` 生成适配包。只有以下四项全部完成才能改为
`integrated`：

1. Executor：节点角色、DP rank、退出、重试和日志回收真实验证；
2. B0/A0：在目标模型、镜像和拓扑上实际部署并测量；
3. Benchmark：served model、tokenizer、数据集和指标门禁对齐；
4. Search Limits：按新场景和镜像重新编译，不能继承旧模型历史最优值。

代码层的 Controller、Agent Provider、比较器和 Session 状态机无需复制。

## W8A8 与 W4A8C8 当前状态

- `glm52-w8a8-a3-2n-dp2-tp16`：`integrated`，正式默认路线。
- `glm52-w4a8c8-a3-1n-dp2-tp8`：`planned`，已有独立 A0、拓扑、执行器、
  Benchmark 和 Search-Space 定义，但四项真实 attestation 尚未全部完成，因此入口会
  拒绝 `prepare/start/resume`，不会误跑。
