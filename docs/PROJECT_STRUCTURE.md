# 项目目录与产物边界

这份文档只回答三个问题：从哪里开始、每个目录负责什么、产物最终写到哪里。

## 一眼看懂：三条业务链 + 两个操作入口

```text
LLM_parameter_tuning/
├─ pipeline/
│  ├─ 01_parameter_knowledge/  画像、迁移、Tags、召回、Search Limits
│  ├─ 02_agent_tuning/         Agent 选参和 Controller 校验
│  └─ 03_benchmark/            测量、比较和结果回填
├─ scenarios/                  选择运行身份
├─ scripts/                    执行操作
├─ portrait_pipeline/          参数知识内部实现
└─ tuning_pipeline/            调优与 Benchmark 内部实现
   └─ workflow/continuous/scenario_runs/
                              按场景隔离的 Session 产物，不进入 Git
```

理解项目先进入 `pipeline/`；运行项目再进入 `scenarios/`。不应该从
`tuning_pipeline/workflow/continuous/` 开始阅读。

## 每个目录的职责

| 目录 | 层级 | 内容 | 谁会修改 |
|---|---|---|---|
| `pipeline/` | 业务导航 | 参数知识、Agent 调参、Benchmark 三条一级链路 | 业务边界变化时维护 |
| `scenarios/` | 用户入口 | 场景身份、个人配置模板、固定产物索引 | 新增模型/拓扑时维护 |
| `portrait_pipeline/sources/` | 源码输入 | 固定 commit 的 vLLM/vllm-ascend | 版本迁移脚本生成，不入 Git |
| `portrait_pipeline/outputs/ParameterYAML/` | 共享知识 | 340 份源码级参数画像 | 画像流水线生成 |
| `tuning_pipeline/tag_params/output/params/` | 共享知识 | 340 份五维 Tags | Tag 流水线生成 |
| `tuning_pipeline/workflow/search_space_compiler/` | 场景编译 | Scenario、策略、Registry → Search Limits | 编译器维护 |
| `tuning_pipeline/workflow/baselines/` | 场景输入 | B0/A0 定义 | 每个场景验证后维护 |
| `tuning_pipeline/workflow/continuous/` | 公共引擎 | Controller、Profile 注册表、远端执行器 | 框架开发者维护 |
| `tuning_pipeline/workflow/continuous/model_loading_profiles.yaml` | 加载适配 | DTFS 页缓存与 RFork seed/client Profile | 模型加载路线变化时维护 |
| `tuning_pipeline/workflow/benchmark_adapters/` | 扩展接口 | 自定义 Benchmark result-v1 适配器 | Benchmark 接入者维护 |
| `tuning_pipeline/workflow/continuous/scenario_runs/<id>/` | 本机运行产物 | state、Session、日志、PID | Controller 自动生成，不入 Git |
| `<remote_project>/workflow/auto/lab_runs/` | 服务器产物 | 完整服务和 Benchmark 原始证据 | 远端执行器自动生成 |

`pipeline/` 不保存第二份配置或产物，只解释业务步骤并指向下表中的唯一实现位置，避免
“为了易读而复制代码”造成两套事实来源。

## 为什么不是每个场景复制 340 份画像

ParameterYAML 描述的是固定 vLLM/vllm-ascend 源码中的参数语义，因此相同源码身份下
W8A8 与 W4A8C8 共享这套知识。场景差异由以下文件施加：

```text
共享 ParameterYAML + Tags
        ↓
场景自己的 Search-Space Scenario、镜像能力、模型/拓扑约束
        ↓
场景自己的 Compiler 输出和 Search Limits
        ↓
Session/00_search_space/ 中冻结的场景参数画像子集
```

如果镜像中的 vLLM/vllm-ascend commit 变化，必须重建或迁移共享画像；如果只是模型、
量化或拓扑变化，则使用同一源码画像重新做场景召回和编译，不能继承旧场景的历史最优值。

## 场景与生产入口

| 场景 | 总配置 | 基线 | 当前状态 |
|---|---|---|---|
| W8A8 2×16 NPU DP4/TP8 Decode-only | `server_autonomous/config.dp4_tp8.decode_priority_v3.yaml` | `expert_decode_glm52_w8a8_dp4_tp8_a10f1_v3.yaml` | production |
| W8A8 2×16 NPU DP2/TP16 | `scenarios/glm52-w8a8-a3-2n-dp2-tp16/scenario.yaml` | `b0_deployable_64k.yaml` | integrated historical |
| W4A8C8 1×16 NPU DP2-local/TP8 | `scenarios/glm52-w4a8c8-a3-1n-dp2-tp8/scenario.yaml` | `a0_glm52_w4a8c8_existing_tuned.yaml` | planned |

通用场景使用 `scenario.yaml` 作为索引；当前生产 Decode V3 由专用 dispatcher 和
继承配置固定，不能用通用场景脚本恢复。通用场景可用下列命令列出引用文件：

```powershell
.\scripts\scenario.ps1 -Action artifacts -Name glm52-w8a8-a3-2n-dp2-tp16
.\scripts\scenario.ps1 -Action artifacts -Name glm52-w4a8c8-a3-1n-dp2-tp8
```

## 一个 Session 的产物

使用 `scripts/scenario.ps1` 启动后：

```text
tuning_pipeline/workflow/continuous/scenario_runs/<scenario-id>/
├─ state.json                         当前 Controller 状态
├─ logs/controller/                   Controller 和后台进程日志
└─ experiments/<session-id>/
   ├─ session_config.yaml             全部冻结配置
   ├─ image_version_manifest.yaml     镜像/源码身份
   ├─ 00_search_space/
   │  ├─ search_space_profile.yaml    人工/自动 Profile
   │  ├─ search_space.compiled.yaml   编译后的 Search Limits
   │  ├─ classified_search_limits.yaml
   │  ├─ parameter_portraits.full.yaml
   │  ├─ parameter_portraits.agent.yaml
   │  ├─ registry.generated.yaml
   │  └─ runtime_rules.yaml
   └─ round_NNN_<label>/
      ├─ 00_context/                  场景、镜像、轮次
      ├─ 01_query/                    知识召回
      ├─ 02_parameters/               候选和真实启动命令
      ├─ 03_submission/               Lease/任务提交证据
      ├─ 04_runtime/                  服务与 Benchmark 日志
      ├─ 05_results/                  metrics/comparison/failure
      └─ 06_agent_analysis/           Agent 证据、决策、下一候选
```

旧版本在 `tuning_pipeline/workflow/continuous/experiments/` 下留下的本地 Session 是兼容
历史，不再作为新场景的推荐位置，也不会自动移动或删除。新场景统一写 `.runtime/`。

## 哪些是源码，哪些是产物

- Git 跟踪：场景索引、B0/A0、Scenario、Profile、代码和文档。
- Git 不跟踪：个人配置、API Key、state、Session、日志、PID、模型、缓存和远端原始结果。
- 服务器完整日志是运行证据权威；本地 Session 保存决策所需的核心镜像和指标。
