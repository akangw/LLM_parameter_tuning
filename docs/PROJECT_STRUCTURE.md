# 项目目录与产物边界

这份文档只回答三个问题：从哪里开始、每个目录负责什么、产物最终写到哪里。

## 一眼看懂的四层结构

```text
Auto_vllm_parameter/
├─ scenarios/             ① 用户层：选择一个模型/量化/拓扑场景
├─ portrait_pipeline/     ② 共享知识层：从固定源码生成参数画像
├─ tuning_pipeline/       ③ 公共引擎层：Tags、Search Limits、Controller、Benchmark
├─ scripts/               ④ 操作入口：初始化、预检、启动、状态、停止
├─ docs/                    说明文档
├─ docker/                  Linux/Docker Controller 封装
└─ .runtime/               本机运行状态和 Session，不进入 Git
```

正常使用者只需要进入 `scenarios/`，不应该先在 `tuning_pipeline/workflow/continuous/`
里寻找配置。

## 每个目录的职责

| 目录 | 层级 | 内容 | 谁会修改 |
|---|---|---|---|
| `scenarios/` | 用户入口 | 场景身份、个人配置模板、固定产物索引 | 新增模型/拓扑时维护 |
| `portrait_pipeline/sources/` | 源码输入 | 固定 commit 的 vLLM/vllm-ascend | 版本迁移脚本生成，不入 Git |
| `portrait_pipeline/outputs/ParameterYAML/` | 共享知识 | 340 份源码级参数画像 | 画像流水线生成 |
| `tuning_pipeline/tag_params/output/params/` | 共享知识 | 340 份五维 Tags | Tag 流水线生成 |
| `tuning_pipeline/workflow/search_space_compiler/` | 场景编译 | Scenario、策略、Registry → Search Limits | 编译器维护 |
| `tuning_pipeline/workflow/baselines/` | 场景输入 | B0/A0 定义 | 每个场景验证后维护 |
| `tuning_pipeline/workflow/continuous/` | 公共引擎 | Controller、Profile 注册表、远端执行器 | 框架开发者维护 |
| `tuning_pipeline/workflow/benchmark_adapters/` | 扩展接口 | 自定义 Benchmark result-v1 适配器 | Benchmark 接入者维护 |
| `.runtime/scenarios/<id>/` | 本机运行产物 | state、Session、日志、PID | Controller 自动生成，不入 Git |
| `<remote_project>/workflow/auto/lab_runs/` | 服务器产物 | 完整服务和 Benchmark 原始证据 | 远端执行器自动生成 |

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

## 两个场景的固定入口

| 场景 | 总配置 | 基线 | 当前状态 |
|---|---|---|---|
| W8A8 2×16 NPU DP2/TP16 | `scenarios/glm52-w8a8-a3-2n-dp2-tp16/scenario.yaml` | `b0_deployable_64k.yaml` | integrated |
| W4A8C8 1×16 NPU DP2-local/TP8 | `scenarios/glm52-w4a8c8-a3-1n-dp2-tp8/scenario.yaml` | `a0_glm52_w4a8c8_existing_tuned.yaml` | planned |

`scenario.yaml` 是唯一需要先看的索引，不重复保存所有底层文件。执行下面的命令可以
解析并列出其引用的每个真实文件：

```powershell
.\scripts\scenario.ps1 -Action artifacts -Name glm52-w8a8-a3-2n-dp2-tp16
.\scripts\scenario.ps1 -Action artifacts -Name glm52-w4a8c8-a3-1n-dp2-tp8
```

## 一个 Session 的产物

使用 `scripts/scenario.ps1` 启动后：

```text
.runtime/scenarios/<scenario-id>/
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
