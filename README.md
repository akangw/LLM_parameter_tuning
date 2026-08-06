# Auto vLLM Parameter

> 第一次阅读只看 [业务链路总览](pipeline/README.md)：参数知识 → Agent 调参 → Benchmark。准备运行时再从 [场景目录](scenarios/README.md) 选择 W8A8/W4A8C8，并阅读 [可迁移快速启动](docs/PORTABLE_QUICKSTART.md)。

> GLM-5.2-W4A8C8 单节点 DP2-local2/TP8 已建立独立 planned 适配路线，现有调优脚本作为 A0，详见 [W4A8C8 A0 场景](docs/GLM52_W4A8C8_A0.md)。该路线使用独立本地 Runtime Root、远端项目、Lease、Session、缓存和实验产物，不改变 W8A8 默认链路。

面向 GLM-5.2、Atlas A3 和 vLLM Ascend 的参数知识构建与连续自动调优项目。项目从 `vllmTKB0706` 的在线闭环迁移而来，但使用独立源码版本、知识产物、Controller 状态、远端目录和 ktp-lab Lease。

## 当前状态

- 固定源码：vLLM `418bd6273c03bf48d5066733769e0a74bdc51694`，vllm-ascend `32c8cf190f596b47f0d0b965e64aea9f2b789ad4`。
- 参数知识：1540 个结构化表面，340 份完整 ParameterYAML，105 份带依据跳过记录。
- 五维标签：340 份，审计错误 0；当前场景召回 109 份高影响画像。
- 新 Session 默认复用人工审计注册表，当前编译为 15 Active、5 Reserve、5 Fixed、1 Rejected。独立自动注册表链路可显式切换，当前编译为 102 个可调维度（22 Active、80 Reserve、40 Fixed、0 Compiler Rejected）。在线权威结果只保存在每个 Session 的 `00_search_space/`。
- 在线闭环：已完成真实远端提交、服务启动、完整 Aligned-L1、结果回收、Agent 选参和 OOM 隔离。
- 当前正式锚点：A0，主分数 `602.5576 output tok/s`；尚未产生通过全部延迟门禁的新赢家。
- 唯一 B0 为 `B0-deployable`：模型原生 `max_model_len=1048576` 需要 `107.25 GiB` KV Cache，而当前拓扑仅有 `28.82 GiB`，因此只固定 `max_model_len=64000`，其余参数继续由目标版本源码解析。

## 可选扩展接口与 Ascend Runtime Adapter

### 更换 vLLM / vllm-ascend 版本

只提供两个 tag、分支或 commit，即可抓取源码并依次完成参数提取、Stage-1、画像、Tags、场景召回和 Search Limits：

```powershell
# 只完成确定性提取、Stage-1 和画像队列准备，不调用 Agent
.\scripts\migrate-versions.ps1 -Vllm <vllm-ref> -VllmAscend <ascend-ref> -PrepareOnly

# 默认用已登录的 Codex 完成全链路
.\scripts\migrate-versions.ps1 -Vllm <vllm-ref> -VllmAscend <ascend-ref>

# 路线一：复用现有画像作为迁移提示（默认）；缺少画像会拒绝启动
.\scripts\migrate-versions.ps1 -Vllm <vllm-ref> -VllmAscend <ascend-ref> `
  -PortraitMode migrate -LegacyPortraitDir .\portrait_pipeline\outputs\ParameterYAML

# 路线二：不读取旧画像，完全根据新源码重新画像
.\scripts\migrate-versions.ps1 -Vllm <vllm-ref> -VllmAscend <ascend-ref> `
  -PortraitMode rebuild

# 替换场景后会重新进行 Tags 召回并产生该场景的 Search Limits
.\scripts\migrate-versions.ps1 -Vllm <vllm-ref> -VllmAscend <ascend-ref> `
  -Scenario .\path\to\scenario.yaml

# 默认复用人工 23 项注册表；需要时可显式切换为独立自动注册表
.\scripts\migrate-versions.ps1 -Vllm <vllm-ref> -VllmAscend <ascend-ref> `
  -SearchSpaceProfile automatic_registry_v1

# 也可使用 Anthropic；Key 只放环境变量
$env:ANTHROPIC_API_KEY = "..."
.\scripts\migrate-versions.ps1 -Vllm <vllm-ref> -VllmAscend <ascend-ref> -Provider anthropic
```

画像路线通过 `-PortraitMode migrate|rebuild` 选择；场景通过 `-Scenario` 选择；召回到 Search Limits 的构建方式通过 `-SearchSpaceProfile automatic_registry_v1|curated_registry_v1` 选择。画像 Provider 当前支持 `codex`、`anthropic`。

Tags 召回到 Search Limits 之间另有一条独立的端到端自动化链路。它不读取或修改现有人工注册表，会自动完成语义归并、固定版本源码能力核验、场景兼容过滤、`unset/omit` 动作规范化、跨参数约束和通用注入校验，再调用现有 Search-Space Compiler。数值参数不再共用倍率边界：B0 前保留源码验证的临时候选，B0 成功后 Controller 从 `master.log` 读取实际生效值，并按参数专属策略重建候选域。这条兼容判定链只使用程序和 YAML 策略，不依赖 AI：

```powershell
cd .\tuning_pipeline
python -m workflow.registry_builder.full_pipeline --dry-run
```

它会输出自动 `registry.generated.yaml` 以及 Active / Reserve / Fixed / Rejected Search Limits。`automatic_registry_v1` 已完整接入 Controller，作为可插拔替代选项；生成的注册表、兼容策略、注入契约和 Search Limits 会冻结到 Session。默认 `curated_registry_v1` 使用人工审计的 23 项注册表。独立命令仍只写指定审计目录、不提交服务器任务。完整使用方式见 [`registry_builder/README.md`](tuning_pipeline/workflow/registry_builder/README.md)。

### 更换 Ascend 模型、镜像、量化或拓扑

模型、镜像和拓扑不再作为互相独立的零散字段迁移。`Runtime Adapter` 将模型家族/变体/权重格式、镜像 Digest 与源码 commit、Topology Profile、Executor Profile、Scenario、B0、Search-Space、Benchmark 和策略组合成一个可校验身份。当前默认适配包 `glm52_w8a8_a3_dp2_tp16` 完整保留现有 GLM-5.2 W8A8、A3、两节点 × 16 NPU、DP2/TP16 主流程。

新组合先生成 `planned` 适配包：

```powershell
.\scripts\new-runtime-adapter.ps1 scaffold `
  --name glm52-bf16-a3-dp4-tp8 `
  --model-family glm --model-variant glm-5.2 --weight-format bf16 `
  --image-manifest <manifest.yaml> --activation <activation.yaml> `
  --scenario <scenario.yaml> --baseline <b0.yaml> `
  --nodes 4 --npu-per-node 8 --data-parallel-size 4 --tensor-parallel-size 8 `
  --worker-replicas 3 --executor ktp_multi_role `
  --output <runtime-adapter.yaml>

.\scripts\new-runtime-adapter.ps1 validate <runtime-adapter.yaml>
```

只有执行器、B0、Benchmark 和 Search-Space 四项真实验证完成，且镜像批准、Scenario 的 Digest/commit、DP×TP 与总 NPU、worker rank 契约全部一致，适配包才能成为 `integrated`。`planned` 包不能提交任务。新拓扑若超出现有 `ktp_two_role` 能力，需要增加对应 Executor Profile 和 rank/Lease 实现；上层 Session、Agent、Benchmark、失败恢复和验收状态机无需重写。完整流程见 [Ascend Runtime Adapter](docs/ASCEND_RUNTIME_ADAPTERS.md)。

### 切换 Search-Space、Agent 策略与 Benchmark

在线闭环默认使用 `curated_registry_v1 + codex + best_anchor_coverage_v2 + aligned_l1_v4`。新建 Session 时可显式选择 Search-Space、策略、Provider 或 Benchmark：

```powershell
.\一键启动.ps1 -NewSession -StrategyProfile best_anchor_coverage_v3
.\一键启动.ps1 -NewSession -AgentProvider anthropic
.\一键启动.ps1 -NewSession -BenchmarkProfile vllm_bench_public_v1
.\一键启动.ps1 -NewSession -SearchSpaceProfile automatic_registry_v1
```

四个接口及定义位置：

| 接口 | 启动参数 | 当前选项 | 定义位置 |
|---|---|---|---|
| 参数画像路线 | `-PortraitMode` | `migrate`、`rebuild` | `scripts/migrate_versions.py` |
| Search-Space 构建 | `-SearchSpaceProfile` | `curated_registry_v1`（默认）、`automatic_registry_v1` | `tuning_pipeline/workflow/search_space_profiles.yaml` |
| Agent 选参策略 | `-StrategyProfile` | `best_anchor_coverage_v2`、`best_anchor_coverage_v3` | `tuning_pipeline/workflow/continuous/strategy_profiles.yaml` |
| Benchmark | `-BenchmarkProfile` | `aligned_l1_v4`、`vllm_bench_public_v1`、`custom_adapter_v1` | `tuning_pipeline/workflow/continuous/benchmark_profiles.yaml` |

Agent Provider 另通过 `-AgentProvider` 选择，支持 `codex`、`anthropic`、`openai_compatible`、`deepseek` 和 `command`。API Key 只通过环境变量配置。以上 Profile 仅允许在新建 Session 时选择；续跑使用该 Session 已保存的配置。

Benchmark 与 Agent Provider、选参策略相互独立。没有 ServeBench/GuideLLM 权限时，推荐选择 `vllm_bench_public_v1`：它只调用服务镜像内已有的公开 `vllm bench serve`，参数在 `config.yaml` 的 `benchmark.vllm_bench_public` 中配置。需要接入自有测试时，选择 `custom_adapter_v1`，并将 `benchmark.custom_benchmark.adapter_path` 指向 `workflow/benchmark_adapters/` 白名单目录内的 Python 适配器；接口与示例见 [`benchmark_adapters/README.md`](tuning_pipeline/workflow/benchmark_adapters/README.md)。不同 Profile/配置会生成不同的 Benchmark 身份摘要，Controller 不会跨口径比较历史结果。

Runtime Adapter、四个接口、失败重试、规则兜底和 V2/V3 差异统一见 [框架总览的“核心控制策略”](框架.md#核心控制策略)。

### Git 克隆后可直接复用的知识产物

仓库会跟踪并随 Git 一起分发当前正式的参数画像、跳过原因与 Tags；接手者无需先重新调用 Agent 就能阅读知识。人工或自动 Search Limits 都在新 Session 创建时重新编译，并只把该 Session 的冻结结果作为在线权威。源码 checkout、迁移运行目录、Session、临时队列和日志属于运行产物，不提交 Git。当前正式知识产物及其入口见 [产物与日志目录](docs/ARTIFACTS.md)。

## 从 GitHub 克隆后的启动流程

### 1. 启动前配置

```powershell
git clone https://github.com/chenasir/Auto_vllm_parameter.git
cd Auto_vllm_parameter

# 建议使用独立 Python 3.11+ 虚拟环境，避免污染系统/Anaconda 环境
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安装运行依赖
python -m pip install --upgrade pip
python -m pip install -r .\tuning_pipeline\requirements-runtime.txt

# 恢复不进入本仓库的固定版本源码
.\scripts\fetch-sources.ps1
```

接着确认：

1. `C:\Users\<用户名>\.ssh\config` 中存在 SSH 别名：

   ```sshconfig
   Host hetao-npu
       HostName 10.1.30.201
       Port 31222
       User demo1
   ```

2. `ssh hetao-npu` 可以连接。
3. 默认模式下 `codex --version` 可用且 Codex 已登录；若选择 API Provider，则对应 Key 环境变量已设置。
4. `tuning_pipeline/workflow/continuous/config.yaml` 中的远端项目、模型、MTP、Benchmark、Runtime Adapter 和 Lease 配置仍适用于目标服务器。
5. `activation.approved.yaml` 中的镜像、Digest、vLLM 和 vllm-ascend 身份与服务器一致；Controller 会逐项与 `remote/image_version_manifest.yaml` 动态核对，换版本不需要改校验代码。
6. 只读 `liuxin-workspace` 依赖仍可访问。

接手者需要修改的服务器项集中在 Git 忽略的 `config.local.yaml`：`remote_host`、`remote_project`、`deployment.*`（主模型、served-model、量化、网卡和环境脚本）、`lab.*` 以及所选 Benchmark 的服务端路径。相同运行组合可继续使用默认 Runtime Adapter；更换模型、量化、镜像或拓扑时，用 `-RuntimeAdapter` 选择新的 integrated 适配包。当前执行器明确支持两节点、每节点 16 NPU 的既定拓扑；其他拓扑会在 Executor Profile 校验阶段失败关闭，不能只改一个数字后静默运行。

最后执行不提交任务的端到端只读预检；它会检查本地配置、AI Provider、SSH 连接和 Lease 空闲状态：

```powershell
.\一键启动.ps1 -CheckOnly -NewSession
```

### 2. Lease 是否需要重新创建

先查看现有 Lease：

```powershell
ssh hetao-npu "cd /mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-418bd627-32c8cf190 && ktp-lab status --lease vllmtkb-418bd627-32c8cf190-glm52-a3-32npu"
```

按结果处理：

| Lease 状态 | 操作 |
|---|---|
| `active`、`nodes=2/2 Ready`、`idle=2` | 直接复用，不要重新创建 |
| Lease 不存在 | 运行 `.\scripts\prepare-remote.ps1` 创建一次 |
| Service slot 为 `running` 或不是 `idle=2` | 不启动重叠实验，先确认当前操作者和任务 |
| 服务镜像、Digest、节点资源或 Lease 模板发生变化 | 修改为新的版本化 Lease 名称，再创建新 Lease；不要用旧名称冒充新环境 |

`prepare-remote.ps1` 会同步远端受管脚本并提交 Lease 创建，因此只在 Lease 不存在或明确升级 Lease 身份时使用。同步时，Controller 会依据 `config.yaml` 的 `remote_project`、`lab.lease_name` 和已验证镜像身份动态生成远端 Lease/Experiment 控制 YAML；接手者无需再搜索并替换模板内的旧服务器路径。

### 3. 新建 Session 实验

适用情况：

- 只从 GitHub 克隆，没有收到历史 `state.json` 和 Session 目录；
- 希望用当前全局配置和最新 Search Limits 开始一条新实验链；
- 旧 Session 已结束或已经完成交接归档。

确认 Lease 为 `idle=2` 后运行：

```powershell
# 后台运行
.\一键启动.ps1 -NewSession

# 或前台运行，便于直接观察日志
.\一键启动.ps1 -NewSession -Foreground

# 可选：为这个新 Session 冻结另一套 Agent 选参策略
.\一键启动.ps1 -NewSession -StrategyProfile best_anchor_coverage_v3

# 可选：同时选择 Agent、策略与 Benchmark
.\一键启动.ps1 -NewSession -AgentProvider codex `
  -StrategyProfile best_anchor_coverage_v2 -BenchmarkProfile aligned_l1_v4 `
  -SearchSpaceProfile curated_registry_v1
```

当前配置的新 Session 会从 `round_000_b0_deployable` 开始：它保留官方源码默认启动，仅显式设置 `--max-model-len 64000` 以满足固定拓扑的 KV Cache 约束。成功完成并从日志回填实际默认值后，Controller 自动进入 Agent 选参和后续实验闭环。参数画像的 `migrate|rebuild` 属于离线知识构建，应在启动在线 Session 前通过 `scripts/migrate-versions.ps1` 单独选择和审计。

如果本地已经存在旧 Session，先用相同选择执行新 Session 预检，避免普通
`-CheckOnly` 自动检查旧 Session：

```powershell
.\一键启动.ps1 -CheckOnly -NewSession -AgentProvider codex `
  -StrategyProfile best_anchor_coverage_v2 -BenchmarkProfile aligned_l1_v4 `
  -SearchSpaceProfile curated_registry_v1
```

新建时会生成：

```text
tuning_pipeline/workflow/continuous/state.json
tuning_pipeline/workflow/continuous/experiments/<new-session>/
```

如果本地已有旧 `state.json`，`-NewSession` 会把状态入口切换到新 Session，但不会删除旧的 `experiments/<old-session>/`。操作前应确认旧 Session 不再需要续跑，并单独保存其 `state.json`。

### 4. 恢复续跑 Session 实验

GitHub 仓库默认不包含运行状态。要在另一台电脑续跑，必须额外交付并放回：

```text
tuning_pipeline/workflow/continuous/state.json
tuning_pipeline/workflow/continuous/experiments/<session>/
```

如果新电脑的仓库绝对路径不同，还要把 `state.json` 中的 `session_dir` 改为新电脑上该 Session 的实际绝对路径。然后依次执行：

```powershell
# 先确认本地状态、Session 文件、依赖和远端 Lease
.\一键启动.ps1 -CheckOnly
.\scripts\status.ps1

# Lease 必须存在，且没有另一台电脑正在控制同一个 Session
.\一键启动.ps1 -Resume

# 或前台恢复
.\一键启动.ps1 -Resume -Foreground
```

`-Resume` 始终读取 Session 内冻结的 `session_config.yaml` 和 `00_search_space/`，不会用新的全局默认值静默改变旧实验。如果状态是 `paused_for_human`，先查看 `failure.yaml` 和 Controller 日志；只有完成外部修复并确认允许同候选重试后，才使用：

```powershell
.\一键启动.ps1 -RetryPausedCurrent
```

### 5. 日常操作

```powershell
# 只检查，不提交任务
.\一键启动.ps1 -CheckOnly

# 自动判断：有可恢复状态则 Resume，否则新建
.\一键启动.ps1

# 查看状态
.\scripts\status.ps1

# 优雅停止：归档当前轮次，不提交下一轮
.\scripts\stop.ps1
```

## 项目层级

```text
Auto_vllm_parameter/
├─ pipeline/                     业务阅读入口（先读这里）
│  ├─ 01_parameter_knowledge/    画像 → 迁移 → Tags → 召回 → Search Limits
│  ├─ 02_agent_tuning/           基线 → Agent 候选 → Controller 校验
│  └─ 03_benchmark/              测量 → 比较 → 结果回填
├─ scenarios/                    选择模型、量化、镜像、拓扑和基线
├─ scripts/                      初始化、预检、启动、恢复、状态和停止
├─ docs/                         操作手册和详细设计
├─ portrait_pipeline/            参数知识内部实现（普通使用者跳过）
├─ tuning_pipeline/              调优与 Benchmark 实现
│  └─ workflow/continuous/scenario_runs/
│                               按场景隔离的状态、Session 和日志（不进入 Git）
├─ docker/                       Linux/Docker Controller 封装
└─ .runtime/                     其他本地临时运行文件（不进入 Git）
```

阅读项目使用 `pipeline → scenarios → scripts`；只有开发框架时才进入两个内部实现目录。

## 经批准保留的外部依赖

Benchmark 运行阶段仍以只读方式挂载：

```text
/mnt/host-model/slai/user-1-wangakang/wangakang/liuxin-workspace
```

该目录提供 ktp-lab Lease 控制文件和 GuideLLM 激活脚本。项目不会修改它；依赖范围、替换条件和风险见 [依赖说明](docs/DEPENDENCIES.md)。除该已声明依赖外，新项目不读取或修改 `vllmTKB0706` 的代码、状态、Lease 或实验目录。

## 文档

- [框架总览](框架.md)：交接时优先阅读，讲清画像、在线闭环和产物。
- [架构与数据流](docs/ARCHITECTURE.md)
- [运行与恢复](docs/OPERATIONS.md)
- [产物与日志索引](docs/ARTIFACTS.md)
- [交接清单](docs/HANDOFF.md)
- [外部依赖](docs/DEPENDENCIES.md)
- [当前实验摘要](docs/CURRENT_SESSION.md)
- [场景选择、迁移与统一启动](scenarios/README.md)
- [项目目录职责与产物边界](docs/PROJECT_STRUCTURE.md)
- [可迁移快速启动](docs/PORTABLE_QUICKSTART.md)
- [Linux / Docker Controller](docs/LINUX_DOCKER_CONTROLLER.md)
- [Ascend 模型、镜像与拓扑适配包](docs/ASCEND_RUNTIME_ADAPTERS.md)
- [服务器自治 systemd / Supervisor 服务](tuning_pipeline/workflow/continuous/server_autonomous/README.md)

## 安全边界

- 本地保存知识、决策、状态和实验归档；远端只执行服务与 Benchmark。
- 远端项目目录固定为 `/mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-418bd627-32c8cf190`。
- 服务镜像和 Benchmark 容器均使用 Digest 固定身份。
- Runtime Adapter、Scenario、B0、镜像、Topology 和 Executor 文件均记录 SHA-256；同一 Session 禁止跨适配身份续跑。
- `planned` Runtime Adapter 和未集成 Executor 一律失败关闭，不能提交远端任务。
- 上游 `--enable-eplb` 在当前 Ascend 版本中禁止进入搜索；Native Dynamic EPLB 接线完成前保持 `false/0`。
- 失败或残缺结果不会进入性能比较。
