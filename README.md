# Auto vLLM Parameter

面向 GLM-5.2、Atlas A3 和 vLLM Ascend 的参数知识构建与连续自动调优项目。项目从 `vllmTKB0706` 的在线闭环迁移而来，但使用独立源码版本、知识产物、Controller 状态、远端目录和 ktp-lab Lease。

## 当前状态

- 固定源码：vLLM `418bd6273c03bf48d5066733769e0a74bdc51694`，vllm-ascend `32c8cf190f596b47f0d0b965e64aea9f2b789ad4`。
- 参数知识：1540 个结构化表面，340 份完整 ParameterYAML，105 份带依据跳过记录。
- 五维标签：340 份，审计错误 0；当前场景召回 109 份高影响画像。
- 新 Session 默认自动构建注册表：36 个最终可调参数，12 Active、24 Reserve、35 Fixed、0 Rejected；人工 23 项注册表作为可切换兼容路径保留。早期隔离审计的 44（12+32）已在接入审查中剔除 8 个已知无效/固定维度。
- 在线闭环：已完成真实远端提交、服务启动、完整 Aligned-L1、结果回收、Agent 选参和 OOM 隔离。
- 当前正式锚点：A0，主分数 `602.5576 output tok/s`；尚未产生通过全部延迟门禁的新赢家。
- B0 已定义为“目标版本源码/启动日志中的官方默认值”，当前状态为等待项目负责人下令提交，尚未运行。

## 四个可选扩展接口

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

# 可选择自动构建注册表（默认）或复用人工 23 项注册表
.\scripts\migrate-versions.ps1 -Vllm <vllm-ref> -VllmAscend <ascend-ref> `
  -SearchSpaceProfile automatic_registry_v1

# 也可使用 Anthropic；Key 只放环境变量
$env:ANTHROPIC_API_KEY = "..."
.\scripts\migrate-versions.ps1 -Vllm <vllm-ref> -VllmAscend <ascend-ref> -Provider anthropic
```

画像路线通过 `-PortraitMode migrate|rebuild` 选择；场景通过 `-Scenario` 选择；召回到 Search Limits 的构建方式通过 `-SearchSpaceProfile automatic_registry_v1|curated_registry_v1` 选择。画像 Provider 当前支持 `codex`、`anthropic`。

Tags 召回到 Search Limits 之间另有一条独立的端到端自动化链路。它不读取或修改现有人工注册表，会自动完成语义归并、固定版本源码能力核验、场景兼容过滤、`unset/omit` 动作规范化、跨参数约束和通用注入校验，再调用现有 Search-Space Compiler。这条兼容判定链只使用程序和 YAML 策略，不依赖 AI：

```powershell
cd .\tuning_pipeline
python -m workflow.registry_builder.full_pipeline --dry-run
```

它会输出自动 `registry.generated.yaml` 以及 Active / Reserve / Fixed / Rejected Search Limits。`automatic_registry_v1` 已接入 Controller 并作为新 Session 默认值；生成的注册表、兼容策略、注入契约和 Search Limits 会冻结到 Session。`curated_registry_v1` 保留人工 23 项注册表，便于历史对照。独立命令仍只写指定审计目录、不提交服务器任务。完整使用方式见 [`registry_builder/README.md`](tuning_pipeline/workflow/registry_builder/README.md)。

### 切换 Search-Space、Agent 策略与 Benchmark

在线闭环默认使用 `automatic_registry_v1 + codex + best_anchor_coverage_v2 + aligned_l1_v4`。新建 Session 时可显式选择 Search-Space、策略、Provider 或 Benchmark：

```powershell
.\一键启动.ps1 -NewSession -StrategyProfile best_anchor_coverage_v3
.\一键启动.ps1 -NewSession -AgentProvider anthropic
.\一键启动.ps1 -NewSession -BenchmarkProfile legacy_random_32k1k
.\一键启动.ps1 -NewSession -SearchSpaceProfile curated_registry_v1
```

四个接口及定义位置：

| 接口 | 启动参数 | 当前选项 | 定义位置 |
|---|---|---|---|
| 参数画像路线 | `-PortraitMode` | `migrate`、`rebuild` | `scripts/migrate_versions.py` |
| Search-Space 构建 | `-SearchSpaceProfile` | `automatic_registry_v1`、`curated_registry_v1` | `tuning_pipeline/workflow/search_space_profiles.yaml` |
| Agent 选参策略 | `-StrategyProfile` | `best_anchor_coverage_v2`、`best_anchor_coverage_v3` | `tuning_pipeline/workflow/continuous/strategy_profiles.yaml` |
| Benchmark | `-BenchmarkProfile` | `aligned_l1_v4`、`legacy_random_32k1k` | `tuning_pipeline/workflow/continuous/benchmark_profiles.yaml` |

Agent Provider 另通过 `-AgentProvider` 选择，支持 `codex`、`anthropic`、`openai_compatible` 和 `command`。API Key 只通过环境变量配置。以上 Profile 仅允许在新建 Session 时选择；续跑使用该 Session 已保存的配置。

四个接口的设计原理、失败重试、规则兜底和 V2/V3 差异统一见 [框架总览的“核心控制策略”](框架.md#核心控制策略)。

### Git 克隆后可直接复用的知识产物

仓库会跟踪并随 Git 一起分发当前正式的参数画像、跳过原因、Tags 与 Search Limits；接手者无需先重新调用 Agent 才能阅读或复用它们。源码 checkout、迁移运行目录、Session、临时队列和日志属于可再生产物，不提交 Git。当前正式知识产物及其入口见 [产物与日志目录](docs/ARTIFACTS.md)。

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
4. `tuning_pipeline/workflow/continuous/config.yaml` 中的远端项目、模型、MTP、Benchmark 和 Lease 配置仍适用于目标服务器。
5. `activation.approved.yaml` 中的镜像、Digest、vLLM 和 vllm-ascend 身份与服务器一致；Controller 会逐项与 `remote/image_version_manifest.yaml` 动态核对，换版本不需要改校验代码。
6. 只读 `liuxin-workspace` 依赖仍可访问。

接手者需要修改的服务器项集中在 `config.yaml`：`remote_host`、`remote_project`、`deployment.*`（主模型、served-model、量化、网卡和环境脚本）、`lab.*` 以及所选 Benchmark 的服务端路径；镜像与源码身份集中在 `remote/image_version_manifest.yaml` 和 `activation.approved.yaml`。当前执行器明确支持两节点、每节点 16 NPU 的既定拓扑；更换主机和路径是配置操作，更换拓扑则属于新的执行器适配，不能只改一个数字后静默运行。

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
  -SearchSpaceProfile automatic_registry_v1
```

当前配置的新 Session 会从 `round_000_b0` 官方默认参数基线开始；B0 成功完成并从日志回填实际默认值后，Controller 自动进入 Agent 选参和后续实验闭环。参数画像的 `migrate|rebuild` 属于离线知识构建，应在启动在线 Session 前通过 `scripts/migrate-versions.ps1` 单独选择和审计。

如果本地已经存在旧 Session，先用相同选择执行新 Session 预检，避免普通
`-CheckOnly` 自动检查旧 Session：

```powershell
.\一键启动.ps1 -CheckOnly -NewSession -AgentProvider codex `
  -StrategyProfile best_anchor_coverage_v2 -BenchmarkProfile aligned_l1_v4 `
  -SearchSpaceProfile automatic_registry_v1
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
├─ README.md                     项目总入口
├─ 框架.md                       三大板块的交接讲解
├─ docs/                         架构、运行、产物和交接文档
├─ scripts/                      面向操作者的稳定入口
├─ portrait_pipeline/            离线参数画像构建
│  ├─ build/                     提取、初筛、迁移和画像程序/证据
│  ├─ outputs/ParameterYAML/     340 份正式参数画像
│  ├─ outputs/skipped/           105 份跳过证据
│  └─ sources/                   固定提交源码，本地生成且不入库
└─ tuning_pipeline/              标签、搜索空间与在线调优
   ├─ tag_params/output/params/  340 份五维标签成品
   ├─ search_limits/             最新独立编译产物
   ├─ logs/                      本地总控日志入口
   └─ workflow/
      ├─ search_space_profiles.yaml  自动/人工注册表 Profile 入口
      ├─ registry_builder/       自动注册表、兼容校验与通用注入
      ├─ search_space_compiler/  Search Limits 编译器
      ├─ sidecars/               画像检索和运行规则
      └─ continuous/             Controller、远端脚本和 Session 运行时
```

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

## 安全边界

- 本地保存知识、决策、状态和实验归档；远端只执行服务与 Benchmark。
- 远端项目目录固定为 `/mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-418bd627-32c8cf190`。
- 服务镜像和 Benchmark 容器均使用 Digest 固定身份。
- 上游 `--enable-eplb` 在当前 Ascend 版本中禁止进入搜索；Native Dynamic EPLB 接线完成前保持 `false/0`。
- 失败或残缺结果不会进入性能比较。
