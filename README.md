# Auto vLLM Parameter

> 第一次阅读只看 [业务链路总览](pipeline/README.md)：参数知识 → Agent 调参 → Benchmark。准备运行时再从 [场景目录](scenarios/README.md) 选择 W8A8/W4A8C8，并阅读 [可迁移快速启动](docs/PORTABLE_QUICKSTART.md)。

> GLM-5.2-W4A8C8 单节点 DP2-local2/TP8 已建立独立 planned 适配路线，现有调优脚本作为 A0，详见 [W4A8C8 A0 场景](docs/GLM52_W4A8C8_A0.md)。该路线使用独立本地 Runtime Root、远端项目、Lease、Session、缓存和实验产物，不改变 W8A8 默认链路。

面向 GLM-5.2、Atlas A3 和 vLLM Ascend 的参数知识构建与连续自动调优项目。项目从 `vllmTKB0706` 的在线闭环迁移而来，但使用独立源码版本、知识产物、Controller 状态、远端目录和 ktp-lab Lease。

## 当前状态

- vLLM 与 vllm-ascend 源码身份已固定，具体 commit 由版本清单和镜像身份文件校验。
- 参数画像、Tags、场景召回、人工注册表和自动注册表知识产物均已建立。
- 新 Session 会重新编译并冻结自己的 Search Limits；在线权威结果只保存在该 Session 的 `00_search_space/`。
- 在线闭环：已完成真实远端提交、服务启动、完整 Aligned-L1、结果回收、Agent 选参和 OOM 隔离。
- `B0-deployable` 是正式默认基线定义；必要的部署兼容覆盖由版本化基线文件记录，其余参数继续从目标源码解析。

README 不公开具体实验分数、吞吐、延迟、逐轮候选或当前 Session
状态；这些数据只保存在对应 Session 的受管产物中，并通过状态和导出命令按需查看。

## 第一次接手：能否直接运行

结论分为两层：在已集成的 `GLM-5.2 W8A8 + Atlas A3 + 2 节点 × 16 NPU + ktp-lab`
环境中，补齐私有配置并通过预检后，可以运行完整自动化闭环；换成任意 NPU 服务器时，
不能只克隆代码就直接提交，必须先完成对应 Runtime/Executor Adapter 的验证。仓库不会分发
API Key、SSH 凭据、模型权重、私有 Benchmark 资产、Lease 或历史 Session。

接手者必须提供：

1. 可登录的调度/开发节点和独立可写目录；
2. 模型与可选 MTP 权重路径、served-model 名称、网卡和环境脚本；
3. 镜像 digest、vLLM/vllm-ascend commit、CANN 版本及 NPU 拓扑；
4. 唯一 Lease 名称及对应资源执行器；
5. Agent Provider 凭据（只放环境变量）；
6. 内部、公开或自定义 Benchmark 三选一。

最短验收顺序是：安装依赖 → 生成 Git 忽略的私有配置 → 密钥扫描/单元测试 →
`CheckOnly` 或 `dry_run.sh` → 镜像身份和 Lease 预检 → 创建新 Session。任何预检失败都不应
绕过；应按错误补齐环境或新建适配包。详细命令分别见下方“两种自动化运行模式”和
[可迁移快速启动](docs/PORTABLE_QUICKSTART.md)。

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

# W8A8 默认使用独立自动注册表；需要历史人工边界时显式切换 curated
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

它会输出自动 `registry.generated.yaml` 以及 Active / Reserve / Fixed / Rejected Search Limits。`automatic_registry_v1` 已完整接入 Controller，并作为 W8A8 统一默认；生成的注册表、兼容策略、注入契约和 Search Limits 会冻结到 Session。人工审计的 23 项 `curated_registry_v1` 保留为显式可选路线。独立命令仍只写指定审计目录、不提交服务器任务。完整使用方式见 [`registry_builder/README.md`](tuning_pipeline/workflow/registry_builder/README.md)。

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

### 更换 ktp-lab 或资源调度系统

默认 `execution_mode: ktp_lab` 及其命令路径保持不变。新 Session 可以显式选择
`execution_mode: executor_adapter`，通过版本化 JSON Bridge 接入普通 SSH、Slurm、
Kubernetes 或内部调度系统。适配器只负责资源准备、只读预检、提交、状态、停止和
释放；B0、候选、Search Limits、Session、指标判定与失败恢复仍由通用 Controller
掌握。适配器源码、非敏感配置和能力声明会计算 SHA-256 并冻结，Resume 禁止更换。

入口、配置样例和返回 Schema 见
[`workflow/executor_adapters/README.md`](tuning_pipeline/workflow/executor_adapters/README.md)。
现有任务不会读取这条扩展路径；只有新建 Session 并显式选择时才加载外部适配器。

### 切换 Search-Space、Agent 策略与 Benchmark

本地与服务器自治在线闭环统一默认使用 `automatic_registry_v1 + codex + hierarchical_throughput_v1 + aligned_l1_v4`，且 `history_source: none`，不会把其他 Session 的候选或指标静默带入新实验。需要复用 B0 时必须显式导入并通过身份校验。新建 Session 时仍可显式选择 Search-Space、策略、Provider 或 Benchmark：

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
| Search-Space 构建 | `-SearchSpaceProfile` | `automatic_registry_v1`（默认）、`curated_registry_v1` | `tuning_pipeline/workflow/search_space_profiles.yaml` |
| Agent 选参策略 | `-StrategyProfile` | `hierarchical_throughput_v1`（默认）、`best_anchor_coverage_v2`、`best_anchor_coverage_v3` | `tuning_pipeline/workflow/continuous/strategy_profiles.yaml` |
| Benchmark | `-BenchmarkProfile` | `aligned_l1_v4`、`vllm_bench_public_v1`、`custom_adapter_v1` | `tuning_pipeline/workflow/continuous/benchmark_profiles.yaml` |

资源执行后端通过配置选择：默认 `ktp_lab`，外部系统使用 `executor_adapter`。它是
Runtime Adapter 的执行后端，不改变上述四个调优接口。

Agent Provider 另通过 `-AgentProvider` 选择，支持 `codex`、`anthropic`、`openai_compatible`、`deepseek` 和 `command`。API Key 只通过环境变量配置。以上 Profile 仅允许在新建 Session 时选择；续跑使用该 Session 已保存的配置。

Benchmark 与 Agent Provider、选参策略相互独立。没有 ServeBench/GuideLLM 权限时，推荐选择 `vllm_bench_public_v1`：它只调用服务镜像内已有的公开 `vllm bench serve`，参数在 `config.yaml` 的 `benchmark.vllm_bench_public` 中配置。需要接入自有测试时，选择 `custom_adapter_v1`，并将 `benchmark.custom_benchmark.adapter_path` 指向 `workflow/benchmark_adapters/` 白名单目录内的 Python 适配器；接口与示例见 [`benchmark_adapters/README.md`](tuning_pipeline/workflow/benchmark_adapters/README.md)。不同 Profile/配置会生成不同的 Benchmark 身份摘要，Controller 不会跨口径比较历史结果。

Runtime Adapter、四个接口、失败重试、规则兜底和 V2/V3 差异统一见 [框架总览的“核心控制策略”](框架.md#核心控制策略)。

### 运行兜底与服务器自治

所有自动恢复均由 Controller 的确定性规则执行，Agent 只能提供结构化建议，不能登录服务器、绕过 Search Limits 或直接提交任务。主要兜底如下：

| 层级 | 触发条件 | 自动动作与上限 | 超限行为 |
|---|---|---|---|
| Lease 准入（仅服务器自治默认启用） | 双节点暂时不是 `2/2 Ready`、心跳缺失或平台正在重调度 | 最多等待 7200 秒，每 30 秒检查；状态检查与提交之间出现 protocol-v2 心跳竞态时，同一候选和 run ID 最多重试 12 次 | 不提交新一轮；只有超时、资源被其他白名单外任务占用或身份矛盾才暂停 |
| 资源隔离 | 任一 `blocked_lease_names` 仍占用资源，或当前槽位未空闲 | 不等待、不抢占、不重叠提交 | 立即失败关闭 |
| 候选生成 | Agent 候选越过白名单、网格距离、组合约束或证据要求 | 最多重新选参 2 次 | 暂停，禁止提交非法候选 |
| Agent 协议 | API/CLI 瞬断、空响应、非 JSON，或输出附带 Schema 禁止的说明性字段 | 同一实验轮最多重试 4 次；仅剥离明确禁止的多余元数据并记录审计，不改参数值、不补字段 | 协议重试耗尽后由服务守护最多恢复 Controller 6 次，随后才暂停人工检查 |
| Benchmark | 单 Case、完整运行或指标编译发生已知可恢复错误（含未知 GuideLLM/ServeBench 错误） | 只要 `SERVICE_READY` 且无服务侧危险签名，先在仍运行的同一服务上：Case 最多 2 次、运行/指标各最多 3 次、完整矩阵共享最多 3 次；退出后仍由 Controller 保持同候选恢复 | 保存每次失败产物；仅 OOM/HCCL/EngineCore、参数非法、身份/权限/路径错误等禁止误套 Benchmark 重试 |
| 未知实验故障 | 确定性规则无法归类或无法给出修复 | Codex+DeepSeek 读取完整证据；若 Agent 仍想暂停但没有明确人工依赖，Controller 自动改为有预算的原参数诊断重跑；也可执行 Search Limits 内的最小参数修正 | 禁止 Agent 改镜像、拓扑、路径、Benchmark 或系统文件；明确人工依赖或预算耗尽才暂停 |
| 恢复总预算 | 多种恢复策略连续触发 | 瞬态同候选最多 6 次、Agent 诊断性重跑最多 2 次、参数修正最多 4 次，且同一失败链总计最多 10 个恢复轮 | 达到任一上限才暂停，防止 32 NPU 无限空耗 |
| Controller 同候选恢复 | Pod、网络、HCCL、端口或超时等瞬态故障 | 参数不变，最多额外提交 2 次 | 暂停人工处理，避免无限消耗 NPU |
| 服务守护 | Controller 进程意外退出 | systemd/Supervisor 按服务策略重启；已有运行轮次从冻结状态恢复；已完成 Benchmark 的轮次从 Agent 分析处续接，不重跑测评 | 明确不可恢复或超过恢复预算的 `paused_*`、终态、状态不一致或保留 STOP 标记时退出码 78，禁止盲目重启 |
| 状态与提交事务 | 写状态时掉电、状态文件损坏，或远端提交成功后 Controller 突然退出 | `state.json` 原子替换并保留同版本备份；提交前持久化 intent，提交后立即落盘 task/run 身份；重启时先对账再继续 | 主状态与备份都损坏，或无法证明远端提交身份时失败关闭，禁止猜测性重复提交 |
| 控制面读故障 | SSH/本地传输暂断、远端产物查询失败 | 单轮原地重试 3 次，仍失败则标为可恢复 Controller I/O 错误，由 Supervisor 走有上限恢复 | 不修改候选、不重跑已完成 Benchmark |
| DP 前端握手超时 | `SERVICE_READY` 前出现精确的五分钟 front-end response timeout，且 Lease 为一活跃一退出、没有 OOM/参数非法/HCCL 证据 | 归类为瞬态进程协调故障，保持候选走基础设施重试预算 | 任一危险签名存在时不套用该规则，转完整失败分析 |

失败目录不等于参数已经完成实验。搜索覆盖只认可两种证据：完整
Benchmark，或日志已把启动失败明确归因到候选参数/组合并经 Controller
校验为 `parameter_invalid`、`parameter_oom` 或可受限绕行的
`model_or_runtime_bug`。Lease、节点地址、网络、抢占和残留进程等失败不消费
候选覆盖，外部问题修复后必须保持原候选重试；需要回放较早的未测候选时用
`--replay-unmeasured-candidate round_NNN_label`，命令会保留原轮和被替代轮的审计
文件，且拒绝回放已经 Benchmark 或已被参数证据淘汰的候选。

若旧版本已因确定性可恢复故障进入 `paused_for_human`，服务器自治入口可执行
`service.sh auto-retry-paused` 后再启动 Supervisor/systemd。请求只会在 Controller
重新匹配白名单签名且重试预算未耗尽时被消费；新 Task/Run/轮次会原子写回状态。

默认 Windows→服务器主链路不启用 Lease 长等待，行为保持不变。服务器自治的参数位于 `server_autonomous/config.yaml`；完整服务启动、停止标记、日志和恢复命令见 [服务器自治文档](tuning_pipeline/workflow/continuous/server_autonomous/README.md)。

### Git 克隆后可直接复用的知识产物

仓库会跟踪并随 Git 一起分发当前正式的参数画像、跳过原因与 Tags；接手者无需先重新调用 Agent 就能阅读知识。人工或自动 Search Limits 都在新 Session 创建时重新编译，并只把该 Session 的冻结结果作为在线权威。源码 checkout、迁移运行目录、Session、临时队列和日志属于运行产物，不提交 Git。当前正式知识产物及其入口见 [产物与日志目录](docs/ARTIFACTS.md)。

## 两种自动化运行模式

仓库不是“本地版”和“服务器版”两套代码。GitHub 分发同一套参数知识、Controller、Agent/Benchmark 接口和远端运行脚本；区别只是 **Controller 运行在哪里**。Session、日志、API Key 和完整 Benchmark 原始产物属于运行态，不进入 Git。

| 模式 | Controller 位置 | vLLM / Benchmark 位置 | 本地关机 | 推荐场景 |
|---|---|---|---|---|
| 本地 → 服务器 | Windows 本地电脑 | NPU 计算节点 | 自动闭环不能继续推进 | 开发、调试、需要本地 Codex 登录态 |
| 服务器自治 | Linux 开发/调度节点，由 systemd 或 Supervisor 托管 | NPU 计算节点 | 不受影响 | 长时间无人值守实验 |

两条模式共用相同的核心闭环：

```text
B0/显式导入基线
  → 编译并冻结 Search Limits
  → Agent 生成候选
  → Controller 确定性校验
  → 资源执行器拉起 vLLM
  → Benchmark（一轮）
  → 指标归档与下一轮候选
```

同一 Lease 同一时间只能由一个 Controller 管理。不要让本地模式和服务器自治模式同时控制同一个 Lease 或 Session。

### 模式 A：本地 → 服务器

本地负责知识查询、候选生成、状态机和结果归档；服务器只负责资源调度、vLLM 与 Benchmark。第一次使用依次执行本 README 后面的“启动前配置”到“新建 Session 实验”：

```powershell
# 生成私有服务器配置
.\scripts\init-local-config.ps1 <所需参数>

# 只读预检
.\一键启动.ps1 -CheckOnly -NewSession

# Lease 不存在时仅执行一次
.\scripts\prepare-remote.ps1

# 从 B0 启动自动闭环
.\一键启动.ps1 -NewSession
```

本地电脑退出或关机后，已经提交的远端进程不等于完整闭环仍在运行；下一轮分析和提交需要本地 Controller 存活。

### 模式 B：服务器自治

代码、Controller、Session 和日志都位于 Linux 开发/调度节点；NPU 计算节点仍只运行本轮服务和 Benchmark。服务器不安装 Codex 时，使用 `agent.provider: deepseek`；没有内部 ServeBench 权限时，使用 `benchmark.profile: vllm_bench_public_v1`。

```bash
git clone https://github.com/akangw/LLM_parameter_tuning.git
cd LLM_parameter_tuning
python3 -m pip install -r tuning_pipeline/requirements-server-autonomous.txt

AUTO=tuning_pipeline/workflow/continuous/server_autonomous
cp "$AUTO/config.local.example.yaml" "$AUTO/config.local.yaml"
# 编辑 config.local.yaml：项目根目录、模型、网卡、环境脚本和唯一 Lease

bash "$AUTO/service.sh" prepare-env
# 编辑 $AUTO/.secrets/controller.env，填入 API Key；该文件必须保持 600 权限
chmod 600 "$AUTO/.secrets/controller.env"

bash "$AUTO/dry_run.sh"       # 不提交 NPU 任务
# dry-run 会留下终态审计文件，显式授权创建真实新 Session
bash "$AUTO/service.sh" authorize-new-session
bash "$AUTO/prepare_lease.sh" # Lease 不存在时仅执行一次
bash "$AUTO/preflight.sh"

# 无系统权限时使用 Supervisor
bash "$AUTO/service.sh" supervisor-install
bash "$AUTO/service.sh" supervisor-start
bash "$AUTO/service.sh" supervisor-status
```

只有选择内部 `aligned_l1_v4` 且拥有对应只读资产时才需要先运行
`bash "$AUTO/seed_assets.sh"`；选择公开 `vllm_bench_public_v1` 或自有
Benchmark 时不要依赖该私有资产复制脚本。

`config.local.yaml` 被 Git 忽略并由所有服务器自治入口自动识别；也可以用绝对路径环境变量 `VLLMTKB_CONFIG` 显式指定另一份配置。需要服务器重启后自动恢复时，优先使用 systemd user service，并由管理员为服务账号启用 lingering。完整命令、恢复边界和日志位置见 [服务器自治文档](tuning_pipeline/workflow/continuous/server_autonomous/README.md)。

## 适配其他 Ascend/NPU 服务器

“服务器有 NPU”只是资源前提，并不代表现有镜像、模型命令、DP/TP 拓扑和调度器可以直接复用。按变化范围选择适配路线：

| 目标环境 | 需要做什么 | 是否需要改 Controller |
|---|---|---|
| 同为 Atlas A3、GLM-5.2 W8A8、2 节点 × 16 NPU、相同 ktp-lab 契约 | 生成私有配置，替换 SSH/项目/模型/网卡/Lease 路径，重新核验镜像身份 | 不需要 |
| 仍为 Ascend，但更换模型、量化、镜像、DP/TP 或节点数 | 建立 Runtime Adapter，重新绑定 Scenario、B0、镜像身份、Topology Profile 和 Executor Profile，并完成真实启动与 Benchmark 验证 | 通常不需要改通用 Controller |
| 有 ktp-lab 以外的 Slurm、Kubernetes、SSH 或内部调度器 | 实现 `executor_adapter/v1` 的 prepare/check/submit/status/stop/release 接口，同时保留远端产物协议 | 不需要改调参状态机；需要实现执行器桥接 |
| 普通单机 Ascend/NPU 服务器、没有任何调度器 | 可实现 local/SSH Executor Adapter，直接管理本机 vLLM 进程、端口、健康检查和产物；仓库当前提供接口模板，但尚未宣称任意裸机零配置可运行 | 需要新增并验证执行器，不需要重写 Agent/Search/Benchmark 主链路 |
| NVIDIA/CUDA | 当前 Runtime Adapter 和参数知识以 Ascend 为边界，不属于已支持迁移 | 需要新的平台适配与完整验证 |

新服务器的最小交付信息是：

1. 登录或本机执行方式，以及一个独立可写项目目录；
2. NPU 型号、节点数、每节点卡数、DP/TP 和 rank 布局；
3. 服务镜像 digest、vLLM/vllm-ascend commit、CANN 版本；
4. 主模型与可选 MTP 权重路径、served-model 名称、网卡和环境脚本；
5. 资源调度器的准备、提交、状态、停止和释放命令；
6. Agent Provider/API Key 环境变量；
7. `aligned_l1_v4`、`vllm_bench_public_v1` 或自定义 Benchmark 三选一。

适配顺序必须是：先只读预检和镜像身份核验，再完成 B0，最后才允许 Agent 搜索。`planned` Runtime Adapter 不能提交真实任务；只有模型、镜像、拓扑、执行器、B0 和 Benchmark 全部验证后才能标记为 `integrated`。详细接口见 [Ascend Runtime Adapter](docs/ASCEND_RUNTIME_ADAPTERS.md) 和 [Executor Adapter](tuning_pipeline/workflow/executor_adapters/README.md)。

## 本地 → 服务器模式：从 GitHub 克隆后的启动流程

### 1. 启动前配置

```powershell
git clone https://github.com/akangw/LLM_parameter_tuning.git
cd LLM_parameter_tuning

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
LLM_parameter_tuning/
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

当前已验证的 `ktp_lab + aligned_l1_v4` 集成在 Benchmark 运行阶段仍以只读方式挂载：

```text
/mnt/host-model/slai/user-1-wangakang/wangakang/liuxin-workspace
```

该目录提供 ktp-lab Lease 控制文件和 GuideLLM 激活脚本。项目不会修改它；依赖范围、替换条件和风险见 [依赖说明](docs/DEPENDENCIES.md)。选择 `vllm_bench_public_v1` 并接入自己的 Executor 时不要求复用这套私有 Benchmark 路径。除已声明依赖外，新项目不读取或修改 `vllmTKB0706` 的代码、状态、Lease 或实验目录。

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
- [模型加载与真正 RFork 热启动](docs/MODEL_LOADING.md)
- [服务器自治 systemd / Supervisor 服务](tuning_pipeline/workflow/continuous/server_autonomous/README.md)

## 安全边界

- 本地模式由本地保存决策、状态和 Session；服务器自治模式将这些运行态隔离保存在服务器部署目录。两种模式都不把运行态提交 Git。
- README 中的 `/mnt/host-model/.../cjx-workspace/...` 是当前集成环境示例，不是框架硬性根目录；实际写入必须受使用者配置的可写根目录约束。
- 服务镜像和 Benchmark 容器均使用 Digest 固定身份。
- Runtime Adapter、Scenario、B0、镜像、Topology 和 Executor 文件均记录 SHA-256；同一 Session 禁止跨适配身份续跑。
- `planned` Runtime Adapter 和未集成 Executor 一律失败关闭，不能提交远端任务。
- 上游 `--enable-eplb` 在当前 Ascend 版本中禁止进入搜索；Native Dynamic EPLB 接线完成前保持 `false/0`。
- 失败或残缺结果不会进入性能比较。
