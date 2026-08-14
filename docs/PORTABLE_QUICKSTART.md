# 可迁移快速启动

这份说明面向第一次拿到仓库、需要在自己的服务器上建立一条新实验链路的使用者。仓库中的参数知识、Search-Space Compiler、Agent Provider、Benchmark 适配器和 Controller 可以直接复用；服务器身份、模型、镜像、Lease 和凭据必须由使用者提供。

如果只想按一个明确场景完成配置和启动，优先使用根目录
[`scenarios/`](../scenarios/README.md) 和统一入口 `scripts/scenario.ps1`。它把 W8A8 与
W4A8C8 作为同级独立场景展示，并为每个场景使用不同 Runtime Root；本文保留底层
配置、版本迁移和手工接入细节。

## 1. 运行边界

当前远端执行器已经验证的正式拓扑是 Atlas A3、两节点、每节点 16 NPU，以及 GLM-5.2 W8A8。更换 SSH 主机、可写目录、模型路径、API Provider 或 Benchmark 属于配置操作。更换模型家族、加速器类型或 DP/TP 拓扑时，还需要重新生成参数画像、场景、B0 和远端执行器配置，不能只替换一个路径后默认结果仍然有效。

## 2. 使用者需要准备

- 本地 Windows PowerShell、Git、Python 3.11+ 和可用的 SSH 客户端。
- 一台能通过 SSH 到达的调度主机，主机上可以使用 `ktp-lab`。
- 服务节点可读取的模型目录、Ascend/CANN 环境脚本、版本固定的服务镜像。
- 一个属于使用者自己的远端可写项目目录。
- 一种 Agent：已登录的 Codex CLI，或 Anthropic/OpenAI-compatible/DeepSeek API Key，或自定义结构化命令。
- 一种 Benchmark：内部 `aligned_l1_v4`、公开 `vllm_bench_public_v1`，或实现 result-v1 协议的自定义适配器。

API Key 只放环境变量，不写入 YAML 或 Git。例如：

```powershell
$env:DEEPSEEK_API_KEY = "..."
```

## 3. 克隆与安装

```powershell
git clone https://github.com/akangw/LLM_parameter_tuning.git
cd LLM_parameter_tuning
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\tuning_pipeline\requirements-runtime.txt
.\scripts\fetch-sources.ps1
```

需要运行仓库测试时，改为安装 `requirements-dev.txt`，随后可在仓库根目录直接执行 `python -m pytest`。

在 `~/.ssh/config` 中配置自己的 SSH 别名，并先确认 `ssh <alias>` 成功。

## 4. 生成个人配置

下面的命令生成 Git 自动忽略的 `config.local.yaml`。顶层启动与远程准备脚本会自动识别它：

```powershell
.\scripts\init-local-config.ps1 `
  -RemoteHost my-npu `
  -RemoteProject /my/writable/path/auto-vllm `
  -LeaseName my-vllm-glm52-a3-32npu-v1 `
  -ModelPath /models/GLM-5.2-w8a8 `
  -ServedModelName glm-5 `
  -NetworkInterface bond0 `
  -InitEnvScript /models/init_env.sh `
  -AgentProvider deepseek `
  -BenchmarkProfile vllm_bench_public_v1 `
  -SearchSpaceProfile automatic_registry_v1
```

也可以复制 `tuning_pipeline/workflow/continuous/config.local.example.yaml` 手工修改，或在启动时显式传入任意叠加配置：

```powershell
.\一键启动.ps1 -Config .\path\to\my-config.yaml -CheckOnly -NewSession
```

## 5. 核验镜像身份

根据目标服务器上的真实镜像生成以下两个文件：

- `tuning_pipeline/workflow/continuous/remote/image_version_manifest.yaml`
- `tuning_pipeline/workflow/continuous/activation.approved.yaml`

必须核对镜像 digest、vLLM commit、vllm-ascend commit 和参数画像版本。`activation.approved.yaml` 只有在完成真实核验后才能设置为批准；Controller 会在提交昂贵任务前 fail closed。

推荐使用封装命令从可信探针 JSON 同时生成并交叉校验两个 YAML，避免手工同步错误：

```powershell
.\scripts\verify-image-identity.ps1 validate
.\scripts\verify-image-identity.ps1 approve --probe-json C:\secure\image-probe.json --approved-by operator --dry-run
```

Linux、Docker、拓扑 Profile、Session 导入导出的完整说明见
[`LINUX_DOCKER_CONTROLLER.md`](LINUX_DOCKER_CONTROLLER.md)。
更换 Ascend 模型、镜像、量化或 DP/TP/节点数时，使用
[`ASCEND_RUNTIME_ADAPTERS.md`](ASCEND_RUNTIME_ADAPTERS.md) 中的运行适配包流程。

如果换了 vLLM 或 vllm-ascend 版本，先执行：

```powershell
.\scripts\migrate-versions.ps1 -Vllm <vllm-ref> -VllmAscend <ascend-ref>
```

## 6. 预检与启动

```powershell
# 只读端到端检查，不提交任务
.\一键启动.ps1 -CheckOnly -NewSession

# Lease 不存在时才执行一次
.\scripts\prepare-remote.ps1

# 从 B0 创建新 Session，成功后自动进入搜索闭环
.\一键启动.ps1 -NewSession

# 查看状态
.\scripts\status.ps1
```

没有 ServeBench/GuideLLM 权限时使用 `vllm_bench_public_v1`。接入自己的 Benchmark 时，复制 `workflow/benchmark_adapters/example_http_adapter.py`，实现 `benchmark_result.schema.json` 定义的 result-v1 输出，再选择 `custom_adapter_v1`。

## 7. 迁移已有 Session

Git 只分发代码和可复现知识，不分发运行态。跨电脑续跑还要安全地复制：

```text
tuning_pipeline/workflow/continuous/state.json
tuning_pipeline/workflow/continuous/experiments/<session-id>/
```

新电脑应先执行 `-CheckOnly`，并确认没有另一台 Controller 正在控制同一个 Lease，之后才运行 `-Resume`。
