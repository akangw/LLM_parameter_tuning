# 运行与恢复

## 前置条件

- Windows PowerShell、Python 3.11+、Git、Codex CLI。
- `pip install -r tuning_pipeline/requirements-runtime.txt`。
- SSH 别名 `hetao-npu` 可用。
- `tuning_pipeline/workflow/continuous/activation.approved.yaml` 与目标镜像、Digest 和两个源码 commit 一致；校验值来自 `remote/image_version_manifest.yaml`，不写死在启动代码中。
- 如本地没有固定源码，先运行 `scripts/fetch-sources.ps1`。

服务器差异统一配置在 `config.yaml` 的 `remote_*`、`deployment.*`、`lab.*` 和 Benchmark 路径中。主模型路径、served-model、量化方式、网卡及环境初始化脚本会写入每轮 `candidate.env` 并冻结进 Session，不需要修改远端 Bash。

## 常用命令

```powershell
# 无远端提交的前置检查
.\一键启动.ps1 -CheckOnly

# 首次同步脚本并创建独立 Lease
.\scripts\prepare-remote.ps1

# 新建 Session
.\一键启动.ps1 -NewSession

# 默认使用人工审计注册表；也可以显式写出或切换自动替代路径
.\一键启动.ps1 -NewSession -SearchSpaceProfile curated_registry_v1
.\一键启动.ps1 -NewSession -SearchSpaceProfile automatic_registry_v1

# 新建 Session 时选择 Benchmark；续跑不可切换
.\一键启动.ps1 -NewSession -BenchmarkProfile aligned_l1_v4

# 续跑当前 Session
.\一键启动.ps1 -Resume

# 前台运行，便于观察
.\一键启动.ps1 -Resume -Foreground

# 查看状态
.\scripts\status.ps1

# 优雅停止
.\scripts\stop.ps1
```

`-CheckOnly` 会验证本地配置、所选 AI Provider、SSH 连通性以及持久 Lease
是否空闲可用，但不会启动 Controller、提交实验或修改服务器文件。
检查新 Session 时使用 `-CheckOnly -NewSession`；检查恢复链路时使用
`-CheckOnly -Resume`，后者读取 Session 冻结配置，且仅在状态记录着活动任务时
允许 Lease 保持运行。

默认启动器会检查 PID、画像/标签进度、激活审批、Python 依赖和实际选中的
Agent Provider。发现可续跑状态时优先使用 `--resume`。

## 停止语义

优雅停止只写入 `STOP_REQUESTED`。Controller 会继续回收当前轮次，然后停止提交下一轮。若服务尚未产生指标，状态分类为 `operator_stop_before_metrics`，不能继承旧轮次的 OOM 分类。

`-StopActiveTask` 会读取当前 Session 冻结的 `remote_host`、`remote_project` 和 `lease_name` 后向该 Lease 发出停止请求，只有明确需要立即停止计算时才使用；它不依赖仓库内的默认服务器硬编码。

## 恢复边界

- `--resume` 总是加载 Session 内冻结的配置和 Search Limits。
- Search-Space、Agent Strategy 和 Benchmark Profile 只允许新建 Session 时选择；恢复时禁止覆盖。
- 服务镜像身份不一致时拒绝恢复。
- Benchmark 容器、Suite、Schema、Tokenizer 或 Dataset 指纹变化时，应创建新 Session。
- 当前平台不允许 Agent 启用上游 `--enable-eplb`。
- 同一项目不得同时运行两个 Controller；旧项目和新项目分别使用自己的锁与 Lease。

## 模型权重加载慢

模型位于 DTFS。固定 vLLM 版本的自动识别只把 NFS/Lustre 当作网络文件系统，
因此默认会退回 Safetensors lazy mmap；在当前 182-shard、TP16/节点的部署中，
历史日志显示权重加载约需 22～29 分钟。

项目现已把以下设置冻结为服务启动契约：

```yaml
model_loading:
  safetensors_load_strategy: prefetch
  safetensors_prefetch_num_threads: 8
  safetensors_prefetch_block_size: 16777216
```

这三个参数会同时应用于 B0 和后续 Agent 候选轮次。它们只改变权重读取策略和启动耗时，不属于 Search Limits，也不改变 B0 对吞吐/延迟相关 vLLM 参数采用官方源码默认值的定义。

每轮的 `candidate.env`、`effective_config.yaml`、`vllm_common_command.txt` 和
`startup_timeline.jsonl` 都会记录实际设置。若日志仍出现
`Auto-prefetch is disabled`，说明远端脚本或 Session 配置未同步，应停止提交
新轮次并先执行脚本哈希核对。不要直接提高线程数：两个节点会同时读取 DTFS，
过高并发可能反而放大共享存储抖动。

镜像拉取和模型权重读取是两件事。当前使用持久 Lease，普通
`ktp-lab run` 不会重建 Pod；只有创建新 Lease 或改变镜像身份时才应出现
ImagePull/ContainerCreating。

## 远端只读检查

```powershell
ssh hetao-npu "cd /mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-418bd627-32c8cf190 && ktp-lab status --lease vllmtkb-418bd627-32c8cf190-glm52-a3-32npu"
```

远端正式运行目录之外的 `/mnt/host-model/slai/user-1-wangakang/wangakang` 内容只允许读取。
