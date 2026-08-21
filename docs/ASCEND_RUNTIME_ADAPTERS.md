# Ascend 模型、镜像与拓扑适配包

> 本文描述通用适配机制。当前生产 Runtime Adapter 是
> `glm52_w8a8_a3_dp4_tp8_decode_priority_v3`，见 [CURRENT_DEFAULTS.md](CURRENT_DEFAULTS.md)；
> 下文出现的 DP2/TP16 “默认”仅指早期通用适配示例。

运行适配包把一次可比较实验所依赖的兼容性边界组合成单一身份：

```text
Ascend 平台
+ 模型家族/变体/权重格式
+ 镜像 digest 与 vLLM/vllm-ascend commit
+ 节点/NPU/DP/TP 拓扑
+ 远端执行器能力
+ Search-Space 场景
+ B0 定义
+ Benchmark Profile
+ Agent 策略
= Runtime Adapter
```

适配包只负责选择和冻结兼容性关键项。SSH、远端可写目录、模型实际路径、网卡、
端口和 API Key 仍属于操作者的 `config.local.yaml`，不会写进公共适配包。

## 当前适配包

默认 `glm52_w8a8_a3_dp2_tp16` 完整保留现有行为：GLM-5.2 W8A8、Atlas A3、
两节点 × 16 NPU、DP2/TP16、现有 ktp 两角色执行器、B0-deployable、人工审计
Search Limits、策略 V2 和 Aligned-L1。

新增的 `glm52_w4a8c8_a3_single_dp2_tp8` 为 planned 适配包：GLM-5.2-W4A8C8、
单节点 × 16 NPU、DP2-local2/TP8、A0 专家基线和独立 Aligned-L1。单节点执行器代码
已经独立于 `ktp_two_role`，但在一次真实镜像身份、A0 和 Benchmark 验证完成前不会
升级为 runnable。操作说明见 [GLM-5.2-W4A8C8 单节点 A0](GLM52_W4A8C8_A0.md)。

Controller 在创建 Session 前解析适配包，并把以下内容冻结到 `session_config.yaml`：

- 适配包完整文档和 SHA-256；
- 模型契约；
- 每个场景、B0、镜像、拓扑和执行器文件的 SHA-256；
- 解析后的拓扑与执行器约束；
- 镜像 manifest 和 activation 内容；
- 最终 Search Limits、Benchmark 和策略定义。

旧 Session 没有适配包字段时按 `legacy_implicit_ascend` 读取自身冻结配置，不会被新
默认值覆盖。

## 新建适配包

先完成新镜像身份探针、场景 YAML 和 B0 定义，再生成 planned 适配包：

```powershell
.\scripts\new-runtime-adapter.ps1 scaffold `
  --name glm52-bf16-a3-dp4-tp8 `
  --model-family glm `
  --model-variant glm-5.2 `
  --weight-format bf16 `
  --image-manifest tuning_pipeline/workflow/continuous/adapters/glm52-bf16/image.yaml `
  --activation tuning_pipeline/workflow/continuous/adapters/glm52-bf16/activation.yaml `
  --scenario tuning_pipeline/workflow/search_space_compiler/scenario.glm52-bf16-a3.yaml `
  --baseline tuning_pipeline/workflow/baselines/b0_glm52_bf16.yaml `
  --nodes 4 --npu-per-node 8 `
  --data-parallel-size 4 --tensor-parallel-size 8 `
  --worker-replicas 3 `
  --executor ktp_multi_role `
  --output tuning_pipeline/workflow/continuous/adapters/glm52-bf16-a3-dp4-tp8.yaml
```

随后查看阻塞项：

```powershell
.\scripts\new-runtime-adapter.ps1 validate `
  tuning_pipeline/workflow/continuous/adapters/glm52-bf16-a3-dp4-tp8.yaml
```

planned 适配包不能启动实验。它必须完成四项真实验证：

1. `executor_validated`：Lease 角色、worker rank、节点发现、退出和日志回收已验证；
2. `b0_validated`：新模型/量化在该拓扑上的可部署官方默认基线已测量；
3. `benchmark_validated`：服务名、tokenizer、数据集和门禁已经对齐；
4. `search_space_validated`：场景镜像身份、能力快照和 Search Limits 已重新编译。

完成实现和验证后，以 `--integrated` 和四个 `--*-validated` 标志重新生成最终适配包。
工具会再次校验拓扑执行器、镜像批准、场景 digest/commit 和 B0 文件，任一项不匹配都
拒绝生成 integrated 状态。

## 使用适配包

生成个人配置时选择适配包：

```powershell
.\scripts\init-local-config.ps1 `
  -RemoteHost my-npu `
  -RemoteProject /my/writable/auto-vllm `
  -LeaseName glm52-bf16-a3-dp4-tp8-v1 `
  -ModelPath /models/GLM-5.2-bf16 `
  -ServedModelName glm-5 `
  -NetworkInterface bond0 `
  -InitEnvScript /models/init_env.sh `
  -RuntimeAdapter workflow/continuous/adapters/glm52-bf16-a3-dp4-tp8.yaml
```

然后仍然走原来的流程：

```text
只读预检 → 创建新 Lease（如需要）→ 新 Session → 新 B0
→ 实际值回读 → Agent 选参 → Benchmark → 验收/恢复闭环
```

不能在已有 Session 中切换适配包；模型、镜像、拓扑和场景变化必须创建新 Session。

## 新拓扑为什么仍需要执行器适配

`topology_profiles.yaml` 描述“想要什么资源”，`executor_profiles.yaml` 描述“现有代码
确实会怎么启动”。例如四节点 DP4 不仅是四个数字，还需要为三个 worker 分配不同的
DP start rank。只有对应执行器实现和测试完成后，才把它标为 `integrated`。

这层封装让后续工作变成一条固定路线，并防止不完整适配误跑；它不会虚构 ktp 平台
尚未确认的 replica-index 环境变量或 rank 规则。
