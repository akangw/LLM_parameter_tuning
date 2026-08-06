# GLM-5.2-W4A8C8 单节点 A0 场景

## 场景身份与隔离

该场景不替换 W8A8 默认链路，而是新增独立 Runtime Adapter：

```text
scenario_id: glm52-w4a8c8-a3-1n16-dp2tp8-aligned-l1
model: /models/share/GLM-5.2-w4a8c8
topology: 1 node x 16 NPU, DP2-local2/TP8
initial anchor: a0_existing_tuned
benchmark: aligned_l1_glm52_w4a8c8_v1
search space: automatic_registry_glm52_w4a8c8_v1
strategy: best_anchor_coverage_v2
```

公共框架代码可以复用；以下可变状态必须独立：

| 状态 | W4A8C8 独立位置/身份 |
|---|---|
| 本地 Controller | `runtime/glm52_w4a8c8_a3_1n16_dp2tp8/` |
| 远端项目 | `.../cjx-workspace/vllmtkb-glm52-w4a8c8-a3-1n16-dp2tp8` |
| Lease | `vllmtkb-glm52-w4a8c8-a3-1n16-dp2tp8-v1` |
| Session | 前缀 `glm52_w4a8c8_a3_1n16_dp2tp8` |
| 模型缓存 | 新远端项目的 `cache/` |
| 实验结果 | 新远端项目的 `workflow/auto/lab_runs/` |

因此本地启动、状态查询、恢复和停止都必须携带同一个 `-RuntimeRoot`。

## A0

[`a0_glm52_w4a8c8_existing_tuned.yaml`](../tuning_pipeline/workflow/baselines/a0_glm52_w4a8c8_existing_tuned.yaml)
保存已有 `start_service.sh` 的参数。A0 使用 `explicit_candidate` 路径，是第一轮实测
incumbent，不执行 B0 的官方默认值回读。

`block_size=128` 根据当前 Ascend 画像保持 Session 固定。DSA-CP 和多流共享专家已进入
该场景独立的自动 Search Limits；实际激活集合会随 Session 一起冻结，且
`history_source=none`，不会继承 W8A8 历史最优值。

## 首次上线门禁

适配包当前为 `planned`，用于防止未经验证的拓扑直接提交。首次在线运行前依次完成：

1. 在目标 Lease 内只读核对镜像 digest、vLLM commit 和 vllm-ascend commit；
2. 确认它们与现有批准身份完全一致，否则停止并迁移画像；
3. 验证单节点 Lease 只有一个 master，且可见 16 张 NPU；
4. 运行一次 A0，核对生成的 `vllm_common_command.txt` 与原脚本；
5. 完成一轮独立 Aligned-L1，并核验 tokenizer、Suite、Dataset 指纹；
6. 保存 Search Limits 快照后，将适配包四项 attestation 设为 true，并改为
   `status: integrated`。

不得通过临时绕过 `planned` 门禁来启动正式 Session。

## 本地—服务器启动

配置模板：

```text
tuning_pipeline/workflow/continuous/launch_profiles/
glm52_w4a8c8_a0_aligned_l1.example.yaml
```

适配包完成真实验证并升级为 `integrated` 后执行：

```powershell
$config = ".\tuning_pipeline\workflow\continuous\launch_profiles\glm52_w4a8c8_a0_aligned_l1.example.yaml"
$runtime = ".\tuning_pipeline\workflow\continuous\runtime\glm52_w4a8c8_a3_1n16_dp2tp8"

# 不提交任务的预检
.\一键启动.ps1 -Config $config -RuntimeRoot $runtime -CheckOnly -NewSession

# 同步受管脚本并创建独立 Lease
.\scripts\prepare-remote.ps1 -Config $config

# A0 完成后自动进入 Agent V2
.\一键启动.ps1 -Config $config -RuntimeRoot $runtime -NewSession

# 独立查看、恢复和停止
.\scripts\status.ps1 -RuntimeRoot $runtime
.\一键启动.ps1 -RuntimeRoot $runtime -Resume
.\scripts\stop.ps1 -RuntimeRoot $runtime
```

本路线仍由 Windows Controller 驱动，运行期间本地电脑需要保持开机。
