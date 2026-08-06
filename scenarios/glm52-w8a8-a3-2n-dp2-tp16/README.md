# GLM-5.2 W8A8 · A3 2×16 NPU · DP2/TP16

这是当前已集成、可正式运行的默认场景。B0 采用官方源码默认启动，只增加经过测量的
`max_model_len=64000` 可部署性覆盖；B0 成功并回填实际默认值后进入 Agent V2。

固定定义与运行产物位置见 [`ARTIFACTS.md`](ARTIFACTS.md)。

本场景的配置、状态和产物不得与 W4A8C8 或其他拓扑共用。默认本地 Runtime Root：

```text
tuning_pipeline/workflow/continuous/scenario_runs/glm52-w8a8-a3-2n-dp2-tp16/
```

首次使用：

```powershell
.\scripts\scenario.ps1 -Action init -Name glm52-w8a8-a3-2n-dp2-tp16
# 编辑 scenarios/.../operator.local.yaml
.\scripts\scenario.ps1 -Action check -Name glm52-w8a8-a3-2n-dp2-tp16
.\scripts\scenario.ps1 -Action prepare -Name glm52-w8a8-a3-2n-dp2-tp16
.\scripts\scenario.ps1 -Action start -Name glm52-w8a8-a3-2n-dp2-tp16
```
