# GLM-5.2 W4A8C8 · A3 1×16 NPU · DP2-local/TP8

该场景与 W8A8 同级但完全隔离，使用专家已有参数作为 A0，不把它解释为官方默认 B0。
当前状态为 `planned`：当前批准的参数画像对应运行镜像只支持 W8A8。必须先取得并核验
支持 W4A8C8 的镜像，再完成真实镜像、A0、Benchmark 和 Search Limits 四项证明；
在此之前，统一入口会拒绝 `prepare/start/resume`。

固定定义和“为什么现在还没有正式产物”见 [`ARTIFACTS.md`](ARTIFACTS.md)。

默认本地 Runtime Root：

```text
tuning_pipeline/workflow/continuous/scenario_runs/glm52-w4a8c8-a3-1n-dp2-tp8/
```

可以先生成个人配置并做结构校验：

```powershell
.\scripts\scenario.ps1 -Action init -Name glm52-w4a8c8-a3-1n-dp2-tp8
.\scripts\scenario.ps1 -Action validate -Name glm52-w4a8c8-a3-1n-dp2-tp8
.\scripts\scenario.ps1 -Action check -Name glm52-w4a8c8-a3-1n-dp2-tp8
```

`check` 会显示尚未满足的适配门禁；不要临时把 `planned` 改成 `integrated`。
