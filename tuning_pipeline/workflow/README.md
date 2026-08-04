# Workflow

- `search_space_compiler/`：从画像、场景、策略和历史生成 Session Search Limits。
- `sidecars/`：Agent 画像召回、关联参数和确定性运行规则。
- `continuous/`：在线 Controller、远端执行脚本、指标门禁和恢复状态机。
- `runs/`：早期只读发现快照，不参与当前主流程。

统一从项目根目录运行：

```powershell
.\一键启动.ps1 -CheckOnly
.\一键启动.ps1
```

交接与运行说明见根目录 `docs/`。
