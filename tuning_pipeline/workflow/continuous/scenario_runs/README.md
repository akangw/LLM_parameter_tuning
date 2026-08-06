# Scenario-isolated runs

新场景入口把本地可变状态和实验产物统一放在这里，并以场景 ID 隔离：

```text
scenario_runs/
├─ glm52-w8a8-a3-2n-dp2-tp16/
│  ├─ state.json
│  ├─ controller.lock
│  ├─ logs/controller/
│  └─ experiments/<session-id>/
└─ glm52-w4a8c8-a3-1n-dp2-tp8/
   ├─ state.json
   ├─ controller.lock
   ├─ logs/controller/
   └─ experiments/<session-id>/
```

除本说明外，`scenario_runs/` 下的内容均不进入 Git。旧版直接运行 Controller 产生的历史
Session 继续保留在相邻的 `experiments/` 中，不自动移动、复制或删除。
