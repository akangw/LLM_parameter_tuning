# 当前实验状态

GitHub 只分发代码、配置模板和可复现知识，不发布具体实验分数、吞吐、
延迟、候选参数、逐轮故障、Lease 身份或 Session 标识。README 和本页因此
不维护运行中的实验摘要，避免把容易过期的运行细节当成项目默认事实。

实际状态以对应运行环境中的受管 Session 为准：

```powershell
# 本地 → 服务器模式
.\scripts\status.ps1
```

```bash
# 服务器自治模式
bash tuning_pipeline/workflow/continuous/server_autonomous/status.sh
```

需要交接实验时，使用 Session 导出/导入能力传递完整审计产物；不要把结果
手工复制进公开 README。B0 定义、Search Limits、Agent 策略和 Benchmark
身份仍会冻结在每个 Session 中，保证结果可追溯。
