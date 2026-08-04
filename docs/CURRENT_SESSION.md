# 当前实验摘要

Session：`glm52_continuous_20260804_112130`

当前状态：`stopped_after_failed_round`。A3 在服务初始化期间收到人工停止请求，没有形成 Benchmark 指标；它不是已证明的参数 OOM。

| 轮次 | 主要变化 | 主分数 | 验收 |
|---|---|---:|---|
| A0 | 基线 | 602.5576 | 正式锚点 |
| A1 | batch tokens 8192、显存 0.95 | 无 | 8192 prefill OOM |
| A1F1 | batch tokens 回退 4096、显存 0.95 | 628.9742 | 吞吐 +4.38%，延迟门禁失败 |
| A2 | MTP K=1、图上限 128 | 508.7708 | 吞吐 -15.56%，拒绝 |
| A3 | seqs 64、显存 0.95、K=3、图上限 256 | 无 | 人工停止，未评价 |

当前唯一正式接受的最佳锚点仍是 A0。`max_num_batched_tokens=8192` 已按当前镜像、模型和拓扑精确隔离。

继续该 Session 前，先复制本机未入库的 `state.json` 和完整 Session 目录，并执行：

```powershell
.\一键启动.ps1 -Resume
```
