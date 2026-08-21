# 03 Benchmark：测量、比较与回填

这一层只负责回答：同一场景下，这个候选是否比基线或当前最佳 Anchor 更好。

## 当前生产与三种通用接入路线

| Profile | 适用情况 | 要提供什么 |
|---|---|---|
| `decode_only_c32_v1` | 当前生产 Decode 吞吐调优 | 固定 `decode-256-2048`、C32 请求集；输出吞吐主指标及 TTFT/TPOT P50/P90 |
| `aligned_l1_v4` | 有内部 ServeBench/GuideLLM 权限 | 内部 Benchmark 环境和数据 |
| `vllm_bench_public_v1` | 没有内部权限 | 镜像中可用的公开 `vllm bench serve` |
| `custom_adapter_v1` | 使用自有 Benchmark | 输出 result-v1 的适配器 |

Benchmark Profile 会冻结负载、并发、输入输出长度、重复次数、指标及门禁。同一 Session
不能中途切换，不同 Benchmark 的分数不能横向混合。

## 测量闭环

```text
候选服务 Ready
→ Warmup
→ 固定 Case/负载执行
→ 完整性、错误率、吞吐、TTFT/TPOT 等门禁
→ metrics.json + comparison
→ Controller 回填本轮结果
→ Agent 下一轮读取
```

## 扩展位置

- Profile 定义：`tuning_pipeline/workflow/continuous/benchmark_profiles.yaml`
- 自定义适配器协议：`tuning_pipeline/workflow/benchmark_adapters/README.md`
- Session 冻结定义：`session_config.yaml`
- 本轮结果：`round_NNN_*/05_results/`
- 远端完整原始证据：`<remote_project>/workflow/auto/lab_runs/`

选择公开 Benchmark：

```powershell
.\scripts\scenario.ps1 -Action start `
  -Name glm52-w8a8-a3-2n-dp2-tp16 `
  -BenchmarkProfile vllm_bench_public_v1
```
