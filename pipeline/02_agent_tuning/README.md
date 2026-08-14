# 02 Agent 调参：从基线到下一候选

这一层只负责回答：在已经冻结的 Search Limits 内，下一轮应该试什么参数。

## Agent 能看到什么

- 当前场景、模型、量化、镜像和拓扑身份。
- B0/A0 基线及实际生效参数。
- `parameter_portraits.agent.yaml` 中与 Active 参数相关的知识。
- Search Limits 候选值、依赖和禁止组合。
- 已完成轮次的指标、失败证据和当前最佳 Anchor。
- Agent Strategy Profile；当前 W8A8 默认是 `hierarchical_throughput_v1`，
  `best_anchor_coverage_v2/v3` 保留为新 Session 的显式可选策略。

更换 Codex、DeepSeek 或其他 OpenAI-compatible Provider，只更换决策模型，不减少以上证据包。

## Agent 不能做什么

- 不能直接 SSH 登录服务器或修改文件。
- 不能加入 Search Limits 之外的参数或候选值。
- 不能改变模型、镜像、拓扑、Benchmark 或 Session 身份。
- 不能用文字结论覆盖 Controller 的确定性校验和指标门禁。

## 一轮调参

```text
最佳已通过 Anchor
→ Agent 提出 1～3 个有证据的变化
→ Controller 校验白名单、值域、组合、距离和重复候选
→ 渲染 candidate.env 和完整 vLLM 命令
→ 远端启动服务
→ Benchmark
→ 接受 / 拒绝 / 有上限重试 / 暂停人工处理
```

## 最终看哪里

```text
round_NNN_*/
├─ 02_parameters/       候选值与实际命令
├─ 05_results/          metrics、comparison 或 failure
└─ 06_agent_analysis/   Agent 输入、证据和结构化决策
```

框架实现集中在 `tuning_pipeline/workflow/continuous/`。普通使用者无需从该目录开始阅读。
