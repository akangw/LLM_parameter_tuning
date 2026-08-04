# Local experiment archives

该目录保存 Controller 生成的本地 Session 证据，默认不提交 Git。

需要跨机器续跑时，单独复制目标 Session 目录和上一级 `state.json`；源码仓库只保存结构、代码和汇总说明。

每个 Session 冻结 `session_config.yaml`、镜像身份和 `00_search_space/`；每轮固定采用 `00_context` 到 `06_agent_analysis` 七层结构。远端 Master、Worker、Benchmark 和 ServeBench 原始文件镜像在每轮 `04_runtime/`，正式指标或失败证据在 `05_results/`。

完整文件含义和日志查看顺序见项目根目录 `框架.md` 与 `docs/ARTIFACTS.md`。
