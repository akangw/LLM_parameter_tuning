# 产物与日志索引

本文是文件级索引；框架原理以根目录 `框架.md` 为准。

## 1. 离线参数知识产物

| 阶段 | 权威产物 | 含义 |
|---|---|---|
| 源码身份 | `portrait_pipeline/build/target-context.snapshot.yaml` | 固定提交、镜像、硬件和目标场景 |
| 结构化提取 | `portrait_pipeline/build/extracted_parameters/` | 1540 个源码参数表面、provenance 和覆盖证据 |
| 初筛/迁移 | `portrait_pipeline/build/migration_candidates/` | Stage-1 候选、排除项和版本分类 |
| 参数画像 | `portrait_pipeline/outputs/ParameterYAML/` | 340 份正式 ParameterYAML |
| 跳过证据 | `portrait_pipeline/outputs/skipped/` | 105 份有理由跳过记录 |
| Tags | `tuning_pipeline/tag_params/output/params/` | 340 份五维标签画像 |
| Tags 审计 | `tuning_pipeline/tag_params/output/audit.json` | 数量、分布、召回与错误 |
| Search Limits | `tuning_pipeline/search_limits/` | 最新编译边界和审批证据 |

以上正式知识产物纳入版本控制并随 Git 克隆分发。当前仓库基线包含 340 份参数
画像、105 份跳过说明、340 份 Tags，以及 Search Limits 目录内的编译产物与证据。
计数变化时以仓库实际文件和各阶段 manifest/audit 为准。

新版本试迁移不会覆盖上表正式产物；其完整镜像链位于
`portrait_pipeline/build/version_migrations/<commit-pair>/00_sources..05_search_limits/`。
场景快照和 `run-manifest.json` 记录版本、Provider、场景与各阶段位置。

Search Limits 文件：

- `agent_search_limits.yaml`：只给 Agent 的 Active 搜索轴。
- `search_space.compiled.yaml`：Active/Reserve/Fixed/Rejected 完整分类。
- `audit.json`：过滤、依赖、风险和主流程接入审计。
- `approval_queue.yaml`：需要人工批准的候选。
- `rotation_report.yaml`：历史驱动的换入/换出说明。
- `manifest.json`：输入和输出身份。

## 2. 离线日志

| 位置 | 文件 | 查看目的 |
|---|---|---|
| `portrait_pipeline/build/codex_portrait_pipeline/run/worker_logs/` | `<task>.prompt.txt` | 画像 Agent 输入 |
| 同上 | `<task>.out.log` | 画像任务标准输出 |
| 同上 | `<task>.err.log` | 画像任务执行轨迹与错误 |
| `portrait_pipeline/build/codex_portrait_pipeline/run/` | `index.json`、`supervisor-status.json` | 445 个画像任务总体状态 |
| `tuning_pipeline/tag_params/output/logs/parameters/` | `<param>.attempt-N.{stdout,stderr}.log` | 单参数 Tags 尝试 |
| 同上 | `<param>.attempt-N.response.json` | Tags Agent 结构化响应 |
| `tuning_pipeline/tag_params/output/logs/pipeline/` | `pipeline-tagging.*.log` | Tags 总流水线 |
| 同上 | `pipeline-audit.*.log` | Tags 审计 |
| 同上 | `pipeline-search-limits.*.log` | Search Limits 编译 |

画像和 Tags 的大体积 Agent 日志属于本地构建证据，默认不进入 Git；正式 YAML、审计和 Manifest 进入 Git。

## 3. 在线 Session 产物

```text
tuning_pipeline/workflow/continuous/experiments/<session>/
├─ session_config.yaml
├─ image_version_manifest.yaml
├─ 00_search_space/
└─ round_NNN_label/
   ├─ 00_context/
   ├─ 01_query/
   ├─ 02_parameters/
   ├─ 03_submission/
   ├─ 04_runtime/
   ├─ 05_results/
   └─ 06_agent_analysis/
```

| 层级 | 关键文件 | 回答的问题 |
|---|---|---|
| `00_context` | `round_manifest.json`、`scenario.yaml` | 这轮是什么场景和身份？ |
| `01_query` | `glm5.2_search_space.yaml`、`query_command.txt` | Agent 看到了哪些知识？ |
| `02_parameters` | `candidate_params.yaml`、`candidate.env`、`effective_config.yaml`、`vllm_common_command.txt` | 实际改了什么、怎样注入？ |
| `03_submission` | `submission.json`、`submit_output.txt`、`benchmark_preflight.log` | 提交到哪个 Lease/run-id，预检是否通过？ |
| `04_runtime` | 见下一节 | 服务和 Benchmark 运行发生了什么？ |
| `05_results` | `metrics.json`、`comparison.json` 或 `failure.yaml` | 结果是否有效、相对锚点怎样？ |
| `06_agent_analysis` | `evidence_bundle.json`、`agent_events.jsonl`、`decision.json`、`next_candidate.yaml` | Agent 为什么作出该决策？ |

## 4. 在线日志：本地与服务器对应关系

### Controller 自身

```text
tuning_pipeline/workflow/continuous/logs/controller/
├─ controller.log
└─ process/
   ├─ controller_process_<time>.stdout.log
   └─ controller_process_<time>.stderr.log
```

- `controller.log`：状态机事件，先看它判断停在哪一轮。
- `stdout.log`：Controller 前台输出。
- `stderr.log`：Python、SSH、依赖和未捕获异常。
- `state.json`：不是日志，但它是当前 Session/round/run-id 的入口。

### 每轮本地核心归档

`round_*/04_runtime/` 包含：

| 文件 | 内容 |
|---|---|
| `run_status.json` | 当前/最终阶段和 outcome |
| `startup_timeline.jsonl` | vLLM 进程启动与 API Ready 时间 |
| `master.log` | DP0/TP16 服务、权重加载、编译、通信和 API 日志 |
| `worker.log` | DP1/TP16 Worker、权重加载、HCCL 和运行异常 |
| `models_response.json` | Ready 后 `/v1/models` 响应 |
| `benchmark_runner.log` | Aligned-L1 总控日志 |
| `benchmark_watchdog.log` | 本地 Controller 中断时的远端守护/恢复日志 |
| `warmup.log`、`formal.log` | Legacy Benchmark 路径日志 |
| `SERVICE_READY`、`BENCHMARK_*`、`MASTER_DONE` | 状态标记，不是空产物 |

本地不复制完整 ServeBench 逐 Case 文件树；它可能包含数百个文件和很深的路径。
Controller 只下载闭环决策所需的核心日志、状态标记和汇总 `metrics.json`。

### 服务器原始位置

```text
<remote-project>/workflow/auto/
├─ runs/<run-id>/
│  ├─ master.log
│  ├─ worker.log
│  ├─ server_run_manifest.yaml
│  ├─ startup_timeline.jsonl
│  ├─ benchmark_runner.log
│  ├─ benchmark_watchdog.log
│  ├─ run_status.json
│  ├─ metrics.json
│  └─ servebench/
└─ lab_runs/<run-id>/service/
   ├─ rank-000.log
   └─ rank-001.log
```

`runs/` 是项目脚本写出的权威原始产物；`lab_runs/` 是 ktp-lab 对两个节点进程的外层捕获。两者都保留。每个新 run 的 `server_run_manifest.yaml` 是该目录的自描述索引，服务器总体布局见 `workflow/auto/ARTIFACT_LAYOUT.md`。Controller 只把核心日志和汇总结果复制到本地，不复制完整 `servebench/`；服务器原路径不改、不删。

## 5. 快速查看

```powershell
# 当前状态和 Controller 尾日志
.\scripts\status.ps1

# 某轮服务日志
Get-Content .\tuning_pipeline\workflow\continuous\experiments\<session>\round_<n>\04_runtime\master.log -Tail 100

# 某轮结果和 Agent 决策
Get-Content .\tuning_pipeline\workflow\continuous\experiments\<session>\round_<n>\05_results\metrics.json
Get-Content .\tuning_pipeline\workflow\continuous\experiments\<session>\round_<n>\06_agent_analysis\decision.json
```

远端只读检查使用 `docs/OPERATIONS.md` 中的命令，并始终限定在获准的 `cjx-workspace`。

## 6. 权威性顺序

1. 某轮实际命令：`02_parameters/vllm_common_command.txt`。
2. 某轮实际配置：`02_parameters/effective_config.yaml`。
3. 原始运行事实：服务器 `runs/<run-id>/`；本地 `04_runtime/` 是核心日志副本。
4. 正式指标：`05_results/metrics.json`；没有该文件就不是有效性能结果。
5. Agent 决策：`06_agent_analysis/decision.json`。
6. 全局默认配置只用于新 Session，不能覆盖既有 `session_config.yaml`。
