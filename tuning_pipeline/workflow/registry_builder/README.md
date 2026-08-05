# Tags 召回到 Search Limits 的自动化链路

这个模块将原来“109 份 Tags 召回画像 → 人工维护 23 项注册表”的中间环节自动化，并将自动注册表交给现有 Search-Space Compiler 生成 Search Limits。

它既可以作为 Controller 的 `automatic_registry_v1` Profile 运行，也可以作为**独立、可审计、不提交任务**的命令运行：

- 不读取、修改或覆盖现有 `workflow/search_space_compiler/registry.yaml`；
- 独立命令不修改 `workflow/continuous/config.yaml` 或任何 Session；
- 不连接服务器，不提交试验；
- 所有产物只写入隔离的审计目录；
- 显式禁止将产物写入 `workflow/continuous/`。

## 完整处理链路

```text
Tags 场景精确召回
  → 标准参数名与候选值提取
  → CLI / ENV / nested 语义合并与废弃别名处理
  → 固定版本 vLLM / vLLM-Ascend 源码能力核验
  → Compatibility Validator
     ├─ 模型 / 硬件 / Benchmark 功能门禁
     ├─ 占位符和 unset / omit 动作规范化
     ├─ 候选值基线边界和场景枚举过滤
     ├─ 跨参数依赖与互斥规则
     └─ CLI / ENV / JSON path 注入渲染校验
  → registry.generated.yaml
  → 现有 Search-Space Compiler
  → Active / Reserve / Fixed / Rejected Search Limits
```

自动链路采用失败关闭策略：只有候选值充足、源码能力可证明、场景功能门禁通过，且所有候选值都能被通用注入协议渲染的参数，才会进入自动注册表。该过程只使用 Python 程序与 YAML 规则，不调用 AI。

## 使用

在线闭环在新建 Session 时选择并冻结：

```powershell
.\一键启动.ps1 -NewSession -SearchSpaceProfile automatic_registry_v1
```

人工 23 项注册表路径可通过 `curated_registry_v1` 切回。Resume/Retry 始终读取 Session 内冻结的 Profile，禁止中途漂移。

在 `tuning_pipeline` 目录下先做不落盘的端到端检查：

```powershell
python -m workflow.registry_builder.full_pipeline --dry-run
```

生成完整隔离产物：

```powershell
python -m workflow.registry_builder.full_pipeline `
  --output workflow/registry_builder/runs/full-review-001
```

换画像版本或场景时：

```powershell
python -m workflow.registry_builder.full_pipeline `
  --knowledge-dir <新的Tags画像目录> `
  --scenario <新的scenario.yaml> `
  --policy workflow/search_space_compiler/policy.yaml `
  --output <隔离审计目录>
```

换场景时可同时传入另一份确定性兼容策略：

```powershell
python -m workflow.registry_builder.full_pipeline `
  --scenario <scenario.yaml> `
  --compatibility-policy <compatibility_policy.yaml> `
  --output <隔离审计目录>
```

如果只想检查“召回 → 注册表提案”，而不调用 Compiler：

```powershell
python -m workflow.registry_builder --dry-run
```

## 产物

- `registry.generated.yaml`：自动生成的参数注册表，包含候选值、风险、注入协议与源码证据。
- `registry.audit.json`：109 召回到自动注册的数量与拒绝原因。
- `compatibility_constraints.yaml`：Controller 提交候选前必须再次校验的跨参数依赖与互斥规则。
- `search_limits/agent_search_limits.yaml`：Compiler 选出的 Active Search Limits。
- `search_limits/search_space.compiled.yaml`：Active、Reserve、Fixed、Rejected 全量结果。
- `search_limits/audit.json`、`approval_queue.yaml`、`rotation_report.yaml`：编译审计与轮换信息。
- `pipeline_manifest.json`：整条链路产物的 SHA-256 清单。

## 提交前的纯程序校验

Agent 候选必须在提交前通过同一套候选域、机器约束、跨参数约束和注入渲染校验。候选 YAML 例如：

```yaml
params:
  max_num_seqs: 32
  speculative_config__method: mtp
```

校验命令：

```powershell
python -m workflow.registry_builder.candidate_validator `
  --compiled workflow/registry_builder/runs/full-review-001/search_limits/search_space.compiled.yaml `
  --candidate <candidate.yaml> `
  --scenario workflow/search_space_compiler/scenario.glm52-a3-aligned-l1.yaml
```

返回码为 `0` 表示可渲染；返回码为 `2` 表示候选越界、违反组合约束或注入无法渲染，不得提交。

## 两种可替换路径

默认主链路为：`109 召回 → 自动 registry.generated.yaml → 同一个 Search-Space Compiler → Controller`。

兼容路径为：`109 召回 → 人工 registry.yaml（23 项）→ 同一个 Search-Space Compiler → Controller`。

自动路径不会读取人工 `registry.yaml`。当前最终生成 36 个可调参数（Active 12 + Reserve 24），并在 Session 内保存自动注册表、审计、兼容策略与注入契约。早期第一轮兼容输出的 44 项中，1 个禁用 EPLB 维度被排除，7 个由平台覆盖、拓扑限制或部署拥有的维度被降为 Fixed/Recovery，避免未来轮换成无效 Active。Controller 对已存在的同名参数复用成熟运行适配器，对自动路径新增的参数使用通用 CLI/ENV/JSON 注入协议；每次提交前再次执行候选域和组合约束校验。
