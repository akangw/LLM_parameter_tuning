# 自动调优知识与控制面

本目录承接 `portrait_pipeline/outputs/ParameterYAML`，形成可审计的参数知识库、搜索空间和连续调优控制器。它借鉴 `vllmTKB0706` 的闭环框架，但运行路径、配置、状态和产物均归属于当前项目，不依赖旧项目目录。

```text
ParameterYAML
  -> 五维标签与审计
  -> Search Limits 编译与审批门
  -> 参数画像、运行规则与候选校验
  -> 远端单轮实验
  -> 指标评估、接受/回滚与历史沉淀
```

## 当前稳定产物

- 参数画像 YAML：340 份。
- Tagged YAML：`tag_params/output/params/`，340 份，标签审计错误为 0。
- 当前场景高影响召回：109 份。
- 人工路径参考快照：`search_limits/`，Active 12、Reserve 4、Fixed 6、Rejected 1；这是新 Session 的默认构建口径，实际权威结果仍冻结在该 Session 的 `00_search_space/`。
- 自动注册表是可插拔替代选项：28 Tunable（Active 12、Reserve 16）、40 Fixed、0 Compiler Rejected。
- 当前固定镜像的上游 CLI 不支持动态 EPLB，因此本场景固定 `enable_eplb=false`、`eplb_num_redundant_experts=0`。
- Controller：`workflow/continuous/continuous_tuning.py`。

`reserve`、`fixed` 和 `rejected` 只作为审计分类；只有 Active 参数会进入 Agent 的可变搜索边界。Controller 再合并必要的单值运行契约，且会拒绝未批准或无法注入的参数。

## 主要入口

```powershell
# 从项目根目录执行
.\scripts\start.ps1 -CheckOnly
.\scripts\status.ps1
.\scripts\start.ps1
.\scripts\stop.ps1
```

知识库离线检查：

```powershell
cd .\tuning_pipeline
python query.py --count
python query.py --list-tags
python -m tag_params.audit
python -m unittest workflow.search_space_compiler.test_compiler workflow.sidecars.test_sidecars workflow.continuous.test_continuous_tuning workflow.continuous.test_aligned_l1_metrics workflow.continuous.test_hierarchical_strategy
```

整体架构、依赖边界、产物和交接方式以项目根目录的 `README.md` 与 `docs/` 为准。

## 可移植 Benchmark

- `aligned_l1_v4`：现有内部 ServeBench/GuideLLM 严格矩阵。
- `vllm_bench_public_v1`：无需内部 Benchmark 权限，只依赖服务镜像中的公开 `vllm bench serve`。
- `custom_adapter_v1`：运行项目白名单目录中的自定义 Python 适配器，协议见 `workflow/benchmark_adapters/README.md`。
- `legacy_random_32k1k`：仅用于复现历史探索口径。

用 `-BenchmarkProfile vllm_bench_public_v1` 在新 Session 中选择公开路径。每个 Profile 的完整定义与 SHA-256 身份都会冻结到 Session 和 `metrics.json`，因此不同 Benchmark 或不同配置不会混合比较。
