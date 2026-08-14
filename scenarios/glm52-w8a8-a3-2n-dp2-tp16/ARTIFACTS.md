# W8A8 固定定义与产物

## 固定定义（Git 跟踪）

- 场景总索引：[`scenario.yaml`](scenario.yaml)
- B0：`tuning_pipeline/workflow/baselines/b0_deployable_64k.yaml`
- Search-Space Scenario：`scenario.glm52-a3-aligned-l1.yaml`
- Runtime Profile：`runtime_profiles.yaml` 中的 `glm52_w8a8_a3_dp2_tp16`
- Topology：`a3_dp2_tp16`
- Executor：`ktp_two_role`
- Benchmark：`aligned_l1_v4`
- Strategy：`hierarchical_throughput_v1`
- 镜像身份：`remote/image_version_manifest.yaml` + `activation.approved.yaml`

共享参数画像位于 `portrait_pipeline/outputs/ParameterYAML/`，共享 Tags 位于
`tuning_pipeline/tag_params/output/params/`。它们会经过本场景的 Scenario 和 Compiler
筛选后，冻结为 Session 内的 `00_search_space/parameter_portraits.*.yaml`。

## 运行产物（Git 忽略）

```text
tuning_pipeline/workflow/continuous/scenario_runs/glm52-w8a8-a3-2n-dp2-tp16/
```

正式 Session 中最重要的场景产物是：

- `session_config.yaml`
- `00_search_space/search_space.compiled.yaml`
- `00_search_space/classified_search_limits.yaml`
- `00_search_space/parameter_portraits.agent.yaml`
- `round_*/02_parameters/`
- `round_*/05_results/`
- `round_*/06_agent_analysis/`

历史上直接使用旧入口生成的 Session 仍可能位于
`tuning_pipeline/workflow/continuous/experiments/`；这些是兼容历史，不再作为新入口。
