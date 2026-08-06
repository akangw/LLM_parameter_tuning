# W4A8C8 固定定义与产物

## 固定定义（Git 跟踪）

- 场景总索引：[`scenario.yaml`](scenario.yaml)
- A0：`tuning_pipeline/workflow/baselines/a0_glm52_w4a8c8_existing_tuned.yaml`
- Runtime Adapter：`glm52_w4a8c8_a3_single_dp2_tp8.yaml`
- Search-Space Scenario：`scenario.glm52-w4a8c8-a3-single-aligned-l1.yaml`
- Topology：`a3_single_16npu_dp2local_tp8`
- Executor：`ktp_single_node_local_dp`
- Benchmark：`aligned_l1_glm52_w4a8c8_v1`
- Strategy：`best_anchor_coverage_v2`

共享 ParameterYAML/Tags 会按 W4A8C8 的模型、量化、单节点拓扑和镜像能力重新筛选。
它不会读取 W8A8 的候选历史、最佳 Anchor 或指标。

## 当前产物状态

该场景仍为 `planned`，所以目前只有固定定义，没有可信的正式 Session Search Limits
快照和 Benchmark 结果。完成真实 A0、Benchmark、Executor、Search Limits 四项验证后，
运行产物才会写入：

```text
.runtime/scenarios/glm52-w4a8c8-a3-1n-dp2-tp8/
```

不要把 W8A8 的 `00_search_space/` 或历史 Session 复制过来充当 W4A8C8 产物。
