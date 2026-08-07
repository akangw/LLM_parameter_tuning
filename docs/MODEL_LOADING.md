# 模型加载与真正热启动

模型加载是独立于 Agent 参数、Search Limits 和 Benchmark 的部署层 Profile。默认流程没有
改变；新 Session 不显式选择 RFork 时，仍使用已经验证过的 DTFS 节点页缓存路线。

## 两种“热”的区别

| Profile | 数据路径 | 含义 |
|---|---|---|
| `dtfs_page_cache_v1` | DTFS → Linux page cache → 普通 vLLM loader → NPU | 减少重复存储读取，不是真正的 NPU 权重热启动 |
| `rfork_seed_v1` | 首个实例允许读存储并注册为 seed | 只用于在额外 Ascend 资源上建立 seed |
| `rfork_external_seed_v1` | 活着的兼容 seed → YuanRong TransferEngine → 新实例 NPU | 真正热启动；任何回退都会终止本轮 |

RFork 的匹配键包括模型身份、部署策略、节点 Rank、TP Rank 和 draft 角色。因此模型、镜像、
DP/TP 或节点拓扑变化后必须建立新的匹配 seed，不能沿用旧 seed。

## 资源前提

真正 RFork 同时需要：

1. 每个 RFork 实例的镜像安装 `openyuanrong-transfer-engine`；
2. 一个持续运行、所有计算节点可访问的 RFork planner；
3. 一套仍然存活且身份、DP/TP/节点拓扑完全兼容的 seed 实例；
4. 另一套 Ascend 资源启动待测 client。

当前 `2 节点 × 16 NPU` 全部被一个 DP2/TP16 实例占满，无法同时容纳 seed 和 client。
所以仅靠当前 32 张卡不能完成“上一轮 A0 留在 NPU、下一轮 A1 同时复制”的真正热启动；
需要额外一套兼容计算资源，或者由平台提供可跨实例保留的 seed 服务。这是容量约束，不是
prefetch 参数问题。

## 启用方式

先在 Controller/Supervisor 的环境文件中配置 planner 地址：

```bash
VLLMTKB_RFORK_SCHEDULER_URL=http://<planner-host>:1223
```

在新 Session 配置中选择 Profile（已冻结的 Session 不会被修改）：

```yaml
model_loading:
  profile: rfork_external_seed_v1
  profiles_file: workflow/continuous/model_loading_profiles.yaml
```

用于建立外部 seed 时将 Profile 改为 `rfork_seed_v1`。seed 首次可以回退到存储；用于实验的
client 必须使用 `rfork_external_seed_v1`。Profile 会注入 `--load-format rfork` 和经过冻结的
`--model-loader-extra-config`，同时关闭 DTFS 页缓存预读，避免混淆两条加载路线。

## 如何证明命中

client Profile 是失败关闭的：

- 任意主/从节点日志出现 `RFork transfer failed:`，本轮立即失败且不会运行 Benchmark；
- API ready 前，主节点和从节点都必须出现 `RFork worker initialized, load_format=rfork`；
- 验证通过后生成 `RFORK_TRANSFER_VERIFIED`；
- `effective_config.yaml` 和 `startup_timeline.jsonl` 记录 backend、load format 与强制命中开关。

因此，没有 `RFORK_TRANSFER_VERIFIED` 的启动不能称为真正热启动。加载耗时仍应与同镜像、
同拓扑的冷启动对照；RFork 只解决权重传输，不保证图编译、KV cache 初始化等阶段消失。

## A0 单一参数来源

W8A8 A0 的唯一参数文件是
`tuning_pipeline/workflow/baselines/a0_glm52_w8a8_existing_tuned.yaml`。Scenario 只保留
`baseline_definition` 引用，不再复制参数。Controller 会在新 Session 中用实际选中的 A0/B0
定义覆盖该引用，再进行 Tags 召回、Search Limits 裁剪与兼容性验证。
