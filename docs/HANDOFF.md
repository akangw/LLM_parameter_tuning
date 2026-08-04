# 项目交接清单

## 1. 接手者先读

1. 根目录 `README.md`。
2. `docs/ARCHITECTURE.md`。
3. `docs/DEPENDENCIES.md`。
4. `docs/CURRENT_SESSION.md`。
5. `docs/OPERATIONS.md`。

## 2. 固定身份

交接前后必须一致：

```text
vLLM commit       418bd6273c03bf48d5066733769e0a74bdc51694
vllm-ascend       32c8cf190f596b47f0d0b965e64aea9f2b789ad4
服务镜像 digest   37f43036c21c80c9cc6c6656f472df0f1b79d2fce2027168d270a43d725e0305
Benchmark digest 46d8bbb49f90f0607a41f972b7d847b434e0cf0f86fce9164549a7fdf0033112
Lease             vllmtkb-418bd627-32c8cf190-glm52-a3-32npu
```

## 3. Git 不携带的内容

- 两个上游源码工作树：使用 `scripts/fetch-sources.ps1` 恢复。
- 当前 `state.json`、Session 实验目录、Controller 日志和 PID。
- Codex 画像生成时的大体积任务上下文和 worker 日志。

如果接手者需要续跑当前 Session，原操作者应通过受控渠道额外交付：

```text
tuning_pipeline/workflow/continuous/state.json
tuning_pipeline/workflow/continuous/experiments/glm52_continuous_20260804_112130/
```

## 4. 新机器验证

```powershell
git clone https://github.com/chenasir/Auto_vllm_parameter.git
cd Auto_vllm_parameter
.\scripts\fetch-sources.ps1
python -m pip install -r .\tuning_pipeline\requirements-runtime.txt
.\一键启动.ps1 -CheckOnly
```

然后确认：

- SSH 别名 `hetao-npu` 指向正确服务器；
- `activation.approved.yaml` 与镜像 Digest 一致；
- 远端 Lease 名称和目录没有与 0706 共用；
- `liuxin-workspace` 只读依赖仍可访问；
- 没有另一台电脑正在运行同一个新项目 Controller。

## 5. 安全规则

- 不在 `/mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-...` 之外写入。
- 不删除远端实验记录。
- 不把缺少 `metrics.json` 的轮次计为性能结果。
- 不在旧 Session 中静默改变镜像、数据集或 Benchmark 依赖。
- 不启用当前 Ascend 不支持的上游 `--enable-eplb`。
