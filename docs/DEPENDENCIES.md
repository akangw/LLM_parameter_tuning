# 外部依赖

## 经批准保留：liuxin-workspace

Aligned-L1 运行阶段只读挂载：

```text
/mnt/host-model/slai/user-1-wangakang/wangakang/liuxin-workspace
```

用途仅限：

- 读取 `tools/ktp-lab/runtime/leases/<lease>/control/service.json`；
- 挂载 `/tools` 给 Benchmark 容器；
- 使用 `/tools/runtime/activations/nightly-main-a3/guidellm.sh` 激活 GuideLLM 0.7.2。

项目自己的 ServeBench、Suite、Schema 和 Dataset 均位于独立 `cjx-workspace/vllmtkb-...` 目录。远端脚本将 `liuxin-workspace` 以 `:ro` 挂载，不对其写入。

若该目录移动或权限变化，需要同步修改 `benchmark.aligned_l1.servebench_workspace` 和 `guidellm_activation`，重新执行前置检查并创建新 Session。不得在旧 Session 中静默替换测量依赖。

## 固定身份

- 服务镜像：`sha256:37f43036c21c80c9cc6c6656f472df0f1b79d2fce2027168d270a43d725e0305`
- Benchmark 容器：`sha256:46d8bbb49f90f0607a41f972b7d847b434e0cf0f86fce9164549a7fdf0033112`
- ServeBench：1.0.1，commit `5f01dcafd9b4ce9c788d8f7c1a537d4b3ea83c73`
- GuideLLM：0.7.2

Benchmark 容器使用 Digest-qualified 引用，并进入测量指纹。
