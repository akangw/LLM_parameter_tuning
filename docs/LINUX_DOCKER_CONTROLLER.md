# Linux / Docker Controller 与迁移工具

这些入口是现有 Windows PowerShell 主链路之外的可选层。它们调用同一个
`continuous_tuning.py`，不复制策略逻辑，也不改变 B0、Search Limits、Agent 或
Benchmark 的语义。Linux 默认把可变状态写到仓库下的 `.runtime/controller`，因此
不会碰现有主链路的 `workflow/continuous/state.json`。

## 原生 Linux

安装 Python 3.11、Git、OpenSSH 和运行依赖，并准备未提交的
`config.local.yaml`：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r tuning_pipeline/requirements-runtime.txt
export DEEPSEEK_API_KEY='...'

# 只读预检
scripts/controller.sh check --agent-provider deepseek

# 新 Session；默认使用隔离的 .runtime/controller
scripts/controller.sh new \
  --agent-provider deepseek \
  --benchmark-profile vllm_bench_public_v1

scripts/controller.sh status
scripts/controller.sh resume
```

通过 `VLLMTKB_CONFIG` 和 `VLLMTKB_RUNTIME_ROOT` 可以显式指定个人配置与状态目录。
`prepare` 会创建 Lease，`new` 会提交实验；只有在完成 `check` 并确认没有共享任务
冲突后才使用它们。

## Docker

Compose 复用宿主机的仓库、只读 SSH 配置和独立运行目录：

```bash
docker compose -f docker-compose.controller.yml build
docker compose -f docker-compose.controller.yml run --rm controller check \
  --agent-provider deepseek
docker compose -f docker-compose.controller.yml run --rm controller new \
  --agent-provider deepseek --benchmark-profile vllm_bench_public_v1
```

API Key 由宿主环境注入，不能写入 Compose 或 YAML。若 SSH Agent、UID/GID 或企业
证书有额外要求，可在个人 Compose override 中添加，公共文件不绑定个人环境。

## 拓扑 Profile

`workflow/continuous/topology_profiles.yaml` 记录执行拓扑。当前唯一标记为
`integrated` 的 Profile 是 `a3_dp2_tp16`：两节点、每节点 16 NPU、DP2/TP16，使用
一个 master 和一个 worker 的 `ktp_two_role` 执行器。Controller 会把解析结果冻结
进 Session、渲染 Lease 资源并导出审计环境变量。

Profile 文件是能力声明，不是任意拓扑开关。尚未实现的新布局必须保持
`status: planned`，Controller 会拒绝启动，直到对应的 rank 分配、远端脚本和测试都
完成集成。

## Session 导出和导入

```bash
# 终态或暂停态 Session 的完整、带 SHA-256 清单的归档
scripts/export-session.sh \
  --runtime-root .runtime/controller \
  --output ./session-20260806.zip

python tuning_pipeline/workflow/continuous/session_bundle.py verify \
  ./session-20260806.zip

# 默认只导入 experiments，不激活 state.json
scripts/import-session.sh ./session-20260806.zip \
  --runtime-root /new/runtime/root

# 仅当目标目录没有 state.json 时显式激活
scripts/import-session.sh ./session-20260806.zip \
  --runtime-root /new/runtime/root --activate
```

导出活动 Session 默认失败；确实需要时间点快照时必须显式使用
`--allow-active-snapshot`。导入和导出都拒绝覆盖已有文件。

## 镜像身份命令

先由可信的、digest 限定的服务器探针产生 JSON，字段格式参见
`image_probe.example.json`。命令先核对画像 commit，再一次生成 manifest 和批准文件：

```bash
scripts/verify-image-identity.sh validate
scripts/verify-image-identity.sh approve \
  --probe-json /secure/path/image-probe.json \
  --approved-by operator --dry-run
scripts/verify-image-identity.sh approve \
  --probe-json /secure/path/image-probe.json \
  --approved-by operator
```

也可用 `--probe-command-file` 传入一个 JSON argv 数组。命令以 `shell=False` 执行，
stdout 必须是探针 JSON。探针 commit 与现有参数画像不一致时必须先跑版本迁移，工具
不会替操作者静默批准不匹配的知识产物。

## CI

GitHub Actions 在 Python 3.11 上执行：YAML/JSON 解析、跨文件配置校验、高置信密钥
扫描和完整 pytest。相同检查可在提交前运行：

```bash
python scripts/validate_repository.py
python -m pytest -q
```
