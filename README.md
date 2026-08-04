# Auto vLLM Parameter

面向 GLM-5.2、Atlas A3 和 vLLM Ascend 的参数知识构建与连续自动调优项目。项目从 `vllmTKB0706` 的在线闭环迁移而来，但使用独立源码版本、知识产物、Controller 状态、远端目录和 ktp-lab Lease。

## 当前状态

- 固定源码：vLLM `418bd6273c03bf48d5066733769e0a74bdc51694`，vllm-ascend `32c8cf190f596b47f0d0b965e64aea9f2b789ad4`。
- 参数知识：1540 个结构化表面，340 份完整 ParameterYAML，105 份带依据跳过记录。
- 五维标签：340 份，审计错误 0；当前场景召回 109 份高影响画像。
- 新 Session 搜索空间：16 个合格可调参数，12 Active、4 Reserve、6 Fixed、1 Rejected。
- 在线闭环：已完成真实远端提交、服务启动、完整 Aligned-L1、结果回收、Agent 选参和 OOM 隔离。
- 当前正式锚点：A0，主分数 `602.5576 output tok/s`；尚未产生通过全部延迟门禁的新赢家。

## 快速入口

```powershell
# 只做本地前置检查
.\一键启动.ps1 -CheckOnly

# 自动判断新建或续跑
.\一键启动.ps1

# 查看状态
.\scripts\status.ps1

# 优雅停止：归档当前轮次，不提交下一轮
.\scripts\stop.ps1
```

首次准备远端独立 Lease：

```powershell
.\scripts\prepare-remote.ps1
```

上游源码不提交到本仓库，可按固定提交恢复：

```powershell
.\scripts\fetch-sources.ps1
```

## 项目层级

```text
Auto_vllm_parameter/
├─ README.md                     项目总入口
├─ docs/                         架构、运行、产物和交接文档
├─ scripts/                      面向操作者的稳定入口
├─ portrait_pipeline/            离线参数画像构建
│  ├─ build/                     提取、初筛、迁移和画像程序/证据
│  ├─ outputs/ParameterYAML/     340 份正式参数画像
│  ├─ outputs/skipped/           105 份跳过证据
│  └─ sources/                   固定提交源码，本地生成且不入库
└─ tuning_pipeline/              标签、搜索空间与在线调优
   ├─ tag_params/output/params/  340 份五维标签成品
   ├─ search_limits/             最新独立编译产物
   └─ workflow/
      ├─ search_space_compiler/  Search Limits 编译器
      ├─ sidecars/               画像检索和运行规则
      └─ continuous/             Controller、远端脚本和 Session 运行时
```

## 经批准保留的外部依赖

Benchmark 运行阶段仍以只读方式挂载：

```text
/mnt/host-model/slai/user-1-wangakang/wangakang/liuxin-workspace
```

该目录提供 ktp-lab Lease 控制文件和 GuideLLM 激活脚本。项目不会修改它；依赖范围、替换条件和风险见 [依赖说明](docs/DEPENDENCIES.md)。除该已声明依赖外，新项目不读取或修改 `vllmTKB0706` 的代码、状态、Lease 或实验目录。

## 文档

- [架构与数据流](docs/ARCHITECTURE.md)
- [运行与恢复](docs/OPERATIONS.md)
- [产物目录](docs/ARTIFACTS.md)
- [交接清单](docs/HANDOFF.md)
- [外部依赖](docs/DEPENDENCIES.md)
- [当前实验摘要](docs/CURRENT_SESSION.md)

## 安全边界

- 本地保存知识、决策、状态和实验归档；远端只执行服务与 Benchmark。
- 远端项目目录固定为 `/mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-418bd627-32c8cf190`。
- 服务镜像和 Benchmark 容器均使用 Digest 固定身份。
- 上游 `--enable-eplb` 在当前 Ascend 版本中禁止进入搜索；Native Dynamic EPLB 接线完成前保持 `false/0`。
- 失败或残缺结果不会进入性能比较。
