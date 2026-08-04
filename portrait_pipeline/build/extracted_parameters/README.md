# 当前版本参数提取产物

由 `python -m migration_pipeline` 写入，不手工编辑。

- `parameters.json`：供本仓库本地分析器使用的窄兼容参数列表；
- `parameters.structured.json`：本仓库的权威结构化提取工件，供 Stage-1 去重与审计。
- `provenance.json`：目标快照、源码目录和精确 commit。

此目录是正式版本的参数提取产物；新版本试运行会写入各自隔离的
`build/version_migrations/<run-id>/01_extract/`，不会覆盖这里。
