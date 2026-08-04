# 当前版本参数提取产物

由 `python -m migration_pipeline` 写入，不手工编辑。

- `parameters.json`：与师兄项目兼容的原始参数列表；
- `parameters.structured.json`：cjx_space 的权威结构化提取工件，供 Stage-1 去重与审计。
- `provenance.json`：目标快照、源码目录和精确 commit。

此目录是本次迁移唯一保留的参数提取产物；项目根目录的旧文件已在隔离链路验证后清理。
