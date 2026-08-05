# 独立编译参考快照

本目录是默认 `curated_registry_v1` 人工注册表路径的独立审计快照，当前为 Active 12、Reserve 4、Fixed 6、Rejected 1。它用于审阅和复现，但不是新 Session 的在线执行边界。

新 Session 默认从人工审计的 23 项注册表重新编译，并把真正执行的 Search Limits、规则和注入契约冻结到：

```text
tuning_pipeline/workflow/continuous/experiments/<session>/00_search_space/
```

续跑时只读取该 Session 的冻结目录；不要把本参考快照复制到运行配置中。若新建 Session 时显式选择 `automatic_registry_v1`，同一目录还会保存现场生成的注册表及其审计证据。
