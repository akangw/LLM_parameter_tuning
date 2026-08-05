# 独立编译参考快照

本目录是 `curated_registry_v1` 人工注册表路径的独立审计快照，当前为 Active 12、Reserve 4、Fixed 6、Rejected 1。它用于复现和比较，不是新 Session 的在线执行边界。

新 Session 默认使用 `automatic_registry_v1`，在创建时从正式画像与 Tags 重新编译，并把真正执行的注册表、Search Limits、兼容策略和注入契约冻结到：

```text
tuning_pipeline/workflow/continuous/experiments/<session>/00_search_space/
```

续跑时只读取该 Session 的冻结目录；不要把本参考快照复制到运行配置中。
