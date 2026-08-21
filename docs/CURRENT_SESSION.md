# 当前实验状态入口

Git 不保存易过期的轮次状态和性能数字。当前生产 Session 使用
`decode_priority_v2.sh` 和独立的 `runtime_decode_priority_v2_live/`；真实状态只从
服务器受管状态读取：

```bash
cd /mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-decode-priority-v1
AUTO=tuning_pipeline/workflow/continuous/server_autonomous
bash "$AUTO/decode_priority_v2.sh" service supervisor-status
bash "$AUTO/decode_priority_v2.sh" status
```

当前 V2 的第 0 轮会有意重新测量 A10F1，建立本 Session 的可比基线；这不是
重复搜索。基线成功后才进入 List 2。V2 冻结了 A1–A15 的兼容历史，Agent 可见
旧候选、指标和归因失败，Controller 也会拒绝再次提交已经覆盖的完整候选。

以下重复属于正常行为：新 Session 基线复测、潜在新 best 的确认测量、没有产生
有效指标的基础设施故障重试。参数归因失败和已有有效测量不会被当作新探索重复。

不要把运行中的 `state.json`、Lease、API Key 或完整性能结果提交到 Git。交接当前
实验时应保留服务器运行目录，并用上述命令查看；需要跨服务器迁移时再使用受控的
Session 导出/导入能力。
