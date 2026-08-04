# 本地日志根目录

本目录只存放机器本地运行日志，日志文件默认不进入 Git。

```text
logs/
├─ offline/                         可选的离线总控启动日志
└─ controller/
   ├─ controller.log               Controller 状态机主日志
   └─ process/
      ├─ controller_process_*.stdout.log
      └─ controller_process_*.stderr.log
```

Tags 阶段的日志跟随对应产物，位于
`tag_params/output/logs/{parameters,pipeline}/`；在线每轮日志跟随 Session，位于
`workflow/continuous/experiments/<session>/round_*/04_runtime/`。
