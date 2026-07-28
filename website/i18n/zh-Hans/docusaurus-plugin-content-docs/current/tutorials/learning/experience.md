---
translation:
  source_commit: "ad233487"
  source_file: "docs/tutorials/learning/experience.md"
  outdated: false
---

# 经验（Experience）

## 概览

经验是 adaptation 使用的在线 Router Learning 证据。它在请求路径的进程内维护，并根据有界的结果和遥测数据进行汇总。离线配方学习可以导出 seed-pack 工件，用于冷启动分析和未来的预热流程，但首个公开 API 不提供运行时 seed-pack 导入开关。

## 主要优势

- 避免在请求路径中进行高开销的聚合。
- 将结果和遥测数据转化为紧凑的读取时证据。
- 为 adaptation 提供类型明确的位置，用于存放学习到的模型选择事实。
- 保持 Router Replay 作为事件日志的事实来源。

## 解决什么问题？

有些路由证据依赖历史请求，例如模型适配度、过度使用、提供商故障、延迟、缓存复用和实际成本。在处理请求时读取所有事件会过于缓慢。路由器会在内存中更新紧凑的模型经验，并将持久证据写入 Router Replay，供离线分析使用。

## 何时使用

- Adaptation 需要来自结果和运行时遥测的聚合证据。
- 运维人员希望通过离线评估解释冷启动时的模型质量证据。
- 离线配方学习需要 seed-pack 工件，用于实验或未来的预热流程。

## 配置

当前公开 API 不提供 `experience.enabled`、`experience.source` 或运行时 seed-pack 导入字段。启用 adaptation 后，模型经验将成为 `routing_sampling` 实现的一部分。

在线经验以匹配的 decision、decision tier 和模型为键：

```text
decision_id + decision_tier + model
  -> decision_tier + model
  -> model
```

结果摄取会更新面向模型的经验。路由、策略、稳定性、提供商和路由器结果会进入 replay 诊断和离线配方学习，而不会直接修改模型质量。
