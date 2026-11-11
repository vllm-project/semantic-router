---
title: 添加偏差
description: 如何声明一个经过测试、有文档的 System One 后端偏离其 schema 的方式。
is_mtpe: false
translation:
  source_commit: "6b782822edead98397acc6401112e6ec1e45af96"
  source_file: "docs/community/system-one-schemas/adding-a-deviation.md"
  outdated: false
---

# 添加偏差 {#adding-a-deviation}

偏差（deviation）是后端偏离其所声称 schema 的一种具名、已声明的方式，例如用于只接受字符串 state 的后端的 `state_text`。只为你针对该后端测试过的真实不一致添加偏差。

1. **编写改写器**，放在 `src/semantic-router/pkg/systemonecodec` 中：一个无状态类型，实现 `ID`（小写下划线）、`AppliesTo`（适用的 schema）、`RewriteRequest`（修改一次已规划的调用，并以诊断信息返回所做的改动）和 `RewriteReply`。偏差在 codec 的计划之后、按部署列出的顺序运行。
2. **注册** 到 `Builtin()`；注册表拒绝不适用于任何已注册 schema 的偏差，配置拒绝不适用于部署 schema 的偏差。
3. **单独测试：** 逐字节检查改写后的请求、它保持不变的调用，以及当它改变模型读取的内容时，各模型族对改写形式的读取结果相同。
4. **文档：** 在 [System One schema](../../system-one/use/schemas.md#deviations) 的偏差表中加一行：schema、作用以及哪些后端需要它。
