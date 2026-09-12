---
title: Router Runtime
description: 配置路由分类、安全检查和嵌入模型。
translation:
  source_commit: "dc7f402642a8b8ecec8218e2086a4c6f186ea406"
  source_file: "docs/installation/native-backends.md"
  outdated: false
---

Router Runtime 运行路由所需的分类、嵌入和安全检查模型。负责回答用户的 LLM 在[模型配置](/zh-Hans/docs/installation/model-configuration)中单独设置。

## 选择运行方式 {#choose-a-running-mode}

| | 进程内模型 | 外部服务 |
| --- | --- | --- |
| 模型在哪里运行 | Router 进程内 | 独立服务中 |
| 需要准备什么 | 模型文件和兼容的 CPU 或 GPU 运行环境 | API 地址和凭据 |
| 从这里开始 | [运行进程内模型](runtime/in-process.md) | [连接外部服务](runtime/external.md) |

需要在本地执行推理时，使用进程内模型。如果模型已在其他地方提供服务，或需要单独管理硬件，使用外部服务。同一个 Router 可以同时使用这两种方式。

## 按用途配置 {#configure-a-use-case}

- [嵌入模型](runtime/embeddings.md)：语义匹配、缓存和向量存储。
- [安全模型](runtime/safety.md)：提示词防护、PII 和幻觉检测。

启动失败、并发限制和配置重载的处理方法见[运维与故障排查](runtime/lifecycle-diagnostics.md)。
