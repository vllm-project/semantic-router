---
translation:
  source_commit: "0f2ba0de7c435366ed68bcf03f5a1bb49b9cb90c"
  source_file: "docs/tutorials/plugin/memory.md"
  outdated: false
---

# 记忆

## 概览

`memory` 是一个路由局部插件，用于检索和存储对话记忆。

## 主要优势

- 将记忆行为限制在受益于它的路由内。
- 在一个插件中支持检索和自动存储。
- 将路由局部记忆策略与共享后台存储配置分开。

## 解决什么问题？

并非每条路由都应承担检索记忆的复杂度或隐私成本。`memory` 让一条已匹配路由检索并存储对话上下文，同时共享存储仍配置在 `global.stores.memory` 下。感知会话的模型稳定性是单独的路由学习自适应，配置在 `global.router.learning` 下。

## 何时使用

- 某条路由应检索先前对话上下文
- 该路由应自动存储有用的新轮次
- 记忆设置应保持在一个路由家族内

## 配置

记忆插件需要在 `global.stores.memory` 下配置后台存储。路由器支持三种后端：

- **Milvus**（默认）— 分布式向量数据库，最适合大规模生产
- **Valkey** — 使用 Search 模块的轻量单二进制选项，最适合开发/测试或已有 Valkey 基础设施
- **Qdrant** — 带 gRPC 的单二进制，运维比 Milvus 更简单，适合小到大规模工作负载

全局 memory 配置见[存储与工具](../global/stores-and-tools)教程，Valkey 专用设置见 [Valkey 记忆部署指南](../../installation/valkey-memory)，Qdrant 专用设置见 [Qdrant 部署指南](../../installation/qdrant)。

在 `routing.decisions[].plugins` 下添加该插件：

```yaml
plugins:
  - type: memory
    configuration:
      enabled: true
      retrieval_limit: 5
      similarity_threshold: 0.72
      auto_store: true
```

记忆可以持久化从请求派生的内容，并将检索到的记忆发送给所选模型。请为这些数据选择合适的用户/租户隔离、保留策略、认证和传输安全。阈值取决于嵌入模型。完整示例见：
[`config/fragments/plugin/memory/session-memory.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/memory/session-memory.yaml)。

## 升级嵌入模型

修改嵌入权重后，重启模型运行时。对于本地 `mmbert` 模型（包括 Vela Embedding），Router 将记忆与实际加载的模型、分词器、推理设置及向量维度绑定。变更这些设置会创建独立的物理集合或索引，以及独立的 Redis 热缓存；使用相同向量表示重启则复用已有存储。配置中的逻辑名称保持不变。

此前没有身份标记的集合会保留，但不会自动接管：相同向量维度不代表两个模型的嵌入兼容。管理 API 没有记忆导入或批量导出端点，因此新模型的集合从空开始，并随新流量重新积累记忆。启动或迁移不会删除旧集合。这项自动身份绑定目前覆盖本地 `mmbert`，其他嵌入提供方保留已有行为。
