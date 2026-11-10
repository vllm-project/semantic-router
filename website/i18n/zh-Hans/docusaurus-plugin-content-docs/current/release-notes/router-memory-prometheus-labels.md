---
translation:
  source_commit: "9649d02f2582471642196bde2e086c34ccb9c88c"
  source_file: "docs/release-notes/router-memory-prometheus-labels.md"
  outdated: false
is_mtpe: true
---

# 发布说明：Router Memory Prometheus 标签集

升级到含 [#4326](https://github.com/vllm-project/semantic-router/issues/4326) 的版本，会改两个 Router Memory 指标的已发布标签集。发布前后，引用了旧标签的仪表盘、录制规则和告警都要更新。

## `llm_memory_retrieval_total`

| | 标签集 |
|---|-----------|
| **改前** | `backend`、`status`、`user_id` |
| **改后** | `backend`、`status` |

`status` 照旧区分检索结果，比如 Milvus、Valkey 和 Qdrant 后端上的 `hit`、`miss`、`error`。计数器不再给每个认证用户建一条时间序列。

迁移示例：

```promql
# 改前（按用户——不再有效）
sum by (backend, status, user_id) (llm_memory_retrieval_total)

# 改后（按后端和结果聚合）
sum by (backend, status) (llm_memory_retrieval_total)
```

## `llm_memory_store_size`

| | 标签集 |
|---|-----------|
| **改前** | `backend`、`user_id` |
| **改后** | `backend` |

早先的版本里，这个 gauge 没接真实存储计数；`user_id` 维度在生产路径上根本没接线。移除它，是将来的接线不会重新引入无界的身份基数。

## Qdrant 检索遥测

早先的版本，Qdrant `Retrieve` 不给 `llm_memory_retrieval_total` 计数，只记一条零时长的通用存储成功。这次改动后，Qdrant 检索发出和 Milvus、Valkey 一样的 hit/miss/error 检索计数和时长。

## 按用户分析

Router Memory 在存储和请求路径上仍按用户圈定检索和存储。按用户排查归日志、追踪和管理 API，不归这些计数器上的 Prometheus 标签。
