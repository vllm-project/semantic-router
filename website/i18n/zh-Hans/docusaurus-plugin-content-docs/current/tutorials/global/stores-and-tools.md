---
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/tutorials/global/stores-and-tools.md"
  outdated: false
---

# 存储与工具

## 概览

本页介绍 `global:` 中的共享存储和工具块。

这些设置为路由局部插件和路由器级工具行为提供支撑。

## 主要优势

- 集中共享后台存储，而不是按路由重复配置。
- 让响应缓存、memory、检索和工具目录保持一致。
- 让路由局部插件保持小而专注。
- 明确共享基础设施依赖。

## 解决什么问题？

路由局部插件常常依赖共享存储或工具状态。若这些依赖在各路由中临时配置，系统会不一致且更难运维。

这些 `global:` 块通过一次定义共享后台服务来解决该问题。

## 何时使用

在以下情况使用这些块：

- 多条路由依赖同一响应缓存或 memory 后端
- 检索功能需要一个共享向量存储
- 路由器应暴露一个共享工具目录
- 后台存储配置属于整台路由器，而不是单条路由

## 配置

### 响应缓存 {#response-cache}

```yaml
global:
  stores:
    response_cache:
      backend_type: memory
  plugins:
    - type: response_cache
      configuration:
        enabled: true
        semantic:
          match:
            recall: {top_k: 1, min_similarity: 0.8}
```

store 只提供缓存后端。[`response_cache` 插件](../plugin/response-cache)负责开启缓存：写在 `global.plugins` 中对每个请求生效，写在决策的插件中只对该决策的请求生效。

#### 否定防护 {#negation-guard}

双编码器相似度无法区分 *“turn on dark mode”* 和 *“turn off dark mode”*：相反含义的查询常常超过召回的 `min_similarity`，而真正的改写反而低于它，因此提高阈值不能可靠地避免误命中。

所以，所有缓存后端都会在返回语义候选前执行词面防护：

- **能拦住的：**词面接近的英文问题中的否定线索和已知反义词替换。
- **需要的：**不需要模型，也没有任何设置。它在内存、Redis、Valkey、Milvus、Qdrant 和混合缓存中都会运行。
- **拦不住的：**没有词面线索的变化、仅词序的变化和非英文的变化。

拒绝会记录为 `cache_negation_reject`，计入未命中，被拒绝的分数仍作为 `x-vsr-trace` 的 `cache_similarity` 提供。远端和混合缓存会拒绝缺少原始问题的候选，并继续检查有数量上限的已检索候选。混合缓存回退到 Milvus 时同样执行该检查。

早期版本用 `polarity_guard` 配置它，内存后端还可选 NLI 层（`nli`、`lexical+nli`）。NLI 模型已退役，路由器会拒绝 `polarity_guard`；`vllm-sr config migrate` 会移除它。

### 记忆 {#memory}

记忆存储支持三种后端：`milvus`（默认）、`valkey` 和 `qdrant`。带 `memory` 源的 [`context` 插件](../plugin/context#memory-sources)负责读写它：写在 `global.plugins` 中对每个请求生效，写在决策的插件中只对该决策的请求生效：

```yaml
global:
  plugins:
    - type: context
      configuration:
        enabled: true
        sources:
          - type: memory
```

**Milvus 后端**（默认）：

```yaml
global:
  stores:
    memory:
      milvus:
        address: milvus:19530
        collection: agentic_memory
        dimension: 256
```

**Valkey 后端**（需要带 Search 模块的 Valkey）：

```yaml
global:
  stores:
    memory:
      backend: valkey
      valkey:
        host: valkey
        port: 6379
        dimension: 256
        collection_prefix: "mem:"
        index_name: mem_idx
        metric_type: COSINE
```

**Qdrant 后端**：

```yaml
global:
  stores:
    memory:
      backend: qdrant
      qdrant:
        host: qdrant
        port: 6334
        collection: agentic_memory
        dimension: 256
      default_retrieval_limit: 5
      default_similarity_threshold: 0.30
```

三个示例都使用 `embeddings.default` 指定的 Vela Embedding 生成嵌入，因此 `dimension` 取其提供的维度之一：64、128、256、512 或 768。

- **别的模型。**用别的模型建索引的库，要在 `embedding` 块里写明它的部署和维度；见[嵌入](../../model-runtime/guides/embeddings.md#stores-indexed-with-another-model)。
- **其他任何维度**会让 Router 记录 `Failed to create memory store: … Memory will be disabled`，并在没有记忆的情况下运行。

:::note
Qdrant 示例的 0.30 阈值基于原始余弦分数，只是起点，而不是校准值。用于你的数据之前，请先检查无关查询和已更正的事实。
:::

完整部署说明见：

- [Valkey 智能体记忆](../../installation/valkey-memory) — Docker、Kubernetes、配置参考、调优和排障
- [Qdrant](../../installation/qdrant) — Docker、Kubernetes、配置参考、调优和排障
- `config/runtime/memory/` 提供后端专用配置参考

`query_rewrite.deployment` 指定在记忆检索前结合对话改写查询的 `openai_chat` 部署；未配置时按原样检索查询。该部署的 `limits.max_response_bytes` 限制每次改写响应（默认 1 MiB）：

```yaml
global:
  model_catalog:
    deployments:
      rewriter:
        provider: openai_chat
        endpoint: http://llm.example.com:8000/v1
        model: qwen3
  stores:
    memory:
      query_rewrite: {deployment: rewriter}
```

#### 混合检索与反思 {#hybrid-search-and-reflection}

`hybrid_mode` 和 `reflection.algorithm` 可以在 `global.stores.memory` 下全局设置，也可以在决策 `context` 插件的记忆源中覆盖。配置其他任何值（包括 recipe 决策中设置的值）时，Router 将拒绝启动：

| 字段 | 可接受的值 | 默认值 |
|------|-----------|--------|
| `hybrid_mode` | `weighted`、`rrf`（完全匹配，小写） | `weighted` |
| `reflection.algorithm` | `heuristic`、`noop` | `heuristic` |

`hybrid_mode` 仅在 `hybrid_search: true` 时生效。使用 `hybrid_mode: rerank` 或 `algorithm: recency_semantic` 的旧配置将无法启动。请分别改为 `weighted` 和 `heuristic`，这两个值与旧值的实际运行行为相同。

#### 写入路径的上限 {#write-path-bounds}

自动持久化首先遵守服务端的限制：Memory 是否启用、保留策略，以及所选决策记忆源上显式的 `store: {auto: false}`。Responses 请求可以选择不写入，但不能推翻这些限制。策略允许持久化时，取值按从具体到一般的顺序：

1. Responses 请求的 `auto_store`；
2. 决策的值；
3. `global.stores.memory`。

只有省略的值才会回退到下一级。

响应处理不会等待持久化完成。身份检查和容量预留发生在有上限的历史快照之前，快照在响应路径仍持有会话状态时生成；协议编码和写入随后在后台运行。其他响应路径上的 Replay 操作仍是同步的。

配置 `global.stores.memory.persistence`：

| 字段 | 含义 | 默认值 |
| --- | --- | --- |
| `timeout_seconds` | 从预留到超时的秒数，包含准备、排队等待和写入；0–9,223,372,036 | 30 |
| `concurrency` | 工作槽数量，包含准备和写入；0–64 | 8 |
| `queue` | 等待工作槽的已预留尝试数；0–1024 | 64 |
| `shutdown_grace_seconds` | 重新加载或关闭时，在取消前排空写入的秒数；0–9,223,372,036 | 5 |

省略某个字段或设为 `0` 即使用默认值。负数和超过上述上限的值会在配置校验时被拒绝，早于工作槽或队列存储的分配。这也包括初始的 `config_source: kubernetes` 文档，早于控制器加载路由 CRD。

每次持久化尝试都有以下上限：

| 上限 | 值 |
| --- | --- |
| 载荷：请求历史、保留的 Responses 历史和当前助手响应，在剥离 think 标签之前计算 | 1 MiB，共享 |
| 历史条目 | 256 条消息或条目 |
| 嵌套 | 32 层内容 |
| 结构：历史与当前响应合计 | 4096 个节点 |

长度检查在预留持久化容量之前运行；文本拼装和历史复制只在准入之后运行。跳过某次尝试时：

| 结果 | 原因 |
| --- | --- |
| `skipped` / `history_too_large`，`fail_open=true` | 超出了某个上限。不占用容量，也不截断任何内容。 |
| `skipped` / `memory_info_unavailable`，`fail_open=true` | 缺少用户身份。 |
| `policy_blocked` | 越狱或幻觉策略拦截了该响应。 |

后台上下文只保留 span context 和 tracestate。

对于带有 Router Replay 记录的请求，已接受的尝试会为 `scheduled` 和一条终态回执预留容量，使两者都不受队列饱和影响：

- **回执仍可能丢失**，原因包括存储错误、关闭时排空超时或进程崩溃。
- **容量用尽时**，新的写入会以 `queue_full` 或 `receipt_queue_full` 被拒绝；已退役的池使用 `shutting_down`。这些情况都保持 fail-open，并按请求 ID 记录日志。
- **监控** `llm_plugin_execution_total{plugin_type="memory_persistence", status="rejected"}`。

超时和取消即使发生在排队期间，也只报告一个终态结果；已取消的任务不会启动。原生嵌入调用无法中断，因此正在运行的工作会一直占用其工作槽和资源，直到退出。取消不会撤销后端已经接受的写入。

### 向量存储 {#vector-store}

```yaml
global:
  stores:
    vector_store:
      enabled: true
      backend_type: milvus
      metadata_store: postgres
```

支持的后端：`memory`、`milvus`、`llama_stack`、`valkey`、`qdrant`。

上传的文件本身保存在 Router 的磁盘上，位于 `file_storage_dir`（默认 `/var/lib/vsr/data`）。[Router 本地磁盘上的文件](../../installation/storage-overview#router-local-files)说明了每种部署方式如何让该目录可写，以及如何让它持久保存。

`metadata_store` 控制向量存储和已上传文件元数据的注册表。本地或类生产堆栈若需重启安全，请使用 `postgres`；CLI 本地运行时在设置 `metadata_store: postgres` 时会配置 Postgres 并填充 `metadata_postgres` 连接默认值。仅对短暂的本地实验使用 `memory`，因为存储和文件元数据会在路由器重启后丢失。

使用来自[模型运行时](../../model-runtime/guides/embeddings)的嵌入（包括 Vela Embedding）时，每个新向量存储都会记录创建向量所用的表示身份：

- **已有的存储**仍然可见，上传的文件也会保留。
- **不兼容或无身份标记的存储**在搜索或关联文件时返回 `409 EMBEDDING_REINDEX_REQUIRED`。请创建新向量存储，并重新关联原上传文件 ID，以生成兼容的向量。
- **身份归 Router 所有。**客户端元数据不能替换 `_router_embedding_identity` 字段。
- **RAG 同样适用。**同样的检查适用于请求时 RAG 和缓存检索结果。

:::note
`llama_stack` 后端在远端生成搜索查询向量，因此不能与绑定身份的运行时文档向量组合。这类配置请使用 `memory`、`milvus`、`valkey` 或 `qdrant`。远程提供方的身份验证属于另一项能力。
:::

### 工具 {#tools}

工具目录属于决策，而不属于 `global`：决策的[工具插件](../plugin/tools.md)中 `select.tools_db_path`
指定其 add 模式从中选择的目录。

## 数据与安全 {#data-and-security}

- 缓存、memory 和向量存储可能包含提示词、响应、嵌入、检索文档或提取的记忆。为所选后端配置认证、加密、保留策略和租户/用户范围。
- 嵌入维度必须与现有集合匹配。嵌入模型或维度变更时，请重建或迁移索引。
- 工具选择控制向模型展示什么；它不授权工具执行。请在工具服务上强制权限。
- 见[完整后端示例](https://github.com/vllm-project/semantic-router/tree/main/config/runtime)以及 [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml) 中的完整配置契约。
