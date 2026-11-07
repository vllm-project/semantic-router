---
is_mtpe: false
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
  source_file: "docs/tutorials/plugin/rag.md"
  outdated: false
---

# RAG

## 概览

`rag` 在生成前为已匹配路由检索外部上下文。可选择 Milvus 或 Qdrant 进行直接向量存储检索，或使用外部 HTTP API、OpenAI 文件搜索、Router 的向量存储服务，或主/备混合。在路由器具备 MCP 工具调用器之前，`mcp` 会在启动时被拒绝。

## 主要优势

- 将检索限制在真正需要它的路由内。
- 在一处支持后端专用检索设置。
- 避免强迫每条路由注入文档或工具上下文。

## 解决什么问题？

有些路由在回答前需要外部文档检索，大多数则不需要。`rag` 让已匹配路由执行检索和注入，而不把该行为全局化。

## 何时使用

- 某条路由应在最终模型调用前获取文档或事实
- 检索应使用 Milvus、Qdrant 或其他显式后端
- 不同路由需要不同检索设置

## 配置

选择一种后端：

| 后端 | 用途 | 必需的后端字段 |
| --- | --- | --- |
| `milvus` | 从 Milvus collection 直接检索 | `collection`；可选复用响应缓存连接 |
| `qdrant` | 从 Qdrant collection 直接检索 | `collection`；可选复用响应缓存连接 |
| `external_api` | 具有自定义 HTTP 请求契约的服务 | `endpoint`、`request_format` |
| `mcp` | 在具备 MCP 工具调用器之前，启动时会被拒绝 | `server_name`、`tool_name` |
| `openai` | OpenAI 文件搜索 | `vector_store_id`、`api_key` |
| `vectorstore` | Router 管理的向量存储服务 | `vector_store_id` |
| `hybrid` | 带可选回退的主后端 | `primary`，以及后端专用嵌套配置 |

对于 `external_api`，`max_response_bytes` 限制每个响应正文；省略或 `0` 使用 4 MiB。

对于 OpenAI `direct_search`，`max_response_bytes` 对每次向量存储搜索响应应用同样的 4 MiB 默认值。

下面的示例展示两种直接存储选项。其他后端请从上面的字段名开始，并在部署前校验完整配置。

在 `routing.decisions[].plugins` 下添加该插件：

**Milvus 后端：**

```yaml
plugins:
  - type: rag
    configuration:
      enabled: true
      backend: milvus
      top_k: 5
      similarity_threshold: 0.78
      injection_mode: tool_role
      on_failure: warn
      backend_config:
        collection: docs
        reuse_cache_connection: true
        content_field: content
```

**Qdrant 后端：**

```yaml
plugins:
  - type: rag
    configuration:
      enabled: true
      backend: qdrant
      top_k: 5
      similarity_threshold: 0.78
      injection_mode: tool_role
      on_failure: warn
      backend_config:
        collection: docs
        reuse_cache_connection: true
        content_field: content
```

检索到的文档会成为绑定提供商的上下文。请应用 collection 级访问控制，并避免在一个不受限的搜索范围内混合租户。相似度阈值取决于嵌入模型。完整示例见：
[`milvus.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/rag/milvus.yaml)
和
[`qdrant.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/rag/qdrant.yaml)。

## 神经网络重排序 {#neural-reranking}

`vectorstore` 后端可以使用 Vela 相关性模型，对结构化检索结果重新排序，再组装上下文。先把重排序模型声明为部署：

```yaml
global:
  model_catalog:
    deployments:
      document-ranker:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Reranker
        device: cpu
        input:
          max_tokens: 4096
          overflow: reject
```

然后在 decision 插件的 `rerank` 块中指定它：

```yaml
plugins:
  - type: rag
    configuration:
      enabled: true
      backend: vectorstore
      backend_config:
        vector_store_id: vs-your-documents
      top_k: 10
      rerank:
        deployment: document-ranker
        top_k: 3
        layer: 22
        dimension: 768
      on_failure: block
```

`rerank.deployment` 也可以指向 `rerank` 部署（Router 向其 `<endpoint>/rerank` 发送请求的远程服务）或挂接的模型运行时；见[模型目录参考](../../installation/model-catalog.md#r2-attached-reranker-named-where-it-is-used)。同一 recipe 的各个 RAG 插件指定相同的重排序模型、层和维度；`layer` 和 `dimension` 只适用于模型运行时重排序模型。

`top_k` 控制检索候选数量；`rerank.top_k` 限制重排序后注入提示词的结果数量，省略时保留全部候选。原始相关性 logit 越高，排名越靠前；同分时保留检索顺序。文档 ID、片段 ID 和原始检索相似度保持不变。重排序 logit 尚未经校准，不能代替 embedding 相似度阈值。

模型使用 tokenizer 的 query/document 配对模板。token 预算包含两段文本及特殊 token；超出预算会被拒绝，不会截断任一文本。加载时会校验所选的 `layer` 和 `dimension` 是否经过训练；设为零或不写时使用模型实际的完整深度或宽度。CPU 开销随候选数量和文本对长度增加，应显式设置部署预算。

本地重排序模型运行在[模型运行时](../../model-runtime/guides/rerank.md)中。启动时路由器会检查模型是否声明了所选的出口，因此模型未训练过的 `layer` 和 `dimension` 会在流量到达前被拒绝。

只有可达且启用了 `rerank` 插件的 recipe 才会加载模型。模型缺失、无效分数和输入超限均遵循 RAG 的 `on_failure` 策略。缓存上下文按 recipe、embedding 身份和重排序模型身份隔离。运行时 trace 记录实际重排序延迟和分数；路由 preview 不执行检索，也不生成重排序耗时。其他 RAG 后端目前不支持 `rerank`，需先提供结构化候选结果。
