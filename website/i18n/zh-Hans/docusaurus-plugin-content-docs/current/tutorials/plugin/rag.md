---
translation:
  source_commit: "96399a94b9030d66f46c5d45f9a838defc091153"
  source_file: "docs/tutorials/plugin/rag.md"
  outdated: false
---

# RAG

## 概览

`rag` 在生成前为已匹配路由检索外部上下文。可选择 Milvus 或 Qdrant 进行直接向量存储检索，或使用外部 HTTP API、MCP 工具、OpenAI 文件搜索、Router 的向量存储服务，或主/备混合。

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
| `mcp` | 作为 MCP 工具暴露的检索 | `server_name`、`tool_name` |
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

`vectorstore` 后端可以使用本地 Vela 相关性模型，对结构化检索结果重新排序，再组装上下文。先声明模型部署和当前 recipe 的 `rag.reranker` 绑定，再为路由启用 `rerank`：

```yaml
global:
  model_catalog:
    deployments:
      document-ranker:
        artifact: models/Vela-1.0-Encoder-307M-Reranker
        provider: candle
        device: cpu
        precision: native
        input:
          max_tokens: 4096
          overflow: reject
routing:
  model_bindings:
    rag.reranker:
      deployment: document-ranker
      contract: relevance_scores.v1
      adapter: vela_reranker
      pair_scorer:
        layer: 22
        dimension: 768
```

在同一 recipe 的 decision 中添加以下插件：

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
        top_k: 3
      on_failure: block
```

`top_k` 控制检索候选数量；`rerank.top_k` 限制重排序后注入提示词的结果数量，省略时保留全部候选。原始相关性 logit 越高，排名越靠前；同分时保留检索顺序。文档 ID、片段 ID 和原始检索相似度保持不变。重排序 logit 尚未经校准，不能代替 embedding 相似度阈值。

模型使用 tokenizer 的 query/document 配对模板。token 预算包含两段文本及特殊 token；超出预算会被拒绝，不会截断任一文本。加载时会校验所选层和维度是否经过训练；设为零时使用模型实际的完整深度或宽度。CPU 开销随候选数量和文本对长度增加，应显式设置部署预算。

Candle 模型目录必须包含 encoder 权重、`config.json`、`tokenizer.json`、`matryoshka_config.json` 和 `classification_heads.safetensors`。ORT 部署通过绑定的 `head` 字段选择完整计算图；图内的 `semantic_router.pair_scorer` 元数据必须声明实际输出层、维度及 `relevance_logit` 契约，图文件名不能作为其语义依据。

只有可达且启用了 `rerank` 插件的 recipe 才会加载模型。模型缺失、无效分数和输入超限均遵循 RAG 的 `on_failure` 策略。缓存上下文按 recipe、embedding 身份和重排序模型身份隔离。运行时 trace 记录实际重排序延迟和分数；路由 preview 不执行检索，也不生成重排序耗时。其他 RAG 后端目前不支持 `rerank`，需先提供结构化候选结果。
