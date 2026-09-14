---
translation:
  source_commit: "ff3c6e01ed8f284edfbf473e478a12403e6863f4"
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
