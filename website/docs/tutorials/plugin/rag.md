# RAG

## Overview

`rag` retrieves external context for a matched route before generation. Choose
Milvus or Qdrant for direct vector-store retrieval, or use an external HTTP
API, MCP tools, OpenAI file search, the Router's vector-store service, or a
primary/fallback hybrid.

## Key Advantages

- Keeps retrieval local to routes that actually need it.
- Supports backend-specific retrieval settings in one place.
- Avoids forcing every route to inject documents or tool context.

## What Problem Does It Solve?

Some routes need external document retrieval before answering, while most do not. `rag` lets the matched route perform retrieval and injection without globalizing that behavior.

## When to Use

- a route should fetch documents or facts before the final model call
- retrieval should use Milvus, Qdrant, or another explicit backend
- different routes need different retrieval settings

## Configuration

Choose one backend:

| Backend | Use it for | Required backend fields |
| --- | --- | --- |
| `milvus` | Direct retrieval from a Milvus collection | `collection`; optionally reuse the response-cache connection |
| `qdrant` | Direct retrieval from a Qdrant collection | `collection`; optionally reuse the response-cache connection |
| `external_api` | A service with a custom HTTP request contract | `endpoint`, `request_format` |
| `mcp` | Retrieval exposed as an MCP tool | `server_name`, `tool_name` |
| `openai` | OpenAI file search | `vector_store_id`, `api_key` |
| `vectorstore` | The Router-managed vector-store service | `vector_store_id` |
| `hybrid` | A primary backend with an optional fallback | `primary`, plus backend-specific nested configuration |

The examples below show the two direct-store options and the external HTTP
API. For the other backends, start from the field names above and validate the
complete config before deployment.

Add the plugin under `routing.decisions[].plugins`:

**Milvus backend:**

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
        metadata_field: metadata
```

**Qdrant backend:**

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

**External API backend:**

```yaml
plugins:
  - type: rag
    configuration:
      enabled: true
      backend: external_api
      top_k: 5
      similarity_threshold: 0.78
      injection_mode: tool_role
      on_failure: warn
      backend_config:
        endpoint: https://search.example.com/query
        request_format: custom
        request_template: '{"query":"${user_content}","top_k":${top_k},"threshold":${threshold}}'
        timeout_seconds: 15
        max_response_body_bytes: 16777216
```

Retrieved documents become provider-bound context. Apply collection-level
access control and avoid mixing tenants in one unrestricted search scope.
Similarity thresholds are embedding-model specific. See complete examples:
[`milvus.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/rag/milvus.yaml),
[`qdrant.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/rag/qdrant.yaml),
and
[`external-api.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/rag/external-api.yaml).
