# RAG

## Overview

`rag` is a route-local plugin for retrieval-augmented generation.

It aligns to `config/plugin/rag/milvus.yaml` (Milvus), `config/plugin/rag/qdrant.yaml` (Qdrant), and `config/plugin/rag/external-api.yaml` (external HTTP API).

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

Use this fragment under `routing.decisions[].plugins`:

**Milvus backend:**

```yaml
plugin:
  type: rag
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
plugin:
  type: rag
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
plugin:
  type: rag
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

The supported request formats are `pinecone`, `weaviate`, `elasticsearch`, and `custom`. Custom
request templates are parsed as non-null JSON objects or arrays at configuration load, including
when `external_api` is a hybrid child. Exact placeholder nodes such as `${top_k}` stay typed,
placeholders cannot be used as object keys, user content cannot add fields or change the configured
object/array shape, and configured JSON numbers retain their original precision. The lowercase
`${user_content}`, `${top_k}`, and `${threshold}` names are reserved for runtime substitution;
other braced lowercase names fail before environment expansion. Use uppercase names such as
`${RAG_TENANT}` for intentional environment references. Successful response bodies default to an
exact 16 MiB limit. Set `max_response_body_bytes` to a positive byte count up to 64 MiB to override
it; a response at the limit is accepted and a response one byte larger is rejected without decoding
a truncated prefix.
