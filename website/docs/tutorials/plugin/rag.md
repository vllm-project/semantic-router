# RAG

## Overview

`rag` is a route-local plugin for retrieval-augmented generation.

It aligns to `config/plugin/rag/milvus.yaml` (Milvus) and `config/plugin/rag/qdrant.yaml` (Qdrant).

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
