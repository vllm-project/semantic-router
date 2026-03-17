# Stores and Tools

## Overview

This page covers the shared storage and tool blocks inside `global:`.

These settings back route-local plugins and router-wide tool behavior.

## Key Advantages

- Centralizes shared backing stores instead of repeating them per route.
- Keeps semantic cache, memory, retrieval, and tool catalogs consistent.
- Lets route-local plugins stay small and focused.
- Makes shared infrastructure dependencies explicit.

## What Problem Does It Solve?

Route-local plugins often depend on shared storage or tool state. If those dependencies are configured ad hoc inside each route, the system becomes inconsistent and harder to operate.

These `global:` blocks solve that by defining shared backing services once.

## When to Use

Use these blocks when:

- multiple routes depend on the same semantic cache or memory backend
- retrieval features need one shared vector store
- the router should expose one shared tool catalog
- backing-store configuration belongs to the whole router rather than one route

## Configuration

### Semantic Cache

```yaml
global:
  stores:
    semantic_cache:
      similarity_threshold: 0.8
      use_hnsw: true
```

### Memory

```yaml
global:
  stores:
    memory:
      enabled: true
```

### Vector Store

```yaml
global:
  stores:
    vector_store:
      provider: milvus
```

### Tools

```yaml
global:
  integrations:
    tools:
      enabled: true
      top_k: 3
      tools_db_path: deploy/examples/runtime/tools/tools_db.json
```
