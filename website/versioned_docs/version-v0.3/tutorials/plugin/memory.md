# Memory

## Overview

`memory` is a route-local plugin for retrieving and storing conversation memory.

It aligns to `config/plugin/memory/session-memory.yaml`.

## Key Advantages

- Keeps memory behavior local to the routes that benefit from it.
- Supports retrieval and auto-store in one plugin.
- Separates route-local memory policy from shared backing-store config.

## What Problem Does It Solve?

Not every route should pay the complexity or privacy cost of memory. `memory` lets one matched route opt into session-aware behavior while the shared store remains configured under `global.stores.memory`.

## When to Use

- a route should retrieve prior conversation context
- the route should automatically store useful new turns
- memory settings should stay local to one route family

## Configuration

The memory plugin requires a backing store configured under `global.stores.memory`. The router supports three backends:

- **Milvus** (default) — distributed vector database, best for large-scale production
- **Valkey** — lightweight single-binary option using the Search module, best for dev/test or existing Valkey infra
- **Qdrant** — single-binary with gRPC, simpler ops than Milvus, good for small-to-large workloads

See the [Stores and Tools](../global/stores-and-tools.md) tutorial for global memory configuration, the [Valkey Memory deployment guide](../../installation/valkey-memory.md) for Valkey-specific setup, or the [Qdrant deployment guide](../../installation/qdrant.md) for Qdrant-specific setup.

Use this fragment under `routing.decisions[].plugins`:

```yaml
plugin:
  type: memory
  configuration:
    enabled: true
    retrieval_limit: 5
    similarity_threshold: 0.72
    auto_store: true
```
