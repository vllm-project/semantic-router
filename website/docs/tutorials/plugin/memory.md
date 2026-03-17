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
