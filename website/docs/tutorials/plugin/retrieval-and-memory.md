# Retrieval and Memory Plugins

## Overview

This page covers the route-local plugins that reuse stored context, retrieval results, or replay state.

It aligns to:

- `config/plugin/semantic-cache/`
- `config/plugin/rag/`
- `config/plugin/memory/`
- `config/plugin/router-replay/`

## Key Advantages

- Lets one route opt into retrieval or reuse behavior without changing every route.
- Keeps shared backing services in `global:` but route-local behavior in the decision.
- Supports faster or more stateful routes without overloading the base router contract.
- Makes debugging and replay capture route-scoped when needed.

## What Problem Does It Solve?

Not every route should pay the same latency, retrieval, or storage cost. Some routes benefit from semantic cache, some need RAG, some need session memory, and some need replay capture.

These plugins solve that by attaching reuse and retrieval behavior only where it belongs.

## When to Use

Use this family when:

- one route should reuse prior responses before calling a model
- retrieval should be injected into one route only
- a route needs memory lookup or session state
- replay/debug capture should be enabled only for selected traffic

## Configuration

Each plugin lives inside `routing.decisions[].plugins`.

### Semantic Cache

```yaml
routing:
  decisions:
    - name: cached_support
      plugins:
        - type: semantic-cache
          configuration:
            enabled: true
            similarity_threshold: 0.9
```

### RAG

```yaml
routing:
  decisions:
    - name: retrieval_route
      plugins:
        - type: rag
          configuration:
            enabled: true
            backend: milvus
            top_k: 5
```

### Memory

```yaml
routing:
  decisions:
    - name: memory_route
      plugins:
        - type: memory
          configuration:
            enabled: true
            retrieval_limit: 5
            similarity_threshold: 0.7
```

### Router Replay

```yaml
routing:
  decisions:
    - name: replayable_route
      plugins:
        - type: router_replay
          configuration:
            enabled: true
            capture_inputs: true
```

These plugins often depend on `global:` backing config such as `global.stores.semantic_cache`, `global.stores.memory`, or shared retrieval stores. Keep the route-local plugin small and put shared storage settings in `global/`.
