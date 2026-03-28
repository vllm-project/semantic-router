# Semantic Cache

## Overview

`semantic-cache` is a route-local plugin for reusing semantically similar prior responses.

It aligns to `config/plugin/semantic-cache/high-recall.yaml` and `config/plugin/semantic-cache/memory.yaml`.

## Key Advantages

- Reuses prior responses only on routes that benefit from cache hits.
- Keeps route-local thresholds separate from global store setup.
- Supports different cache policies for different routes.

## What Problem Does It Solve?

Some routes benefit strongly from reuse, while others need fresh generation every time. `semantic-cache` keeps the reuse policy local to the route instead of making cache behavior global by default.

## When to Use

- one route should prefer cached responses when queries are very similar
- different routes need different similarity thresholds or TTLs
- the route should use a shared semantic cache backend configured in `global.stores.semantic_cache`

## Configuration

Use this fragment under `routing.decisions[].plugins`:

```yaml
plugin:
  type: semantic-cache
  configuration:
    enabled: true
    similarity_threshold: 0.92
    ttl_seconds: 86400
```
