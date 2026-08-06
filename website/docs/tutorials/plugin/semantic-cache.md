# Semantic Cache

## Overview

`semantic-cache` is a route-local plugin for reusing semantically similar prior responses.

It aligns to `config/fragments/plugin/semantic-cache/high-recall.yaml` and `config/fragments/plugin/semantic-cache/memory.yaml`.

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
    mode: exact_then_semantic
    allow_request_controls: true
    control_header: x-vsr-cache-control
    similarity_threshold: 0.92
    ttl_seconds: 86400
```

`mode` accepts:

- `semantic` (default): vector lookup only.
- `exact`: normalized exact request lookup only.
- `exact_then_semantic`: exact lookup first, then vector lookup on a miss.

The exact tier is available with the in-memory, Redis, Valkey, Milvus, Qdrant,
and hybrid cache backends. Anthropic client requests are replayed in the
Anthropic response or SSE wire format.

Streaming and non-streaming requests use separate cache identities so replay
never translates a cached response across wire modes. Semantic matching uses a
compatibility fingerprint over system/history, tools, response format,
generation parameters, client protocol, and route policy, plus hard recipe,
tenant, request-model, and selected-model partitioning.

Streaming replay preserves content, reasoning, refusal, tool calls, terminal
usage, and the finish reason for a single choice. Multi-choice streaming
responses fail explicitly instead of silently dropping choices.

When request controls are enabled, the configured header accepts `no-cache`
(skip reads), `no-store` (skip writes), or `bypass` (skip both). The controls
are ignored unless the route explicitly enables them.
