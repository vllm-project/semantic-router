# Response Cache

## Overview

`response_cache` is the route-local plugin for reusing exact or semantically
compatible prior responses.

## Key Advantages

- Reuses prior responses only on routes that benefit from cache hits.
- Keeps route-local thresholds separate from global store setup.
- Supports different cache policies for different routes.

## What Problem Does It Solve?

Some routes benefit strongly from reuse, while others need fresh generation
every time. `response_cache` keeps the reuse policy local to the route.

## When to Use

- one route should prefer cached responses when queries are very similar
- different routes need different similarity thresholds or TTLs
- the route should use a cache backend configured in `global.stores.response_cache`

## Configuration

Add the plugin under `routing.decisions[].plugins`:

```yaml
plugins:
  - type: response_cache
    configuration:
      enabled: true
      mode: exact_then_semantic
      scope: user
      semantic:
        similarity_threshold: 0.92
      ttl_seconds: 86400
      request_controls:
        enabled: true
        header: x-vsr-cache-control
        allowed: [no-cache, no-store, bypass, max-age, ttl]
        max_ttl_seconds: 86400
      personalized:
        mode: disabled
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
usage, finish reasons, and choice indexes for complete single- or multi-choice
streams. Incomplete streams are never cached.

When request controls are enabled, the configured header accepts the authorized
directives. `max-age` bounds read freshness and `ttl` bounds write lifetime;
caller TTL values are clamped to `max_ttl_seconds`.

## Migration

`semantic-cache`, `semantic_cache`, and `response-cache` are accepted as
deprecated aliases and normalize to `response_cache`. Likewise,
`global.stores.semantic_cache` is read as a deprecated alias for
`global.stores.response_cache`. Do not configure both spellings in the same
document. Export, Dashboard saves, and DSL decompilation always emit the
canonical names.

## Operations

The management API exposes redacted health, capabilities, statistics, candidate
configuration testing, scoped invalidation, epoch-based flush, and a
hash-chained audit view under `/api/v1/response-cache/*`. Invalidation defaults
to dry-run. Flush requires the explicit confirmation phrase
`flush response cache` and never calls backend-wide `FLUSHALL`.

Cached responses can contain user or tenant data. Choose an appropriate scope,
TTL, backend authentication, encryption, and invalidation process. Semantic
thresholds must be calibrated for the configured embedding model, and routes
with personalized RAG or memory should not reuse pre-enrichment responses
without an explicit policy. See complete examples:
[`high-recall.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-cache/high-recall.yaml)
and
[`memory.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-cache/memory.yaml).
