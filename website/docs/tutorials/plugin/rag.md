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

## Observability

The `rag` plugin emits Prometheus metrics and an OpenTelemetry tracing span for every
retrieval, so you can monitor retrieval health, latency, cache effectiveness, and result
quality per backend and per decision.

### Metrics

All metrics are exported on the router metrics endpoint (default `:9190/metrics`).

| Metric                          | Type      | Labels                       | Description                                                  |
| ------------------------------- | --------- | ---------------------------- | ------------------------------------------------------------ |
| `rag_retrieval_attempts_total`  | counter   | `backend`, `decision`, `status` | Retrieval attempts. `status` is `success`, `error`, or `config_error`. |
| `rag_retrieval_latency_seconds` | histogram | `backend`, `decision`        | End-to-end retrieval latency (embedding + backend search).   |
| `rag_similarity_score`          | gauge     | `backend`, `decision`        | Latest best similarity score of the retrieved documents.     |
| `rag_context_length_chars`      | histogram | `backend`, `decision`        | Length, in characters, of the retrieved context.             |
| `rag_cache_hits_total`          | counter   | `backend`                    | RAG result-cache hits (requires `cache_results: true`).      |
| `rag_cache_misses_total`        | counter   | `backend`                    | RAG result-cache misses (requires `cache_results: true`).    |

> `rag_similarity_score` is a gauge, so it reflects the most recent score per
> `(backend, decision)` rather than a distribution. Cache metrics are labeled by `backend`
> only.

### Tracing

Each retrieval creates the span `semantic_router.rag.retrieval` with these attributes:
`rag.backend`, `rag.decision`, `rag.similarity_threshold`, `rag.top_k`,
`rag.latency_seconds`, `rag.success`, `rag.context_length`, and `rag.similarity_score`.
On failure the span records the error and is marked with an error status.

### Example queries

```promql
# Retrieval latency P95 by backend
histogram_quantile(0.95, sum(rate(rag_retrieval_latency_seconds_bucket[5m])) by (le, backend))

# Retrieval success rate
sum(rate(rag_retrieval_attempts_total{status="success"}[$__range])) / sum(rate(rag_retrieval_attempts_total[$__range])) * 100

# Cache hit rate
sum(rate(rag_cache_hits_total[$__range])) / (sum(rate(rag_cache_hits_total[$__range])) + sum(rate(rag_cache_misses_total[$__range]))) * 100
```

These metrics are charted in the **RAG Retrieval Metrics** row of the bundled Grafana
dashboard (`llm-router-dashboard.serve.json`).
