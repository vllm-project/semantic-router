# Release note: Router Memory Prometheus label sets

Upgrading to a release that includes [#4326](https://github.com/vllm-project/semantic-router/issues/4326) changes the published label sets for two Router Memory metrics. Update dashboards, recording rules, and alerts that reference the old labels before or immediately after rollout.

## `llm_memory_retrieval_total`

| | Label set |
|---|-----------|
| **Before** | `backend`, `status`, `user_id` |
| **After** | `backend`, `status` |

`status` continues to distinguish retrieval outcomes such as `hit`, `miss`, and `error` on Milvus, Valkey, and Qdrant backends. The counter no longer creates one time series per authenticated user.

Example migration:

```promql
# Before (per user — no longer valid)
sum by (backend, status, user_id) (llm_memory_retrieval_total)

# After (aggregate by backend and outcome)
sum by (backend, status) (llm_memory_retrieval_total)
```

## `llm_memory_store_size`

| | Label set |
|---|-----------|
| **Before** | `backend`, `user_id` |
| **After** | `backend` |

This gauge was not wired to live store counts on earlier releases; the `user_id` dimension was unused in production paths. The label is removed so a future wiring does not reintroduce unbounded identity cardinality.

## Qdrant retrieval telemetry

On earlier releases, Qdrant `Retrieve` did not increment `llm_memory_retrieval_total`. It recorded a generic successful store operation with zero duration instead. After this change, Qdrant retrieval emits the same hit/miss/error retrieval counters and durations as Milvus and Valkey.

## Per-user analysis

Router Memory still scopes retrieval and storage by user in the store and request path. Per-user debugging belongs in logs, traces, and management APIs—not Prometheus labels on these counters.
