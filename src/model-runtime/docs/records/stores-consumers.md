# Embedding consumers: legacy in-process candle against the serving path

Semantic cache, memory and RAG rerank latency and throughput on the same
inputs and the same CPU cores. Legacy is the base tree's CPU default: candle,
in the router process. The new side is this branch: the `serving` facade with
a managed model runtime process.

- **Date:** 2026-10-04.
- **Machine:** node B, AMD EPYC 9575F. Each side ran in its own cgroup cpuset
  scope on the same 16 cores (`systemd-run --scope -p AllowedCPUs=48-63`), so
  the router process, the managed runtime it starts and every ORT, oneDNN or
  candle thread stayed on them. `GOMAXPROCS`, `OMP_NUM_THREADS` and
  `MKL_NUM_THREADS` were 16. The host's 1-minute load was 9–29 of 160 during
  the runs.
- **Commits:**
  - legacy: base `1c6d372ec`;
  - serving: `80460bc67`, which has the batch-invariance probe for embedders
    and rerankers (`ebecf9a3a`). The harness sets the embedding profile itself.

  Both are exact mirrors; a temporary harness was added to scratch copies.
  The base links a stub ONNX Runtime library that fails every call: the build
  node cannot build ORT, and the candle path never calls it.
- **Models (the same files on both sides):**
  - `vllm-sr/Vela-1.0-Encoder-307M-Embedding` at `1e57cebf`;
  - `vllm-sr/Vela-1.0-Encoder-307M-Reranker` at `a388e41c`.
- **Runtime:** `vllm-sr-runtime` from the same mirror, `native` engine
  (PyTorch 2.14.1 CPU), 16 threads. Embeddings run at `batching`, which the
  router's implicit embedding deployments use, and at `exact`. The reranker
  runs at `exact`, as the maintained config declares it.
- **Inputs:** 2,024 distinct, deterministic prompt-like texts of 15–60 words.

## Workloads

- **Semantic cache:** the in-memory HNSW backend at the cache's view (256
  dimensions, layer 6), through the response-cache adapter. 1,000 writes,
  then two sets of 200 lookups:
  - **new queries:** texts never stored, so every lookup embeds;
  - **repeated queries:** stored texts. All 200 hit on both sides.
- **Memory:** the in-memory store at 256 dimensions, full depth. 1,000 writes,
  then 200 retrievals of new queries.
- **RAG rerank:** 50 queries × 20 documents, one `ScorePairs` call per query.
- **Four concurrent callers:** 400 cache lookups, 400 memory retrievals and 24
  reranks of 20 documents. Every text is one that nothing has embedded before,
  so neither the router's vector cache nor the runtime's result cache answers.
  The result is operations per second over the wall time, with the latency
  under that load.

Each sequential operation was timed on its own, after a 20-call warm-up. The
tables show the mean of two interleaved rounds (legacy, `exact`, `batching`);
the rounds agree within 2 %.

## One caller (ms per operation, p50 / p95)

| Consumer | Legacy candle | `exact` | `batching` |
| --- | --- | --- | --- |
| Cache write | 15.1 / 20.9 | 5.9 / 6.8 | 7.9 / 9.0 |
| Cache lookup, new query | 15.4 / 20.7 | 6.0 / 6.8 | 8.1 / 9.0 |
| Cache lookup, repeated query | 13.4 / 20.8 | 0.07 / 0.13 | 0.07 / 0.12 |
| Memory write | 55.1 / 76.6 | 18.5 / 21.0 | 20.4 / 22.7 |
| Memory retrieval | 57.2 / 77.1 | 18.8 / 21.3 | 20.9 / 22.9 |
| RAG rerank, 20 documents | 1,952 / 2,439 | 201 / 236 | 201 / 240 |

## Four concurrent callers (operations per second; p50 / p95 ms under load)

| Consumer | Legacy candle | `exact` | `batching` |
| --- | --- | --- | --- |
| Cache lookup, new query | 187 (21 / 27) | 159 (25 / 29) | 268 (15 / 17) |
| Memory retrieval | 49.4 (82 / 101) | 53.6 (75 / 86) | 96.9 (41 / 49) |
| RAG rerank, 20 documents | 1.45 (2,473 / 3,199) | 4.77 (828 / 945) | 4.82 (821 / 928) |

The RAG rerank row is the reranker at `exact` in every column; only the
embedding profile changes.

## Why embeddings stay at `batching`

At this head the probe passes: the Vela Embedding package loads
`batch_invariant`, so `exact` may share forwards between concurrent jobs.
Four callers still get 159 cache lookups/s at `exact`, 15 % below legacy, and
four callers' lookups take as long as four lone lookups (25 ms against 6 ms).
An instrumented copy of the profile, which logged each plan's batch sizes,
shows why: at `exact`, 3,223 of 3,225 plans held one row.

Two things keep the rows apart:

- The scheduler plans each take on its own. With closed-loop callers, a take
  is usually the one request that just arrived.
- `merged()` groups rows by power-of-two length class, and these texts span
  several classes.

With the length class removed from `merged()` (a scratch copy of the runtime;
the same cores, two rounds), `exact` reaches 213 cache lookups/s (19 / 22 ms)
and 87 memory retrievals/s, above legacy, with the same one-caller numbers. The
probe already checks one mixed-length batch against each row alone. The native
CPU forward is unpadded, so mixed lengths cost no padding.

Until the scheduler change lands, the router's implicit embedding deployments
use `batching`. Embeddings feed similarity thresholds, not bit-exact answers.
A deployment that declares `exact` keeps it.

Without the router's vector cache (`VLLM_SR_EMBEDDING_CACHE_MB=0`), a repeated
query took 0.66 / 0.92 ms at the previous head (node D): one round trip
answered by the runtime's own result cache.

## ROCm

The consumers' model work is the runtime's Vela Embedding and Reranker. On an
MI325X the runtime beats the legacy ONNX Runtime ROCm path for both, in latency
and throughput (`embed-performance.md`, "ROCm"). The router-side work around
those calls does not depend on the device.

## Reading

- **The model work.** It moved out of process, and it got faster. The
  runtime's PyTorch encoder runs the layer-6 exit and full depth well ahead of
  candle's CPU path on the same cores. Its reranker scores a query's 20
  documents in one packed forward.
- **Concurrency.** `batching` coalesces concurrent embeddings into shared
  forwards. That adds about 2 ms to a lone request (8.1 ms against 6.0 ms at
  `exact`). It gets four callers 1.4× legacy's cache lookups and 2.0× its
  memory retrievals.
- **The round trip.** The runtime call (JSON over its socket) costs less than
  1 ms, as the uncached repeated-query lookup shows.
- **Repeated text.** Legacy re-embeds a text it has seen before. The router's
  content-hash vector cache answers it in 0.07 ms, with no runtime call.
- **Tail latency.** p95 / p50 is 1.1–1.2 on the serving path and 1.3–1.6 for
  legacy.

At the serving profile (`batching` embeddings, `exact` reranker), no consumer
regressed: every row is faster than legacy at p50, at p95 and in throughput.
At `exact` embeddings, only four callers' cache lookups fall short of legacy.
