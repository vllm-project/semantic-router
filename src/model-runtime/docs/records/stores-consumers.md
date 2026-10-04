# Embedding consumers: legacy in-process candle against the serving path

Semantic cache, memory and RAG rerank latency and throughput on the same
inputs and the same CPU cores. Legacy is the base tree's CPU default: candle,
in the router process. The new side is this branch: the `serving` facade with
a managed model runtime process.

- **Date:** 2026-10-04.
- **Machine:** node D, AMD EPYC 9575F. Both sides were pinned to the same 16
  cores (`taskset -c 48-63`); the managed runtime inherits that affinity.
  Other workloads ran on other cores; the host's 1-minute load was 29–42 of
  160 during the runs.
- **Commits:**
  - legacy: base `1c6d372ec`;
  - serving: `edc1a2347`, which has the packed-linear `exact` encoders and the
    uvloop server. The commit after it only changes which profile the router
    asks for, and the harness sets the profile itself.

  Both are exact mirrors; a temporary harness was added to scratch copies.
  The base links a stub ONNX Runtime library that fails every call: this node
  cannot build ORT, and the candle path never calls it.
- **Models (the same files on both sides):**
  - `vllm-sr/Vela-1.0-Encoder-307M-Embedding` at `1e57cebf`;
  - `vllm-sr/Vela-1.0-Encoder-307M-Reranker` at `a388e41c`.
- **Runtime:** `vllm-sr-runtime` from the same mirror, `native` engine
  (PyTorch 2.14.1 CPU), 16 threads. Embeddings run at the `batching` profile,
  which the router's implicit embedding deployments use, and at `exact` for
  comparison. The reranker runs at `exact`, as the maintained config declares it.
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
tables show the mean of two interleaved rounds. Each round ran legacy, serving,
serving without the vector cache, and serving with embeddings at `exact`.

## One request at a time (ms per operation, p50 / p95)

| Consumer | Legacy candle | Serving | Speed-up (p50) |
| --- | --- | --- | --- |
| Cache write | 16.4 / 22.2 | 8.2 / 9.6 | 2.0× |
| Cache lookup, new query | 16.7 / 22.4 | 8.4 / 10.1 | 2.0× |
| Cache lookup, repeated query | 15.0 / 22.2 | 0.07 / 0.13 | 214× |
| Memory write | 60.7 / 83.7 | 21.4 / 24.3 | 2.8× |
| Memory retrieval | 63.7 / 84.7 | 22.0 / 24.9 | 2.9× |
| RAG rerank, 20 documents | 2,036 / 2,629 | 205 / 250 | 9.9× |

## Four concurrent callers (operations per second; p50 / p95 ms under load)

| Consumer | Legacy candle | Serving | Speed-up |
| --- | --- | --- | --- |
| Cache lookup, new query | 162 (24 / 34) | 249 (16 / 19) | 1.54× |
| Memory retrieval | 46.3 (87 / 108) | 87.8 (44 / 51) | 1.90× |
| RAG rerank, 20 documents | 1.40 (2,638 / 3,232) | 4.76 (806 / 987) | 3.4× |

## Embeddings at `exact`

With the embedding deployment at `exact`, one request at a time is faster
still (cache lookup 6.4 / 7.7 ms, memory retrieval 20.3 / 23.1 ms). Four
callers, however, get 146 cache lookups/s, below legacy's 162, and 49.7
memory retrievals/s. At `exact` the runtime merges concurrent jobs only for a
model probed batch-invariant at load, and it probes the classification heads,
not the pooled embedding heads, so every embedding forward runs alone. The
router's implicit embedding deployments therefore use `batching`: embeddings
feed similarity thresholds, not bit-exact answers. A deployment that declares
`exact` keeps it.

Without the router's vector cache (`VLLM_SR_EMBEDDING_CACHE_MB=0`), a repeated
query took 0.66 / 0.92 ms: one round trip answered by the runtime's own result
cache. The other rows did not change beyond noise.

## ROCm

The consumers' model work is the runtime's Vela Embedding and Reranker. On an
MI325X the runtime beats the legacy ONNX Runtime ROCm path for both, in latency
and throughput (`embed-performance.md`, "ROCm"). The router-side work around
those calls does not depend on the device.

## Reading

- **The model work.** It moved out of process, and it got faster. The runtime's
  PyTorch encoder runs the layer-6 exit and full depth well ahead of candle's
  CPU path on the same cores, and its reranker scores a query's 20 documents in
  one packed forward.
- **Concurrency.** `batching` coalesces concurrent embeddings into shared
  forwards. That adds about 2 ms to a lone request (8.4 ms against 6.4 ms at
  `exact`) and roughly doubles what four callers get.
- **The round trip.** The runtime call (JSON over its socket) costs less than
  1 ms, as the uncached repeated-query lookup shows.
- **Repeated text.** Legacy re-embeds a text it has seen before. The router's
  content-hash vector cache answers it in 0.07 ms, with no runtime call.
- **Tail latency.** p95 / p50 is 1.1–1.2 on the serving path and 1.3–1.4 for
  legacy.

No consumer regressed: every row is faster than legacy at p50, at p95 and in
throughput.
