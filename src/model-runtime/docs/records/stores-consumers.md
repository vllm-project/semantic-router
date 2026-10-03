# Embedding consumers: legacy in-process candle against the serving path

Semantic cache, memory and RAG rerank latency on the same inputs and the same
CPU cores. Legacy is the base tree's CPU default: candle, in the router
process. The new side is this branch: the `serving` facade with a managed
model runtime process.

- **Date:** 2026-10-04.
- **Machine:** node D, AMD EPYC 9575F. Both sides were pinned to the same 16
  cores (`taskset -c 96-111`); the managed runtime inherits that affinity.
- **Commits:**
  - legacy: base `1c6d372ec`;
  - serving: `75c1aa5e0`, whose Go tree matches the IP1 sha.

  Both are exact mirrors; a temporary harness was added to scratch copies.
  The base links a stub ONNX Runtime library that fails every call: this node
  cannot build ORT, and the candle path never calls it.
- **Models (the same files on both sides):**
  - `vllm-sr/Vela-1.0-Encoder-307M-Embedding` at `1e57cebf`;
  - `vllm-sr/Vela-1.0-Encoder-307M-Reranker` at `a388e41c`.
- **Runtime:** `vllm-sr-runtime` from the same mirror, default `native`
  engine (PyTorch 2.14.1 CPU), profile `exact`.
- **Inputs:** 1,200 distinct, deterministic prompt-like texts of 15–60 words.

## Workloads

- **Semantic cache:** the in-memory HNSW backend at the cache's view (256
  dimensions, layer 6), through the response-cache adapter. 1,000 writes,
  then two sets of 200 lookups:
  - **new queries:** texts never stored, so every lookup embeds;
  - **repeated queries:** stored texts. All 200 hit on both sides.
- **Memory:** the in-memory store at 256 dimensions, full depth. 1,000 writes,
  then 200 retrievals of new queries.
- **RAG rerank:** 50 queries × 20 documents, one `ScorePairs` call per query.

Each operation was timed on its own, after a 20-call warm-up. The table shows
milliseconds per operation, p50 / p95, averaged over two interleaved runs of
each side (legacy, serving, serving without vector cache, then again).

## Results (ms per operation, p50 / p95)

| Consumer | Legacy candle | Serving | Speed-up (p50) |
| --- | --- | --- | --- |
| Cache write | 15.7 / 21.6 | 7.6 / 9.0 | 2.1× |
| Cache lookup, new query | 16.1 / 21.1 | 7.7 / 8.9 | 2.1× |
| Cache lookup, repeated query | 14.0 / 21.7 | 0.07 / 0.14 | 200× |
| Memory write | 56.7 / 78.9 | 24.7 / 28.8 | 2.3× |
| Memory retrieval | 59.8 / 79.8 | 26.1 / 29.8 | 2.3× |
| RAG rerank, 20 documents | 1,994 / 2,521 | 231 / 284 | 8.6× |

With the router's vector cache disabled (`VLLM_SR_EMBEDDING_CACHE_MB=0`), a
repeated query took 0.79 / 1.2 ms. That is one round trip answered by the
runtime's own result cache. New-query lookups (8.2 ms p50), memory
retrievals (25.5 ms) and rerank (237 ms) did not change beyond noise.

## Reading

- **The model work.** It moved out of process, and it got faster. The runtime's
  PyTorch encoder runs the layer-6 exit and full depth well ahead of candle's
  CPU path on the same cores, and its reranker scores a query's 20 documents in
  one batch.
- **The round trip.** The runtime call (JSON over its socket) costs less than
  1 ms, as the uncached repeated-query lookup shows.
- **Repeated text.** Legacy re-embeds a text it has seen before. The router's
  content-hash vector cache answers it in 0.07 ms, with no runtime call.
- **Tail latency.** p95 / p50 is 1.1–1.2 on the serving path and 1.3–1.6 for
  legacy.

No consumer regressed.
