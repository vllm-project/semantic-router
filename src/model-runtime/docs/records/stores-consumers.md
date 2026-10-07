# Embedding consumers: legacy in-process candle against the serving path

Semantic cache, memory and RAG rerank latency and throughput on the same
inputs and the same CPU cores. Legacy is the base tree's CPU default: candle,
in the router process. The new side is this branch: the `serving` facade with
a managed model runtime process.

- **Date:** 2026-10-05.
- **Machine:** node B, AMD EPYC 9575F. Each run had its own cgroup cpuset
  scope on the same 16 cores (`systemd-run --scope -p AllowedCPUs=48-63`), so
  the router process, the managed runtimes it starts and every oneDNN or
  candle thread stayed on them. The effective cpuset, read inside each of the
  18 scopes, was 48–63. `GOMAXPROCS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS`
  were 16. Other timed runs used other cores; the host's 1-minute load was
  34–63 of 160.
- **Commits:**
  - legacy: base `1c6d372ec`;
  - serving: `b604edeab`. It has the batch-invariance probe for embedders and
    rerankers (`ebecf9a3a`), and `exact` shares batches across lengths on
    models that pack rows (`10564d6c0`). The harness sets the embedding
    profile itself, so the change of the implicit default that follows
    (`c7b48cb1b`) does not change what these rows run.

  Both are exact mirrors; a temporary harness was added to scratch copies.
  The base links a stub ONNX Runtime library that fails every call: the build
  node cannot build ORT, and the candle path never calls it.
- **Models (the same files on both sides):**
  - `vllm-sr/Vela-1.0-Encoder-307M-Embedding` at `1e57cebf`;
  - `vllm-sr/Vela-1.0-Encoder-307M-Reranker` at `a388e41c`.
- **Runtime:** `vllm-sr-runtime` from the same mirror, `native` engine, 16
  threads. Its venv has the router image's pins (`Dockerfile.extproc`):
  PyTorch 2.10.0 CPU, then the runtime's dependencies and its `multimodal`
  extra as the resolver picked them (ONNX Runtime 1.30.0, NumPy 2.5.3).
  Embeddings run at `exact`, which the router's implicit embedding
  deployments use, and at `batching`. The reranker runs at `exact`, as the
  maintained config declares it.
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

## Method

Six rounds. A round runs legacy, `exact` and `batching` once each, every run
in a fresh process and its own scope, with the order rotated: legacy first in
rounds 1 and 4, `exact` in rounds 2 and 5, `batching` in rounds 3 and 6. Each
sequential operation was timed on its own, after a 20-call warm-up. The tables
show means over the six rounds. The intervals pair each round's runtime value
with legacy's from the same round: a 95% t interval (5 degrees of freedom) over
the six differences.

## One caller (ms per operation, p50 / p95)

| Consumer | Legacy candle | `exact` | `batching` |
| --- | --- | --- | --- |
| Cache write | 15.1 / 20.7 | 4.41 / 5.53 | 6.18 / 7.17 |
| Cache lookup, new query | 15.7 / 21.1 | 4.39 / 5.50 | 6.38 / 7.27 |
| Cache lookup, repeated query | 14.1 / 21.4 | 0.07 / 0.14 | 0.07 / 0.13 |
| Memory write | 55.4 / 77.7 | 12.9 / 16.3 | 14.3 / 17.0 |
| Memory retrieval | 56.7 / 76.3 | 13.5 / 16.3 | 14.9 / 17.3 |
| RAG rerank, 20 documents | 1,932 / 2,453 | 240 / 296 | 226 / 272 |

## Four concurrent callers (operations per second; p50 / p95 ms under load)

| Consumer | Legacy candle | `exact` | `batching` |
| --- | --- | --- | --- |
| Cache lookup, new query | 181 (21.9 / 28.6) | 275 (15.0 / 18.9) | 318 (12.6 / 14.8) |
| Memory retrieval | 48.0 (84.6 / 104) | 98.4 (42.2 / 50.7) | 118 (34.5 / 40.6) |
| RAG rerank, 20 documents | 1.42 (2,559 / 3,256) | 4.31 (906 / 1,061) | 4.50 (862 / 1,006) |

The RAG rerank row is the reranker at `exact` in every column; only the
embedding profile changes.

## Runtime minus legacy, with 95% intervals

Latency in ms (negative is faster); throughput in operations per second
(positive is more). "1 caller" throughput is 1,000 / the mean latency.

| Row | Metric | `exact` | `batching` |
| --- | --- | --- | --- |
| Cache write, 1 caller | p50 | −10.7 [−11.2, −10.2] | −8.94 [−9.32, −8.55] |
| Cache write, 1 caller | p95 | −15.2 [−15.9, −14.5] | −13.6 [−13.9, −13.2] |
| Cache write, 1 caller | ops/s | +159 [+139, +180] | +95.7 [+87.6, +104] |
| Cache lookup, new query, 1 caller | p50 | −11.3 [−12.7, −9.89] | −9.31 [−10.7, −7.96] |
| Cache lookup, new query, 1 caller | p95 | −15.6 [−17.8, −13.5] | −13.9 [−15.9, −11.8] |
| Cache lookup, new query, 1 caller | ops/s | +162 [+140, +185] | +92.9 [+83.9, +102] |
| Cache lookup, repeated query, 1 caller | p50 | −14.1 [−15.8, −12.3] | −14.1 [−15.8, −12.3] |
| Cache lookup, repeated query, 1 caller | p95 | −21.2 [−23.2, −19.3] | −21.3 [−23.2, −19.3] |
| Cache lookup, repeated query, 1 caller | ops/s | +11,517 [+10,919, +12,115] | +12,078 [+11,455, +12,702] |
| Memory write, 1 caller | p50 | −42.6 [−45.5, −39.6] | −41.1 [−43.8, −38.4] |
| Memory write, 1 caller | p95 | −61.4 [−66.6, −56.2] | −60.7 [−65.7, −55.7] |
| Memory write, 1 caller | ops/s | +60.3 [+50.1, +70.5] | +52.2 [+44.2, +60.2] |
| Memory retrieval, 1 caller | p50 | −43.3 [−45.4, −41.2] | −41.9 [−43.5, −40.3] |
| Memory retrieval, 1 caller | p95 | −59.9 [−62.1, −57.8] | −58.9 [−60.5, −57.4] |
| Memory retrieval, 1 caller | ops/s | +58.2 [+46.3, +70.0] | +50.3 [+43.8, +56.8] |
| RAG rerank, 20 documents, 1 caller | p50 | −1,691 [−1,762, −1,620] | −1,706 [−1,765, −1,646] |
| RAG rerank, 20 documents, 1 caller | p95 | −2,158 [−2,296, −2,019] | −2,181 [−2,302, −2,060] |
| RAG rerank, 20 documents, 1 caller | ops/s | +3.79 [+3.11, +4.48] | +4.04 [+3.45, +4.63] |
| Cache lookup, new query, 4 callers | p50 | −6.90 [−10.3, −3.45] | −9.35 [−11.0, −7.74] |
| Cache lookup, new query, 4 callers | p95 | −9.66 [−13.0, −6.28] | −13.8 [−16.0, −11.6] |
| Cache lookup, new query, 4 callers | ops/s | +94.3 [+44.2, +144] | +137 [+110, +165] |
| Memory retrieval, 4 callers | p50 | −42.5 [−53.6, −31.3] | −50.2 [−56.6, −43.8] |
| Memory retrieval, 4 callers | p95 | −53.5 [−66.4, −40.6] | −63.6 [−71.8, −55.4] |
| Memory retrieval, 4 callers | ops/s | +50.4 [+31.3, +69.6] | +69.8 [+55.4, +84.2] |
| RAG rerank, 20 documents, 4 callers | p50 | −1,653 [−1,888, −1,417] | −1,698 [−1,918, −1,477] |
| RAG rerank, 20 documents, 4 callers | p95 | −2,195 [−2,460, −1,929] | −2,250 [−2,507, −1,993] |
| RAG rerank, 20 documents, 4 callers | ops/s | +2.89 [+2.22, +3.55] | +3.08 [+2.45, +3.70] |

Every interval lies on the runtime's side of zero, at both profiles: no row
regressed. The widest is four callers' cache lookups at `exact`, whose rounds
ran 208–327/s against legacy's 169–187/s.

## `exact` is the default again

At `exact` a batch-invariant model shares forwards between the jobs that are
queued together. Before `10564d6c0`, those shared batches held one
power-of-two length class each. These texts span several classes, and with
closed-loop callers the queue is short, so 3,223 of 3,225 plans held one row:
four callers got 159 cache lookups/s against legacy's 187, and the router's
implicit embedding deployments ran at `batching` instead.

A `task_heads` forward lays rows back to back with no padding, so the length
class bought nothing there. The model now says so (`packs_rows`, which
`task_heads` sets from its engine), and `exact` drops the class for it.
Models that pad, such as multimodal Omni, keep it. The load-time probe
already checks one mixed-length batch, over five length classes, against
each row alone. With that, `exact` clears legacy on every row above, so the
implicit `@embedding.*` deployments use it again (`c7b48cb1b`), as the design
makes it the default. Their vectors are the same alone and inside shared
batches. `batching` still gets four callers more lookups (318 against 275/s)
at about 2 ms more per lone request (6.38 against 4.39 ms). A deployment that
declares a profile keeps it.

## PyTorch 2.14.1 (diagnostic)

The same six-round A/B with a venv on PyTorch 2.14.1 (the earlier records'
version, otherwise the same pins) agreed: `exact` and `batching` were wholly
better than legacy on all 27 rows. Four callers got 217 cache lookups/s at
`exact` against legacy's 186 (+30.8 [+14.2, +47.3]) and 91.5 memory
retrievals/s against 49.0. One caller's lookup took 6.04 ms at p50 there,
against 4.39 ms on 2.10.0; the reranker took 199 ms against 240 ms.

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
- **Concurrency.** At `exact`, four callers' embeddings share packed forwards,
  so they get 1.5× legacy's cache lookups and 2.0× its memory retrievals.
  `batching` waits for more rows and gets 1.8× and 2.5×.
- **The round trip.** The runtime call (JSON over its socket) costs less than
  1 ms. Without the router's vector cache (`VLLM_SR_EMBEDDING_CACHE_MB=0`), a
  repeated query took 0.66 / 0.92 ms at an earlier head (node D): one round
  trip, answered by the runtime's own result cache.
- **Repeated text.** Legacy re-embeds a text it has seen before. The router's
  content-hash vector cache answers it in 0.07 ms, with no runtime call.
- **Tail latency.** p95 / p50 is 1.17–1.26 on the serving path at `exact` (the
  0.07 ms repeated-query lookups aside) and 1.23–1.52 for legacy.
