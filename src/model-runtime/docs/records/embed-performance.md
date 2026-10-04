# Embedders, rerankers and Omni: performance

The runtime against the legacy bindings it replaces, on the same inputs and
hardware (design section 18), and the CPU engine decision between ONNX
Runtime and PyTorch.

- **Date:** 2026-10-04.
- **Runtime:** the encoders' router-path rows at `bf4a889a9` and their
  engine rows at `7c2c6e21b` (the commits since change the scheduler and
  Omni, not `LoadedModel.run` for these models); Omni latency at
  `2e62199d9`, Omni throughput at `45286db73`; ROCm rows at `5a73fc17e`
  (later commits change no code these models run on the GPU).
- **CPU:** 16 vCPUs of a KVM guest on AMD EPYC 9575F hosts, one NUMA node;
  PyTorch 2.10 (oneDNN, MKL) and ONNX Runtime 1.30 in the runtime, ONNX
  Runtime 1.22 in the legacy bindings. Every process of both sides runs in
  one cgroup cpuset of those 16 vCPUs (a systemd scope, or `--cpuset-cpus`
  for containers). A `taskset` mask is not enough: the legacy ONNX Runtime
  sessions pin their threads to CPUs they read from the host, which a mask
  does not stop; inside the cpuset that pinning fails and the threads stay
  on the 16 vCPUs. The guest's 1-minute load was 14–60 of 160 during the
  runs (other workloads, on other vCPUs).
- **ROCm:** one AMD Instinct MI325X per side (gfx942) with 8 host vCPUs in a
  container cpuset, PyTorch 2.12 on ROCm 7.2; the legacy side in the legacy
  router image (ONNX Runtime 1.22.1 ROCm with the CK flash-attention operator
  library).
- **Legacy paths:** the router's native facade at `61aa7eb2d` (candle on the
  CPU for Vela Embedding, Reranker and Qwen3-Embedding, the router's default;
  ONNX Runtime on the prepared bundle for Omni), and the legacy ONNX Runtime
  execution as the bindings ran it: one sequence per session run, pairs one
  after another (`tools/embed_legacy_baseline.py`; on ROCm the AMD recipe's
  `onnx/model_fa.onnx` with `ck_flash_attention`).
- **Runtime side:** `Runtime.call` with the result cache off for the router
  path; `LoadedModel.run` on token rows for engine-level scenarios
  (`tools/embed_bench.py`), the default engine options (graphs and fused
  kernels on).

## CPU: legacy router path vs `Runtime.call`

`tools/embed_legacy.py ab`: the legacy facade's test binary serves calls in
one process, the runtime in another, on the same 16 vCPUs. Per input one
legacy and one runtime call alternate (the order flips every round), each
50 ms after the previous call, so neither side's thread pools still spin on
the cores when the other side's call starts; then 10 s closed-loop windows
of 4 callers per side, rotating. Latency over all pairs; throughput is the
median window. Four rounds over 31 corpus texts (short queries to documents
of a few thousand tokens) or 4 rerank sets of 8 documents.

| Job | Pairs | Legacy p50 / p95 ms | Runtime p50 / p95 ms | Median speedup | Legacy → runtime req/s |
| --- | --- | --- | --- | --- | --- |
| Embedding (768, layer 22) | 124 | 41.2 / 3,352 | 14.0 / 297 | 3.23× | 5.3 → 24.6 |
| Embedding (256, layer 22) | 124 | 42.2 / 3,258 | 13.1 / 263 | 3.32× | 5.4 → 23.6 |
| Embedding (768, layer 11) | 124 | 22.5 / 1,674 | 8.6 / 157 | 2.63× | 12.8 → 44.3 |
| Reranker (22, 768), 8 documents | 16 | 343.2 / 559.6 | 40.6 / 72.1 | 7.93× | 6.9 → 20.8 |
| Reranker (6, 256), 8 documents | 16 | 84.3 / 142.2 | 14.1 / 23.9 | 6.37× | 26.4 → 66.1 |
| Qwen3-Embedding | 124 | 88.3 / 10,838 | 30.6 / 808 | 3.17× | 1.9 → 8.0 |

The runtime runs the encoder's linear layers through oneDNN's pre-packed
FP32 kernels (the `exact` path's kernel variant on x86 CPUs), packs a rerank
set's pairs into one forward (the legacy scorer ran them one by one), and
runs long documents packed with block-local attention instead of candle's
dense padded path.

## CPU: Omni, the legacy facade's ONNX Runtime adapter vs `Runtime.call`

The same `ab` method on the Omni jobs: the corpus's 26 short texts, the
bundle's 3 golden images (encoded bytes) and its 4 (Nano) or 5 (Mini) golden
audio clips. Latency comes from 16 rounds without load windows, with
`Runtime.call` given the body size as the HTTP server gives it (runtime at
`2e62199d9`); throughput from 4 rounds of 4-caller windows (runtime at
`45286db73`).

| Job | Pairs | Legacy p50 / p95 ms | Runtime p50 / p95 ms | Legacy → runtime req/s |
| --- | --- | --- | --- | --- |
| Omni Nano text | 416 | 5.11 / 8.36 | 5.82 / 9.09 | 206.5 → 201.5 |
| Omni Nano image | 48 | 121.0 / 130.8 | 125.2 / 136.4 | 8.14 → 8.05 |
| Omni Nano audio | 64 | 214.5 / 373.0 | 163.7 / 331.7 | 4.54 → 6.39 |
| Omni Mini text | 416 | 29.7 / 79.1 | 29.0 / 75.2 | 24.68 → 24.67 |
| Omni Mini image | 48 | 322.2 / 355.5 | 325.2 / 337.8 | 3.03 → 3.02 |
| Omni Mini audio | 80 | 610.7 / 824.4 | 625.9 / 890.4 | 1.55 → 1.68 |

**Omni is not yet at the bar.** Nano text trails by 0.7 ms per request at
p50 and p95, the images by 1–3.5 % at p50, and Mini audio by 2.5 % at p50
and 8 % at p95; text and image throughput is within 2.4 % of legacy, on the
wrong side. Nano audio passes every column; Mini text passes latency and
ties throughput. Both sides run the same graphs, so the difference is in
what surrounds them:

- **Audio:** the runtime's NumPy preprocessing builds each resampler kernel
  once and skips Whisper frames that hold only padding, and the CLAP windows
  and the audio graph share one ONNX Runtime pool (with a pool per session,
  CLAP then audio took 124 ms on 16 cores instead of 43 ms).
- **Short text:** a bare ONNX Runtime 1.30 run of the text graph beats the
  legacy call in the same conditions (4.7 vs 5.2 ms). `Runtime.call` hands
  the request from the event loop to the model's worker and back, and on
  these KVM guests waking an idle thread costs about 0.5 ms. Omni already
  skips the third hand-off, to the CPU device thread (`LoadedModel.device_thread`;
  1.4 ms per request).
- **Concurrency:** each input runs its graphs alone, so Omni is
  batch-invariant; the `exact` profile hands it every queued request and it
  runs up to four inputs at once on the shared pool. Images and audio carry a
  scheduler cost (text-token equivalents of their forward time), so each runs
  in its own batch.
- **Images:** the vision graphs take the same time on both ONNX Runtime
  versions (bare runs: Nano 103 ms, Mini 283 ms); the runtime's Pillow
  decode, resize and normalization took 3.0 / 1.6 ms of a request.
  Normalizing through a per-channel table (bit-identical, `0e8c6f458`) cuts
  that by about 40 %; an image-only A/B there (16 rounds, idle node): Nano
  p50 / p95 114.9 / 121.6 → 115.9 / 125.7 ms, Mini 310.7 / 323.2 → 308.2 /
  323.9 ms.
- **Mini audio:** its NumPy features for clips of several CLAP windows are
  what remains to profile.

## CPU: ONNX Runtime pools in a process that serves several CPU models

One process serving Vela Embedding on `onnxruntime` and Vela Domain on
`native`, 16 threads, the same 16 vCPUs: per text length, Domain alone,
Domain called right after an embedding, and the embedding alone (p50 ms over
40 calls, 50 ms between calls). Measured at `ebecf9a3a` with each pool
policy swapped in; the runtime takes the second row from `0b4c563cd`.

| CPU pool policy | Domain alone 16 / 64 / 256 | Domain after embedding | Embedding alone |
| --- | --- | --- | --- |
| One shared pool that spins (before `7c21e05a5`) | 9.3 / 16.3 / 34.4 | 21.5 / 33.7 / 66.4 | 16.8 / 25.0 / 58.5 |
| Own pool per session, spins inside a run, `force_spinning_stop` | 9.5 / 16.7 / 35.0 | 10.0 / 16.7 / 34.6 | 16.4 / 24.7 / 58.5 |
| Own pool per session that never spins (`7c21e05a5`) | 10.1 / 16.9 / 34.9 | 10.3 / 16.9 / 34.8 | 19.8 / 31.4 / 80.5 |
| Own pool per session that spins (legacy's policy) | 9.4 / 16.9 / 35.0 | 21.0 / 37.1 / 66.4 | 16.8 / 24.5 / 58.7 |

ONNX Runtime's threads spin for milliseconds after a run, on the cores the
next model's forward needs. The shared pool can't be told to stop
(`session.force_spinning_stop` reaches only a session's own pool), so a
process with more than one CPU model (`EngineOptions.exclusive_cpu` False)
gives each CPU session its own pool that spins while a run lasts and stops
when it returns. Never spinning was not enough: Omni Nano text, whose
forward is a few milliseconds of small parallel sections, took 7.4 ms against
legacy's 5.1 in the A/B process that serves Nano and Mini. A process with
one CPU model keeps the shared pool, which Omni's CLAP-then-audio chain needs.

## CPU: legacy ONNX Runtime execution vs both runtime engines

Synthetic token rows (`tools/embed_corpus.py`), three rounds on the same 16
vCPUs (legacy, runtime `onnxruntime`, runtime `native` in turn each round),
p50 / p95 ms, medians of the rounds; throughput in items per second.

| Vela Embedding | Legacy ORT | Runtime `onnxruntime` | Runtime `native` |
| --- | --- | --- | --- |
| one text, 16 tokens | 11.9 / 12.5 | 12.1 / 13.0 | 7.2 / 7.5 |
| one text, 64 tokens | 20.1 / 21.7 | 21.2 / 21.3 | 14.4 / 14.6 |
| one text, 256 tokens | 53.8 / 55.7 | 55.6 / 56.2 | 33.4 / 34.2 |
| one text, 1,024 tokens | 198.3 / 204.3 | 201.4 / 207.8 | 116.5 / 124.3 |
| 32 texts of 16–256 tokens | 1,137 / 1,162 (28.0/s) | 957 / 966 (33.6/s) | 528 / 541 (60.6/s) |

| Vela Reranker | Legacy ORT | Runtime `onnxruntime` | Runtime `native` |
| --- | --- | --- | --- |
| query + 10 documents | 377.0 / 380.3 (26.5/s) | 313.6 / 317.8 (32.0/s) | 182.2 / 185.0 (55.0/s) |
| query + 50 documents | 1,828 / 1,850 (27.3/s) | 1,345 / 1,399 (36.9/s) | 914 / 961 (54.8/s) |

Qwen3-Embedding has no ONNX graph in the pinned package, so it runs native
only: p50 21.4 / 42.7 / 100.5 / 1,190 ms for one text of 16 / 64 / 256 /
1,024 tokens and 5.6 s for 32 texts (5.7/s); the router-path table above
compares it with legacy candle.

## ROCm: legacy ONNX Runtime + CK flash attention vs the native engine

Engine-level scenarios, three rounds side by side (one GPU and 8 host vCPUs
each, so both sides meet the same host load), 100 timed runs per scenario,
p50 (p95) ms, medians of the rounds; the native engine as it serves (gfx942
fused rotary, encoder graphs, block-local attention from 2,048 tokens).

| Vela Embedding | Legacy | Runtime |
| --- | --- | --- |
| one text, 16 tokens | 4.13 (4.42) | 1.54 (1.56) |
| one text, 64 tokens | 4.14 (4.47) | 1.74 (1.76) |
| one text, 256 tokens | 4.57 (4.88) | 2.66 (2.69) |
| one text, 1,024 tokens | 7.21 (7.40) | 6.50 (6.96) |
| 32 texts of 16–256 tokens | 119.6 (146.8) | 27.0 (27.8) |

| Vela Reranker | Legacy | Runtime |
| --- | --- | --- |
| query + 10 documents | 41.7 (42.2) | 9.57 (10.1) |
| query + 50 documents | 183.0 (217.1) | 34.2 (35.3) |

Throughput, items per second (median of the rounds): 32 texts 247 → 1,187;
query + 10 documents 240 → 1,044; query + 50 documents 252 → 1,455.

Short rows replay a captured HIP graph per length bucket; multi-row batches
replay one only when the bucket pads little (else they run packed); a rerank
request is one packed forward of all its pairs. The 1,024-token row was the
last one behind: the fused rotary kernel (`rotary_half`, bit-exact) took it
from 7.5 to 6.5 ms, and block-local attention, which costs more than dense
masked attention below 2,048 tokens on this GPU, now starts there.
Qwen3-Embedding has no legacy GPU path (candle has no ROCm); on the MI325X it
takes 14–16 ms for one text of up to 256 tokens (launch-bound: its decoder
path runs without graphs), 26.5 ms at 1,024 tokens and 104 ms for 32 texts.

## ONNX Runtime vs PyTorch on the CPU, per model

- **Vela Embedding:** PyTorch. The native engine is 1.4–1.7× faster than
  ONNX Runtime (the runtime's engine or the legacy execution) on single texts
  and 1.8× faster than the runtime's ONNX Runtime engine on 32 texts. Before
  the `exact` path took oneDNN's pre-packed FP32 linears (`5a73fc17e`), ONNX
  Runtime was 7–30 % faster than native on single texts.
- **Vela Reranker:** PyTorch: 1.7× faster than the ONNX Runtime engine with
  10 documents and 1.5× with 50. Native also serves all twenty pair scorers
  (four exits × five dimensions) from one forward; the graph engine runs one
  graph per scorer.
- **Qwen3-Embedding:** PyTorch only (the pinned package ships no ONNX graph).
- **Omni:** ONNX Runtime only (the prepared bundle is its graphs).

Decision: `auto` keeps native first for the three `task_heads` models, and no
`BuiltinModel.engines` entry sets a CPU preference. `engine: onnxruntime`
stays available for a package that ships graphs: it matches the legacy ONNX
Runtime execution on single texts (within 5 %) and beats it on batches, but
it serves only the exits it has graphs for. Torch's OpenMP threads spin after
a native forward and used to slow an ONNX Runtime run that followed in the
same process (16 tokens: 13.0 → 19.8 ms); the runtime sets
`GOMP_SPINCOUNT=10000` before torch loads, which removes that and leaves
native unchanged.
