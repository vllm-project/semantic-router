# Native engine performance on ROCm (MI325X)

> **Phase 1 record** (Decision 2.0, [#4481](https://github.com/vllm-project/semantic-router/pull/4481) and its follow-ups). The Phases 2–4 records are the `decision1-*`, `vela1-*`, `vela2-*`, `embed-*`, `stores-*`, `router-latency-cpu*`, `router-latency-rocm*`, `rocm-router-image` and `removal-footprint*` files.

The native engine against each released package's own runtime, on the
same node and GPU type.

- **Date:** 2026-10-03.
- **Device:** one AMD Instinct MI325X (gfx942) per run, in the packages' release
  images.
- **Commits:** the native side at `df51b0d5f` (graph cap 4,096 padded tokens);
  `99208b6d3` (no graph eviction) does not change these benches.
- **Released side:** each package's runtime: Transformers remote code with
  the phase A fast path (fused kernels and HIP graphs).
- **Contention:** the five smaller sizes ran one at a time on an otherwise idle
  node. Vega's native run shared the host with other benches, which shows in
  its p95.

Benches (`tools/gpu_bench.py`; the released side
`v2/release/runtime_bench.py`):

- **Single request:** the first 400 typed-final prompts as single requests on
  the exact profile; two untimed passes (graphs captured), then one timed
  pass.
- **Many questions:** the public request of `tools/many_questions.py` (one
  ~300-token ticket) at 16, 64 and 128 questions; p50 of 20 runs.
- **Throughput:** 512 pre-rendered requests in waves of C concurrent requests
  through the scheduler (2 ms batch window); two untimed passes, then one
  timed pass.

## Single request, exact (ms)

| Model | Released p50 | Released p95 | Native p50 | Native p95 |
| --- | --- | --- | --- | --- |
| Kai-0.6B | 4.94 | 13.50 | 4.85 | 4.90 |
| Eos-0.8B | 5.72 | 6.60 | 5.57 | 5.62 |
| Sol-2B | 7.17 | 7.20 | 7.22 | 7.32 |
| Nox-4B | 12.73 | 14.40 | 12.54 | 12.77 |
| Lux-9B | 18.55 | 18.69 | 18.39 | 18.64 |
| Vega-27B | 71.0 | 73.9 | 71.2 | 99.3 |

## Many questions about one input (ms, p50)

"Shared" is the released runtime's `share_context=True` against the native
`shared_context` profile; the two answer identically (parity record).

| Model | Questions | Exact, released | Exact, native | Shared, released | Shared, native |
| --- | --- | --- | --- | --- | --- |
| Kai-0.6B | 16 | 30.8 | 28.2 | 27.8 | 22.7 |
| | 64 | 109.6 | 94.4 | 64.3 | 54.7 |
| | 128 | 217.8 | 199.2 | 109.7 | 112.0 |
| Eos-0.8B | 16 | 32.0 | 32.6 | 30.0 | 33.4 (below break-even: exact) |
| | 64 | 113.8 | 110.9 | 67.1 | 66.9 |
| | 128 | 228.6 | 232.3 | 153.1 | 136.9 |
| Sol-2B | 16 | 51.2 | 52.3 | 40.5 | 42.0 |
| | 64 | 184.5 | 183.3 | 87.4 | 88.0 |
| | 128 | 490.9 | 372.6 | 162.8 | 171.1 |
| Nox-4B | 16 | 109.8 | 111.9 | 64.7 | 57.5 |
| | 64 | 414.9 | 409.0 | 177.6 | 166.4 |
| | 128 | 809.7 | 826.0 | 328.3 | 314.9 |
| Lux-9B | 16 | 173.0 | 177.9 | 82.9 | 80.2 |
| | 64 | 650.9 | 647.9 | 246.9 | 233.0 |
| | 128 | 1,297.1 | 1,319.6 | 461.1 | 447.7 |
| Vega-27B | 16 | 703.2 | 685.9 | 298.8 | 275.6 |
| | 64 | 2,583.7 | 2,551.0 | 868.0 | 817.0 |
| | 128 | 5,319.0 | 5,286.0 | 1,690.8 | 1,561.9 |

- `shared_context` is 1.8–3.4× faster than exact at 128 questions.
- Native exact at 128 questions costs up to 2% over the released runtime on
  Nox and Lux: batches above 4,096 padded tokens run without graphs (graph cap
  below).

## Throughput (requests/s, single-question requests)

| Model | Exact (any C) | `batching` C = 1 | C = 4 | C = 16 | C = 64 |
| --- | --- | --- | --- | --- | --- |
| Kai-0.6B | 234 | 154 | 402 | 681 | 837 |
| Eos-0.8B | 197 | 138 | 371 | 606 | 721 |
| Sol-2B | 151 | 113 | 257 | 365 | 415 |
| Nox-4B | 84 | 71 | 131 | 165 | 177 |
| Lux-9B | 58 | 51 | 83 | 101 | 108 |
| Vega-27B | 15 | 14 | 22 | 26 | 28 |

- Exact runs one request per forward, so concurrency does not raise its
  throughput.
- `batching` coalesces the questions of concurrent requests: 3.6× (Kai) to
  1.9× (Vega) at C = 64. At C = 1 it pays the 2 ms window.

## Graph cap

Graphs are captured only up to 4,096 padded tokens per batch, measured on an
idle GPU:

- **Kai-0.6B:** a graph saves 5% at 2,700 padded tokens, 3% at 5,400, 2% at
  10,800 and nothing at 43,500. Below about 1,400 tokens it halves the latency
  (one row: 4.5 vs 8.8 ms).
- **Eos-0.8B:** 6% at 2,800 tokens, 1% at 5,600, nothing from 11,000.
- **Concurrent traffic:** clients form many distinct large shapes; capturing
  each cost several forwards and kept graph replays rare.
- **Cache limit:** at most 512 graphs and 4 GiB of outputs are captured; new
  shapes then run eagerly. Graphs are never evicted: evicting graphs that share
  the memory pool caused GPU memory access faults on ROCm (an Index-scale run).

## Reproduce

```bash
python3 tools/gpu_bench.py --package PACKAGE_DIR --prompts TYPED_FINAL.prompts.jsonl --count 400 \
  --output native.json [--base-path BASE_DIR]
```
