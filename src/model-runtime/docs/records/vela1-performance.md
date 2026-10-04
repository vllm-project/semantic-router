# Vela 1.0 text models: performance against the legacy router path

On the same inputs and hardware, the runtime answers every Vela 1.0 text
model faster than the router path it replaces, per request (p50 and p95) and
under load, on CPU and on ROCm. Answers are the ones `vela1-parity.md`
records; raw results are in `vela1-performance.json`.

- **Date:** 2026-10-04. **Runtime commit:** `0a2ced483`, unless a row
  names another.
- **Legacy side:** the router's native facade at `61aa7eb2d`
  (`tools/legacy_parity.py`). On CPU that is the CPU recipe (candle). On
  ROCm it is the AMD recipe (`config/recipes/vela-amd`: ONNX Runtime
  MIGraphX / ROCm EP).
- **Hardware:**
  - CPU: AMD EPYC 9575F, 16 pinned vCPUs (node B 96–111) shared by both
    sides, with the node's 1-minute load logged every 30 s;
  - ROCm: one AMD Instinct MI325X with 8 pinned host vCPUs.
- **Runtime side:** `Runtime.call` as the HTTP server calls it: one event
  loop, the encoded body size passed so small requests plan inline, result
  cache off, `exact` unless a row says otherwise.

## CPU, interleaved with the legacy facade

A shared node's load moves faster than a run lasts, so `tools/legacy_parity.py
ab` runs both sides on the same 16 vCPUs at once:

- **Per input:** the facade's test binary, in serve mode, answers one
  input, then the runtime answers the same input. The order flips each
  round, over 2 rounds of all 547 / 64 inputs.
- **Under load:** 20 s windows of 4 closed-loop callers then rotate over the
  legacy facade, the runtime's `exact` profile and its `batching` profile
  (2 rounds; medians).

`exact` batches the requests queued together, because the CPU forward is
batch-invariant (next section), so its answers stay those of a request
alone. The node's 1-minute load stayed at or below 61 (median 37) on 160 vCPUs during the run.

| Job | Pairs | p50 legacy → runtime (ms) | p95 legacy → runtime (ms) | Median per-input speedup | 4 callers, calls/s: legacy → `exact` (`batching`) |
| --- | --- | --- | --- | --- | --- |
| domain | 1,094 | 35.32 → 11.31 | 199.75 → 32.19 | 3.20× | 40.6 → 70.5 (101.5) |
| guard | 1,090 | 35.29 → 11.22 | 195.05 → 29.53 | 3.18× | 18.5 → 43.4 (49.0) |
| safety | 1,094 | 33.40 → 10.68 | 195.15 → 28.84 | 3.27× | 39.5 → 73.1 (109.2) |
| shield | 1,094 | 34.56 → 11.24 | 201.44 → 30.36 | 3.20× | 38.4 → 68.2 (106.7) |
| factcheck | 1,094 | 34.68 → 11.47 | 196.32 → 30.92 | 3.17× | 39.8 → 70.6 (110.6) |
| feedback | 1,094 | 34.27 → 11.30 | 205.47 → 30.38 | 3.26× | 40.5 → 66.3 (104.7) |
| modality | 1,094 | 33.80 → 11.05 | 200.75 → 29.91 | 3.25× | 39.4 → 67.8 (104.8) |
| hazard | 1,094 | 34.47 → 11.36 | 201.95 → 30.91 | 3.25× | 12.5 → 29.6 (18.5) |
| pii | 1,094 | 33.43 → 10.84 | 190.25 → 28.43 | 3.16× | 18.3 → 30.9 (20.2) |
| pii_truncate | 1,094 | 84.70 → 11.42 | 254.88 → 30.75 | 7.08× | 28.0 → 77.5 (114.3) |
| halu | 128 | 1343.81 → 135.59 | 40,269.44 → 2,753.80 | 10.23× | 1.5 → 3.5 (2.5) |

Under load, `exact` serves 1.6–2.8× legacy's calls per second.

- **Windowed jobs (Guard, Hazard, PII) and Halu:** they scan long prompts
  in full. The scheduler (design section 9) answers each job as soon as its
  own batches have run and runs short work first, so a short call no longer
  waits behind a long windowed one (Guard 2.3×, Hazard 2.4×, PII 1.7×, Halu 2.4× legacy).
- **Short prompts:** the scheduler plans the jobs it takes between two
  forwards on their own. With 4 closed-loop callers that is usually one new
  request, so concurrent short calls rarely share a forward (66–78
  calls/s). `batching` waits up to 2 ms for company and serves
  102–114 calls/s on the same prompts. At `93a3492c0`, before
  this scheduler, `exact` merged everything queued (102–110/s on
  them) but left PII and Hazard at 1.1× and 1.5× legacy.

## ROCm

One MI325X, the same inputs. The AMD recipe compiles one fixed
8,192-token ONNX Runtime session per model, so every legacy request is an
8K forward (Hazard's operating point compiles 2,048-token windows).

| Job | p50 legacy → runtime (ms) | p95 legacy → runtime (ms) | 4 callers, calls/s |
| --- | --- | --- | --- |
| domain | 154.06 → 1.79 | 155.91 → 3.01 | 6.0 → 424.5 |
| guard (ROCm EP) | 245.81 → 1.90 | 258.07 → 3.30 | 4.1 → 418.0 |
| safety | 155.26 → 1.74 | 164.00 → 3.03 | 6.4 → 439.1 |
| factcheck | 158.24 → 1.82 | 163.86 → 3.22 | 6.3 → 437.6 |
| feedback | 157.93 → 1.83 | 164.11 → 3.14 | 6.0 → 424.0 |
| modality | 158.56 → 1.86 | 167.20 → 3.12 | 6.2 → 421.6 |
| hazard | 13.38 → 1.80 | 14.13 → 3.08 | 61.3 → 341.1 |
| pii | 134.03 → 1.87 | 136.05 → 3.17 | 7.4 → 400.2 |
| shield (ORT graph ≠ checkpoint, timing only) | 151.22 → 1.74 | 157.22 → 2.99 | 6.5 → 450.1 |

Shield's ORT row is timing only: its package's ONNX graph answers
differently from its checkpoint (`vela1-parity.md`).

Some deployments move from the router's CPU defaults to a GPU. On the same
inputs, against legacy candle on CPU:

- Halu: p50 1,424 → 9.2 ms and p95 38.7 s → 69 ms;
- sequence models: p50 33–40 → 1.8–1.9 ms, and 4-caller throughput
  35–41 → 436–473 calls/s.

Profiles on ROCm (AMD-recipe inputs, 4 callers):

| Profile | Domain p50 / p95 (ms) | Calls/s, sequence models | Values |
| --- | --- | --- | --- |
| `exact` | 1.79 / 3.01 | 422–439 | the parity record's |
| `batching` (`b1eafc87a`) | 3.98 / 5.13 | 596–619 | identical to `exact` one request at a time |
| `max_speed`, BF16 copy (`973842d9e`, records only) | 6.69 / 11.00 | 437–468 | fails the 99% floor for PII and Halu |

`batching` waits up to 2 ms for concurrent requests, so a lone request pays
the window, and under load it serves a third more. The BF16 copy adds a
cast per linear to a launch-bound forward and is slower still, so the
family consents to none (`vela1-parity.md`).

## Where the time goes, and what the runtime does about it

Every number below is Vela 307M in FP32.

- **One forward for every head.** A request's inputs, their windows, every
  head that reads them and every task of a bundle become one packed
  forward. Identical token sequences are computed once, each head reads its
  rows of the shared hidden states, and the result cache keys rows by
  content.
- **Packed rows, attention in grids of similar length.** Embeddings, norms,
  projections and MLPs run on the real tokens only, and attention scatters
  rows into grids. Rows whose grid would pad more than 25% of its tokens
  (past 1,024) attend in separate grids, because attention costs
  rows × width². One grid against length groups, 16 EPYC vCPUs / MI325X:

  | Batch (tokens) | CPU | MI325X |
  | --- | --- | --- |
  | 2,000 + 7 × 20 | 4,154 → 708 ms | 60.7 → 12.9 ms |
  | 8,192 + 7 × 16 | 22.5 → 3.5 s | 287 → 51 ms |
  | 4,096 + 31 × 24 | 32.8 → 1.6 s | 464 → 25 ms |
  | 512, 300, 64, 32, 16, 16, 10, 8 | 622 → 210 ms | 18.3 → 10.6 ms |
  | 16 × 128 (uniform) | 499 → 493 ms | 8.9 → 8.8 ms |

- **Local layers in query blocks for long rows.** Two thirds of the layers
  attend within 64 positions. Long rows run them in 128-query blocks over a
  256-key span through the fused 4-D SDPA kernels, from 1,024 tokens on CPU
  and from 2,048 on MI325X. On MI325X a 2,048-token row goes 17.7 → 11.7 ms
  and 8,192 tokens 91 → 50 ms. Below 2,048 dense masked SDPA is faster
  there: 1,024 tokens take 7.5 ms dense against 11.0 blocked.
- **GPU: a graph per length bucket.** A short forward is launch overhead,
  so batches pad to a (rows, width) bucket and replay a HIP graph captured
  on the bucket's second use: 8 tokens 3.96 → 2.28 ms, 128 tokens
  4.17 → 2.98 ms, 16 × 64 tokens 6.02 → 4.86 ms. A bucket replays only while
  it pads at most 25% of the real tokens or is launch-bound (≤ 1,024
  tokens); other batches run packed. The fused gfx942 rotary kernel
  (`embed`) runs inside.
- **CPU: oneDNN's packed FP32 linears.** GEMMs are 80% of a short forward,
  and `F.linear` (MKL on EPYC) streams the weights at ~37 GB/s. oneDNN with
  weights reordered once is 2.5–4.4× faster per GEMM (768 → 2,304: M = 10
  71 → 23 µs, M = 512 1,098 → 426 µs), and gives a row the same result in
  any batch. Whole forward, one row, interleaved: 10 tokens 14.5 → 6.8 ms,
  32 tokens 19.4 → 9.7, 64 27.1 → 14.1, 128 84 → 66, 512 153 → 118, 1,024
  288 → 275. At 256 tokens MKL is ahead, 86 against 95 ms, the one length
  where it is.
- **CPU: `exact` batches concurrent requests without changing an answer.**
  Each row's forward is bit-identical alone or inside any batch, given:
  - packed linears;
  - GeGLU and the score sigmoid on aligned rows (`rowwise`);
  - unmasked grids for unpadded rows;
  - one rotary table per grid width;
  - grids of one length only.

  The family probes this on the host at load before it lets `exact`
  batch, so a PyTorch build where a kernel is not invariant keeps one
  request per forward.
- **CPU: one device thread.** PyTorch's OpenMP keeps a thread team per
  calling thread. Two teams on 16 vCPUs stop libgomp's spin-waiting, so
  every parallel region pays a wake-up. With one thread doing all CPU device
  work, a short forward goes 25 → 13 ms.
- **`Runtime.call` overhead:** 0.58 ms p50 on a 16 ms Domain call: planning
  0.22 ms (tokenization 0.11 of it), queue hops 0.08, the response 0.29.

## Reproduce

```bash
python3 tools/legacy_parity.py build-legacy --recipe cpu --tree <legacy tree> \
  --cache <hf cache> --flat <dir> --out legacy-cpu.test      # in the bindings' userland
taskset -c 96-111 python3 tools/legacy_parity.py ab --binary legacy-cpu.test \
  --tree <legacy tree> --cache <hf cache> --threads 16 --rounds 2 --concurrency 4 \
  --seconds 20 --profile batching --out ab.json
```
