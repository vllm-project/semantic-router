# Vela 2.0 performance

On the exact profile (answers identical to the packages' engine,
`vela2-parity.md`) the `vela2` family is faster than the Vela 2.0 engine at
every measured length on ROCm (by 7–49% at the median) and equal to it on
CPU, where both run the same operations on MKL. The opt-in approximate
profiles serve the 4B and 9B 2–4× faster than the engine on ROCm and 1.5×
on CPU, and the 0.3B up to 28% faster. On CPU, `max_speed` runs the 0.3B on
a `float32-packed` copy of its linear layers, 1.6–1.7× faster than `exact`
at every length. One Vela 2.0 0.3B request also replaces the seven Vela 1.0
classifier calls a router request made: on CPU it is 2.2× faster than their
sum at the median and 10× at p95, and 3.2× at the median under `max_speed`.

- **Date:** 2026-10-04.
- **Devices:** one AMD Instinct MI325X (gfx942) per run, in the Decision 2.0
  release image, 8 host cores; CPU: 16 cores of an AMD EPYC (`taskset`), with
  PyTorch 2.10's CPU build (MKL) for both sides.
- **Engine side:** the package's `vela2_inference.py` in the same process,
  one request at a time as its server serves them (presets expanded to the
  questions they stand for).
- **Runtime side:** the family through the runtime's scheduler and profiles:
  `exact`, `shared_context`, and `batching` (2 ms window); FLA kernel choices
  pinned on both sides.
- **Workload** (`tools/vela2_bench.py`): the router's signals as one request
  — domain over 14 subject areas, jailbreak, PII spans (the `pii` preset),
  fact-check, feedback, modality and an 8-label safety set — over a prompt of
  about N tokens, a different prompt per request. Per side and length: 20
  warm-up requests (10 on CPU), then 30 at concurrency 1 (latency p50 / p95,
  ms) and 30 at concurrency 4 (throughput, requests per second).

## Vela-2.0-0.3B on ROCm

| Tokens | exact p50 / p95 | batching p50 / p95 | engine p50 / p95 |
| --- | --- | --- | --- |
| 32 | 8.1 / 8.6 | 10.5 / 11.3 | 8.7 / 9.2 |
| 128 | 8.3 / 8.7 | 11.2 / 11.7 | 9.3 / 9.8 |
| 512 | 13.4 / 13.9 | 13.9 / 14.8 | 15.7 / 16.2 |
| 2,048 | 29.4 / 32.1 | 25.1 / 26.9 | 35.0 / 36.1 |

| Tokens | exact req/s | batching req/s | engine req/s |
| --- | --- | --- | --- |
| 32 | 120.6 | 132.4 | 123.0 |
| 128 | 107.3 | 135.2 | 109.0 |
| 512 | 72.4 | 89.5 | 63.3 |
| 2,048 | 34.0 | 39.6 | 28.2 |

## Vela-2.0-0.3B on CPU

| Tokens | exact p50 / p95 | batching p50 / p95 | engine p50 / p95 |
| --- | --- | --- | --- |
| 32 | 123.4 / 125.2 | 125.6 / 127.5 | 124.5 / 126.4 |
| 128 | 142.3 / 143.9 | 144.6 / 146.5 | 143.8 / 145.8 |
| 512 | 229.8 / 232.9 | 211.6 / 213.9 | 232.8 / 236.4 |
| 2,048 | 882.6 / 922.7 | 712.1 / 738.4 | 875.6 / 912.1 |

| Tokens | exact req/s | batching req/s | engine req/s |
| --- | --- | --- | --- |
| 32 | 8.11 | 8.36 | 8.04 |
| 128 | 7.01 | 6.96 | 6.93 |
| 512 | 4.34 | 3.21 | 4.28 |
| 2,048 | 1.15 | 1.21 | 1.11 |

- The exact path runs the engine's operations on the same MKL build, and its
  latency is the engine's within 1%: the sides alternate per request, so the
  machine's drift reaches both alike (separate runs on this shared node vary
  by up to 25%).
- On the parity requests, whose long documents run in windows with spans
  over thousands of words, the exact path took 1,476 s against the engine's
  1,836 s for 300 requests (sides alternating per request): the span
  decoding the engine runs word by word in Python is vectorized here.
- The approximate profiles run packed sequences with local layers in query
  blocks from 1,024 tokens: 9% faster per request at 512 prompt tokens, 19% at
  2,048. `batching` (and `max_speed`) also merge concurrent requests, which a
  CPU does not run faster (one request already keeps the 16 cores busy) and
  which costs a quarter of the throughput at 512 tokens.

## Vela-2.0-0.3B reduced copies (`max_speed`)

`max_speed` serves the 0.3B as `batching` does (packed sequences, 2 ms
window), on a reduced copy of the backbone's linear layers where the built-in
entry consents to one (design section 5.4). The 0.3B consents to
`float32-packed` on CPU and to no copy on GPUs: `vela2-parity.md` has the
accuracy, and `vela2-reduced.json` the raw summaries of both records.

- **Date:** 2026-10-04, at `7b76fdd29` (the IP2 staging head merged, with
  its scheduler changes). A first run at `10ca64d02` agrees within the node's
  noise; `vela2-reduced.json` has both.
- **Sides:** `exact` and `batching` on one model without a copy, and one
  `max_speed` model per copy kind (consented for the run), all in one process
  (`tools/vela2_bench.py --sides runtime,runtime-batching,max_speed:KIND`).
  Same workload, warm-up and request counts as above; at concurrency 1 the
  sides take turns per request.
- **CPU:** 16 cores of an AMD EPYC with AVX-512 BF16 (no AMX), PyTorch 2.10's
  CPU build (MKL, oneDNN), node load 15–42. **ROCm:** one MI325X, 16 host
  cores, release image.

CPU, p50 / p95 ms at concurrency 1:

| Tokens | exact | batching | `float32-packed` | `bfloat16` | `int8` |
| --- | --- | --- | --- | --- | --- |
| 32 | 128.2 / 132.8 | 130.4 / 137.6 | 79.8 / 83.6 | 72.2 / 74.6 | 66.5 / 69.1 |
| 128 | 146.5 / 153.7 | 149.7 / 160.4 | 91.3 / 99.0 | 80.3 / 85.3 | 75.5 / 77.6 |
| 512 | 253.0 / 272.2 | 231.3 / 244.2 | 145.8 / 153.0 | 129.3 / 137.2 | 119.7 / 126.4 |
| 2,048 | 963.5 / 1,052.5 | 808.3 / 881.6 | 595.6 / 681.4 | 385.2 / 413.2 | 563.8 / 619.8 |

CPU, requests per second at concurrency 4:

| Tokens | exact | batching | `float32-packed` | `bfloat16` | `int8` |
| --- | --- | --- | --- | --- | --- |
| 32 | 7.76 | 9.23 | 15.05 | 16.20 | 18.47 |
| 128 | 6.82 | 7.28 | 12.10 | 13.87 | 15.07 |
| 512 | 3.90 | 3.36 | 4.46 | 6.76 | 4.76 |
| 2,048 | 1.07 | 1.08 | 1.41 | 1.97 | 1.27 |

ROCm, p50 / p95 ms at concurrency 1 and requests per second at concurrency 4:

| Tokens | exact | batching | `bfloat16` | exact req/s | batching req/s | `bfloat16` req/s |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 7.4 / 8.0 | 9.0 / 9.8 | 10.6 / 15.1 | 151.4 | 157.2 | 286.3 |
| 128 | 8.2 / 8.7 | 9.8 / 10.7 | 10.4 / 11.2 | 130.1 | 160.7 | 229.3 |
| 512 | 12.2 / 12.8 | 14.6 / 15.4 | 12.4 / 13.3 | 79.8 | 91.5 | 123.1 |
| 2,048 | 27.3 / 30.2 | 24.4 / 26.4 | 17.7 / 19.8 | 36.6 | 48.0 | 78.2 |

- **CPU, `float32-packed`:** oneDNN's pre-packed FP32 linear (weights
  reordered once at load) is 1.5–1.8× faster than `exact` at concurrency 1,
  on p50 and p95 at every length (1.6–1.7× at the median), and gives
  1.1–1.9× its throughput at concurrency 4. It also beats `batching`, the
  same packed path on MKL's `F.linear`, on every row. At concurrency 4
  `max_speed` merges concurrent requests as `batching` does, which widens the
  latency spread: at 512 tokens its p95 was 1,142 ms against `exact`'s
  1,041 ms in this run, and 998 against 1,122 ms in the first. Its answers
  stay within about 1e-5 of `exact`.
- **CPU, `bfloat16` and `int8`:** faster still on this CPU (BF16 2.5× at
  2,048 tokens), but `int8` fails the accuracy floor; `vela2-parity.md`
  explains why BF16 gets no consent.
- **ROCm, `bfloat16`:** faster at 2,048 tokens and in throughput, but a
  single request up to 512 tokens is slower than `exact` (the batching window
  and the packed path on a launch-bound forward), and it fails the accuracy
  floor on spans, so GPUs load no copy.

## Vela-2.0-4B and 9B on ROCm

4B:

| Tokens | exact p50 / p95 | shared_context p50 / p95 | batching p50 / p95 | engine p50 / p95 |
| --- | --- | --- | --- | --- |
| 32 | 88.8 / 146.8 | 56.8 / 71.3 | 52.9 / 59.7 | 156.1 / 176.1 |
| 128 | 103.8 / 163.4 | 56.9 / 58.5 | 57.1 / 58.2 | 137.7 / 174.4 |
| 512 | 163.1 / 299.7 | 80.3 / 85.3 | 79.9 / 81.7 | 318.4 / 409.9 |
| 2,048 | 344.1 / 530.2 | 230.2 / 273.1 | 224.1 / 236.4 | 459.6 / 518.9 |

| Tokens | exact req/s | shared_context req/s | batching req/s | engine req/s |
| --- | --- | --- | --- | --- |
| 32 | 11.4 | 18.6 | 16.0 | 5.2 |
| 128 | 9.7 | 18.2 | 18.2 | 6.1 |
| 512 | 6.0 | 12.7 | 12.5 | 4.4 |
| 2,048 | 2.9 | 4.4 | 4.5 | 2.1 |

9B:

| Tokens | exact p50 / p95 | shared_context p50 / p95 | batching p50 / p95 | engine p50 / p95 |
| --- | --- | --- | --- | --- |
| 32 | 123.3 / 180.0 | 70.0 / 71.5 | 69.8 / 72.8 | 217.7 / 865.0 |
| 128 | 156.6 / 219.5 | 109.5 / 170.2 | 80.7 / 81.6 | 198.8 / 711.1 |
| 512 | 238.6 / 378.6 | 110.1 / 112.1 | 109.3 / 110.9 | 290.4 / 794.8 |
| 2,048 | 477.1 / 670.0 | 316.2 / 322.4 | 313.5 / 319.1 | 632.1 / 1176.3 |

| Tokens | exact req/s | shared_context req/s | batching req/s | engine req/s |
| --- | --- | --- | --- | --- |
| 32 | 8.4 | 14.9 | 15.0 | 2.8 |
| 128 | 6.1 | 10.1 | 12.7 | 3.1 |
| 512 | 4.2 | 9.1 | 9.3 | 2.3 |
| 2,048 | 2.1 | 2.5 | 3.2 | 1.4 |

- The exact path's p95 is 1.4–1.8× its p50: a tree shape seen for the first
  time pays a one-time setup (about 50 ms on the 4B: kernel specialization
  and library algorithm choices), and every prompt length is a new shape. The
  engine pays the same on its own shapes, plus its lock at concurrency 4.
- Throughput at concurrency 4 is the scheduler's: `exact` runs one request
  per forward, as the engine does; the approximate profiles run the packed
  trees, `batching` also several requests per forward.
- Measured at `f08c63a14`; later commits change no forward on these paths
  (the forest forward stopped reading its shape back from the device).

4B on CPU (16 cores, 8 requests per length, concurrency 1, seconds):

| Tokens | exact | shared_context | engine |
| --- | --- | --- | --- |
| 128 | 30.1 | 19.9 | 30.4 |
| 512 | 55.7 | 36.4 | 57.3 |

## Against the Vela 1.0 path

Vela 1.0 answered the same router signals with seven classifier calls per
request (domain, jailbreak guard, PII, fact-check, feedback, modality,
safety; one ModernBERT 307M model each). `vela1` timed every call through
the router's legacy runtime at `61aa7eb2d` (`tools/legacy_parity.py legacy`)
on its 547-input corpus: the repository's E2E prompts plus long inputs
derived from them. The Vela 2.0 0.3B answers the same inputs, one request
each, at concurrency 1 (`tools/vela2_bench.py --prompts`); the legacy column
is the sum of a prompt's seven calls.

CPU (the legacy candle path, 16 cores; the 545 inputs on which all seven
legacy calls succeeded):

| | p50 | p95 | mean | req/s |
| --- | --- | --- | --- | --- |
| Vela 1.0, seven calls | 252.9 | 1,445.9 | 754.5 | 1.33 |
| Vela 2.0 0.3B, exact | 117.6 | 141.7 | 141.1 | 7.09 |
| Vela 2.0 0.3B engine | 118.4 | 143.6 | 141.9 | 7.05 |
| Vela 2.0 0.3B, `max_speed` (`float32-packed`) | 82.9 | 107.7 | 96.5 | 10.37 |

Per input, the Vela 2.0 request is 2.2× faster than the seven calls at the
median, and 3.2× under `max_speed`. Long inputs gain the most: the legacy
jailbreak and PII models read them in 512-token windows that overlap by half
(the other five cut them at 512 tokens), where the Vela 2.0 request reads
them once. The `max_speed` row comes from a later run at `7b76fdd29` that
interleaved it with `exact` on the same inputs; `exact` measured 134.5 /
168.2 / 157.6 ms and 6.34 requests per second there (the node was about 10%
slower than in the first run).

ROCm (the AMD recipe's ONNX Runtime path, MIGraphX; the 360 inputs on which
all seven legacy calls succeeded):

| | p50 | p95 | mean | req/s |
| --- | --- | --- | --- | --- |
| Vela 1.0, seven calls | 1,170.1 | 1,232.6 | 1,177.7 | 0.8 |
| Vela 2.0 0.3B, exact | 7.2 | 8.5 | 7.4 | 135.4 |
| Vela 2.0 0.3B engine | 8.9 | 10.1 | 9.0 | 111.6 |

The legacy ROCm path compiles a MIGraphX program per input shape, which
dominates its per-call latency (134–247 ms at the median); the ratio says
more about that path than about the models.

## How

- **One forward per request.** The 0.3B packs every question, option and
  label into one marker sequence, and the 4B / 9B compute the
  request's parts once with every question as its own block continuing from
  them (the shared-context tree), where Vela 1.0 ran one model per signal.
- **The engine's shapes, fused.** The exact 4B / 9B forward keeps the
  engine's tensor shapes (left-padded prefix rows, right-padded blocks, one
  attention call per block) so it rounds identically, and runs its
  element-wise steps as the gfx942 fused kernels, which round exactly like
  them.
- **Packed trees and sequences.** The approximate profiles pack the 4B / 9B
  trees into one row per prefix (blocks merged by log-sum-exp, no block
  padding) and the 0.3B sequences back to back (no padding; local layers in
  query blocks on long rows); `vela2-parity.md` records their accuracy.
- **A reduced copy where the records allow one.** Under `max_speed` the
  0.3B's packed batches run on a `float32-packed` copy of its linear layers on
  CPU (oneDNN's pre-packed FP32 GEMMs), while `exact` keeps its own linear
  weights; the copy shares every other parameter with it.
- **Graphs, measured and left out.** A graph per exact 4B tree shape replays
  in 1–2 ms less than the 80–165 ms eager forward at 32–512 prompt tokens
  (the forward is GPU-bound), while capturing a shape costs about four
  forwards, so trees run eagerly. The 0.3B sequences are at least 560 tokens
  with the schema, above the widths where `vela1`'s encoder graphs pay off.
- **Pinned kernel choices.** The 4B / 9B FLA kernels run the recorded
  configurations (`registry/kernel_choices.json`) instead of autotuning:
  no tuning stalls on new shapes, and the same numerics in every process.
- **Host work.** Word units and span decoding are vectorized (a 2,048-token
  PII span read went from 5.0 to 0.4 ms); question, option and label token
  IDs are cached across requests; on CPU every forward runs on the process's
  one device thread.
