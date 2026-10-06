# Vela 2.0 performance

On the exact profile (answers identical to the packages' engine,
`vela2-parity.md`) the `vela2` family is faster than the Vela 2.0 engine on
every measured row on ROCm (15–30% at the median with one caller, 1.2–1.6×
the requests per second with four). On CPU, where both run the same
operations on MKL, the 0.3B is level with it or better, the 0.8B is level or
better at 32 and 512 tokens (re-timed: the first run's 0.3–2.5% came from
where the process's memory landed), and the 4B's CPU timings are informative
only. The opt-in approximate profiles serve the
0.8B 1.6–2.4× and the 4B and 9B 1.8–2.8× faster than the engine on ROCm, the
4B 1.5× and the 0.8B 1.2× on CPU, and the 0.3B up to 1.9× the engine's
throughput with four callers. On CPU, `max_speed` runs the 0.3B on a
`float32-packed` copy of its linear layers, 1.6–1.7× faster than `exact` at
every length. One
Vela 2.0 0.3B request also replaces the seven Vela 1.0 classifier calls a
router request made: on CPU it is 2.2× faster than their sum at the median
and 10× at p95, and 3.2× at the median under `max_speed`.

- **Dates:** each section gives its own (2026-10-04 to 2026-10-06).
- **Devices:** ROCm: one AMD Instinct MI325X (gfx942) per run, 8–16 host
  cores, in the router's ROCm image for the timed A/B sections (the
  reduced-copy and Vela 1.0 comparisons ran in the Decision 2.0 release
  image); CPU: 16 cores of an AMD EPYC, PyTorch 2.10's CPU build (MKL) for
  both sides.
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

- **Date:** 2026-10-05, at `fa4baf009` (the per-device lock, graph captures
  in thread-local mode, MIOpen's FAST find mode; the 0.3B captures no graph).
- **Stack:** the router's ROCm image built from `a580be6b9`, which the PR
  ships slimmed (`31d00387c`; every file it keeps is unchanged): vLLM's ROCm PyTorch `2.12.0+git6bbd260` (AOTriton 0.13.50, ROCm
  7.2.3), Triton 3.7.0, FLA 0.5.2, `causal-conv1d` 1.7.0
  (`rocm-router-image.md`), on one AMD Instinct MI325X (gfx942), both sides in
  one process. The engine side's `transformers` 5.17.0, which the image
  doesn't carry, is mounted beside it.
- **Setup:** 16 host cores in the container's cpuset, threads capped at 16;
  node load at most 10, with the 4B and 9B runs below on two other GPUs.
- **Rounds:** 10 interleaved rounds, side order rotated per round; each cell
  is 20 warm-up requests, then 30 requests at 1 or 4 concurrent callers.
- **Reading:** `exact`, the default, is faster than the engine on every row,
  with every interval on the better side (15–28% at the median with one
  caller). `batching` (opt-in) is faster at 512 and 2,048 tokens and with 4
  callers. With 1 caller at 32–128 tokens it pays its 2 ms collection window
  (p50 +1.1 to +1.2 ms, 14–18 fewer requests per second): the opt-in
  profile's documented trade-off, never the default path.
- **Earlier stack:** the same run on the official-wheel stack (`venv-rocm72-cc`,
  at `fb67e3bb3`) read the same, with `batching` also behind at 512 × 1.

p50 ms, engine and Δ against it (mean [95% interval]):

| Tokens × callers | engine | exact | batching |
| --- | --- | --- | --- |
| 32 × 1 | 7.3 | −1.1 [−1.2, −1.0] | +1.2 [+1.1, +1.2] |
| 32 × 4 | 30.3 | −5.2 [−5.5, −4.9] | −13.0 [−13.2, −12.7] |
| 128 × 1 | 8.2 | −1.2 [−1.3, −1.2] | +1.1 [+0.9, +1.2] |
| 128 × 4 | 34.0 | −6.3 [−7.0, −5.6] | −13.1 [−14.0, −12.3] |
| 512 × 1 | 13.6 | −2.3 [−2.4, −2.2] | −0.3 [−0.4, −0.2] |
| 512 × 4 | 56.3 | −12.4 [−13.1, −11.8] | −16.3 [−17.3, −15.2] |
| 2,048 × 1 | 34.7 | −9.7 [−9.8, −9.6] | −13.0 [−13.1, −12.8] |
| 2,048 × 4 | 139.5 | −41.5 [−42.5, −40.6] | −63.7 [−65.6, −61.8] |

p95 ms:

| Tokens × callers | engine | exact | batching |
| --- | --- | --- | --- |
| 32 × 1 | 7.7 | −1.3 [−1.4, −1.1] | +1.2 [+1.0, +1.3] |
| 32 × 4 | 31.7 | −6.1 [−7.5, −4.8] | −11.9 [−14.3, −9.5] |
| 128 × 1 | 8.4 | −1.3 [−1.4, −1.2] | +1.2 [+0.9, +1.5] |
| 128 × 4 | 34.8 | −6.2 [−7.1, −5.4] | −5.4 [−8.1, −2.7] |
| 512 × 1 | 14.0 | −2.3 [−2.5, −2.1] | −0.3 [−0.4, −0.2] |
| 512 × 4 | 57.8 | −12.6 [−13.6, −11.6] | −8.7 [−9.8, −7.5] |
| 2,048 × 1 | 35.8 | −8.6 [−8.8, −8.4] | −12.5 [−14.2, −10.9] |
| 2,048 × 4 | 140.8 | −41.1 [−42.2, −39.9] | −56.8 [−59.1, −54.6] |

Requests per second:

| Tokens × callers | engine | exact | batching |
| --- | --- | --- | --- |
| 32 × 1 | 135.59 | +25.35 [+22.85, +27.84] | −18.05 [−19.46, −16.64] |
| 32 × 4 | 130.10 | +28.30 [+26.36, +30.24] | +91.34 [+87.24, +95.44] |
| 128 × 1 | 122.12 | +23.41 [+22.25, +24.56] | −13.56 [−15.46, −11.66] |
| 128 × 4 | 116.37 | +26.24 [+24.13, +28.35] | +62.99 [+59.45, +66.53] |
| 512 × 1 | 73.06 | +15.38 [+14.38, +16.38] | +1.93 [+1.50, +2.36] |
| 512 × 4 | 70.23 | +19.06 [+17.96, +20.17] | +27.66 [+25.65, +29.68] |
| 2,048 × 1 | 28.65 | +10.76 [+10.27, +11.25] | +16.89 [+16.24, +17.54] |
| 2,048 × 4 | 28.55 | +12.07 [+11.59, +12.55] | +23.10 [+22.34, +23.87] |

## Vela-2.0-0.3B on CPU

- **Date:** 2026-10-05, at `d4c6d9a50`. Later commits change no CPU forward
  or batching path (only cancellation bookkeeping, load retries and the GPU
  device lock). `a1e1b4ccb` also freezes the heap after each load pass,
  which leaves fewer objects for a request's garbage collections to walk: it
  can only shorten pauses. The huge-page default (`129be34ea`) came later
  too and changes how CPU weight copies are laid out; these rows were not
  re-timed under it (`decision1-performance.md` re-times Decision 1.0).
- **Setup:** 16 cores of an AMD EPYC 9575F in a `systemd-run` scope
  (effective cpuset logged), threads capped at 16, PyTorch 2.10's CPU build
  for both sides; node load at most 68.
- **Rounds:** 10 interleaved rounds, side order rotated per round; each cell
  is 10 warm-up requests, then 30 requests at 1 or 4 concurrent callers. The
  engine serves concurrent callers in arrival order, as its HTTP server does.
  `max_speed` runs the consented `float32-packed` copy.
- **Reading:** a row passes when its 95% interval on the difference is at or
  better than the engine. No row's interval lies on the worse side. Where
  the point estimate is worse (`exact` at 32 × 4 and 2,048 × 4, `batching`
  at 32 × 1 and its 32 × 4 p95), the interval straddles zero after the
  10-round cap: these rows are level with the engine, within the intervals
  shown.

p50 ms, engine and Δ against it (mean [95% interval]):

| Tokens × callers | engine | exact | batching | max_speed |
| --- | --- | --- | --- | --- |
| 32 × 1 | 153.3 | −1.2 [−1.9, −0.4] | +0.6 [−0.3, +1.5] | −57.1 [−59.4, −54.9] |
| 32 × 4 | 605.8 | +4.5 [−27.0, +36.0] | −45.4 [−71.5, −19.4] | −269.1 [−298.9, −239.3] |
| 128 × 1 | 175.4 | −2.8 [−3.6, −1.9] | −0.8 [−1.6, −0.0] | −66.1 [−67.2, −65.0] |
| 128 × 4 | 682.8 | −12.0 [−21.7, −2.2] | −42.1 [−65.4, −18.8] | −270.2 [−288.7, −251.7] |
| 512 × 1 | 263.7 | −4.8 [−5.9, −3.8] | −28.9 [−31.4, −26.4] | −115.1 [−119.8, −110.5] |
| 512 × 4 | 1,047.8 | −9.3 [−40.1, +21.5] | −114.2 [−155.1, −73.3] | −455.8 [−500.6, −411.1] |
| 2,048 × 1 | 785.7 | −11.1 [−12.8, −9.5] | −210.9 [−213.5, −208.2] | −414.2 [−417.2, −411.2] |
| 2,048 × 4 | 3,124.7 | +16.8 [−52.3, +85.9] | −842.3 [−891.3, −793.2] | −1,653.6 [−1,706.4, −1,600.8] |

p95 ms:

| Tokens × callers | engine | exact | batching | max_speed |
| --- | --- | --- | --- | --- |
| 32 × 1 | 166.3 | +0.9 [−5.8, +7.7] | +7.0 [−2.3, +16.4] | −54.8 [−60.5, −49.0] |
| 32 × 4 | 631.6 | +13.5 [−21.6, +48.7] | +50.7 [−23.6, +125.0] | −248.0 [−277.3, −218.7] |
| 128 × 1 | 184.7 | −2.9 [−5.3, −0.6] | −0.5 [−2.7, +1.6] | −67.4 [−70.3, −64.6] |
| 128 × 4 | 703.4 | −10.9 [−23.9, +2.1] | −36.6 [−56.0, −17.3] | −273.6 [−291.6, −255.7] |
| 512 × 1 | 282.9 | −6.9 [−9.6, −4.3] | −31.3 [−34.7, −28.0] | −123.2 [−130.7, −115.7] |
| 512 × 4 | 1,079.5 | −10.5 [−47.1, +26.1] | −116.2 [−166.5, −65.8] | −474.8 [−527.6, −421.9] |
| 2,048 × 1 | 815.8 | −14.5 [−25.0, −4.0] | −219.7 [−228.9, −210.5] | −413.5 [−426.5, −400.5] |
| 2,048 × 4 | 3,200.3 | +12.4 [−82.0, +106.7] | −852.4 [−931.5, −773.3] | −1,669.9 [−1,759.7, −1,580.0] |

Requests per second:

| Tokens × callers | engine | exact | batching | max_speed |
| --- | --- | --- | --- | --- |
| 32 × 1 | 6.47 | +0.07 [+0.03, +0.10] | −0.04 [−0.11, +0.03] | +3.76 [+3.52, +4.00] |
| 32 × 4 | 6.61 | −0.07 [−0.39, +0.26] | +0.44 [+0.13, +0.75] | +5.07 [+4.65, +5.48] |
| 128 × 1 | 5.71 | +0.08 [+0.04, +0.12] | +0.01 [−0.02, +0.03] | +3.39 [+3.26, +3.52] |
| 128 × 4 | 5.85 | +0.10 [+0.03, +0.18] | +0.37 [+0.21, +0.53] | +3.82 [+3.50, +4.14] |
| 512 × 1 | 3.76 | +0.07 [+0.05, +0.09] | +0.46 [+0.43, +0.49] | +2.96 [+2.78, +3.13] |
| 512 × 4 | 3.82 | +0.03 [−0.07, +0.13] | +0.45 [+0.30, +0.60] | +2.99 [+2.77, +3.21] |
| 2,048 × 1 | 1.27 | +0.02 [+0.02, +0.02] | +0.46 [+0.46, +0.47] | +1.39 [+1.37, +1.41] |
| 2,048 × 4 | 1.27 | −0.00 [−0.03, +0.02] | +0.47 [+0.44, +0.50] | +1.42 [+1.35, +1.48] |

- The exact path runs the engine's operations on the same MKL build, so its
  latency is the engine's or slightly better.
- On the parity requests, whose long documents run in windows with spans
  over thousands of words, the exact path took 1,476 s against the engine's
  1,836 s for 300 requests (sides alternating per request): the span
  decoding the engine runs word by word in Python is vectorized here.
- The approximate profiles run packed sequences with local layers in query
  blocks from 1,024 tokens. `batching` and `max_speed` also merge concurrent
  requests into one forward, but on CPU only up to 2,048 tokens per forward:
  measured with `vela2_bench --trace` at an earlier head, 2.7 packed
  2,048-token sequences cost 379 µs per token against 267 µs for one, and
  without the budget `batching` served 0.95 against the engine's 1.15
  requests per second at 2,048 × 4. Short sequences of several requests still
  share a forward; GPUs and the 4B / 9B have no budget.

## Vela-2.0-0.3B reduced copies (`max_speed`)

`max_speed` serves the 0.3B as `batching` does (packed sequences, 2 ms
window), on a reduced copy of the backbone's linear layers where the built-in
entry consents to one (design section 5.4). The 0.3B consents to
`float32-packed` on CPU and to no copy on GPUs: `vela2-parity.md` has the
accuracy, and `vela2-reduced.json` the raw summaries of both records.

- **Date:** 2026-10-04, at `7b76fdd29` (the IP2 staging head merged, with
  its scheduler changes). A first run at `10ca64d02` agrees within the node's
  noise; `vela2-reduced.json` has both. The CPU rows predate the heap freeze
  (`a1e1b4ccb`) and the huge-page default (`129be34ea`) and were not
  re-timed under them.
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

- **Date:** 2026-10-05, at `fa4baf009`, in the router's ROCm image as above,
  with `MIOPEN_FIND_MODE=FAST`, which the runtime sets at import (design
  section 12). The earlier runs on the official-wheel stack, under FAST and
  under MIOpen's default find mode, read the same.
- **Setup:** one MI325X per model, the 4B and the 9B side by side on two
  GPUs, 8 host cores each in the container's cpuset, threads capped at 8;
  node load at most 10. 10 interleaved rounds, side
  order rotated per round; each cell is 20 warm-up requests, then 30
  requests at 1 or 4 concurrent callers.
- **Reading:** every profile is faster than the engine on every row of both
  models, with every interval on the better side but two: the 9B's `exact`
  p95 at 32 tokens, −85 ms [−178, +8] with one caller and −549 ms
  [−1,111, +13] with 4, level to better at the 10-round cap (the same two as
  on the earlier stack). With one caller `exact` is 21–30% faster at the
  median on the 4B and 17–28% on the 9B; `shared_context` and `batching` are
  1.8–2.8× faster, and serve 1.9–2.9× the engine's requests per second with 4
  callers.

4B, p50 ms, engine and Δ against it (mean [95% interval]):

| Tokens × callers | engine | exact | shared_context | batching |
| --- | --- | --- | --- | --- |
| 32 × 1 | 115.3 | −34.1 [−34.3, −33.9] | −69.0 [−69.2, −68.7] | −67.1 [−67.5, −66.6] |
| 32 × 4 | 584.8 | −262.3 [−473.3, −51.2] | −401.5 [−612.2, −190.9] | −402.2 [−613.3, −191.0] |
| 128 × 1 | 133.2 | −35.6 [−35.9, −35.4] | −80.8 [−81.1, −80.6] | −78.7 [−79.0, −78.4] |
| 128 × 4 | 607.9 | −219.9 [−378.1, −61.6] | −399.2 [−556.9, −241.5] | −406.8 [−563.0, −250.6] |
| 512 × 1 | 202.2 | −43.7 [−44.0, −43.5] | −130.7 [−131.1, −130.4] | −128.3 [−128.7, −128.0] |
| 512 × 4 | 820.6 | −184.9 [−212.2, −157.5] | −535.7 [−562.6, −508.8] | −533.9 [−561.5, −506.3] |
| 2,048 × 1 | 410.5 | −85.7 [−86.6, −84.7] | −205.4 [−206.5, −204.4] | −204.2 [−205.2, −203.2] |
| 2,048 × 4 | 1,654.8 | −357.2 [−410.5, −303.9] | −837.0 [−889.9, −784.1] | −835.8 [−888.5, −783.0] |

4B, p95 ms:

| Tokens × callers | engine | exact | shared_context | batching |
| --- | --- | --- | --- | --- |
| 32 × 1 | 146.6 | −58.5 [−105.9, −11.0] | −99.2 [−155.9, −42.5] | −97.4 [−154.2, −40.5] |
| 32 × 4 | 657.9 | −324.5 [−611.3, −37.6] | −468.6 [−753.1, −184.1] | −440.3 [−707.9, −172.8] |
| 128 × 1 | 153.9 | −51.4 [−82.3, −20.5] | −100.5 [−140.1, −60.9] | −98.6 [−138.4, −58.8] |
| 128 × 4 | 671.7 | −276.0 [−484.7, −67.3] | −461.5 [−668.0, −254.9] | −382.3 [−594.1, −170.5] |
| 512 × 1 | 207.3 | −40.8 [−41.3, −40.3] | −133.8 [−139.2, −128.4] | −131.7 [−137.5, −126.0] |
| 512 × 4 | 847.2 | −204.3 [−277.8, −130.8] | −559.7 [−632.6, −486.9] | −503.3 [−578.0, −428.6] |
| 2,048 × 1 | 421.4 | −89.2 [−93.3, −85.1] | −214.3 [−229.7, −199.0] | −213.4 [−228.7, −198.0] |
| 2,048 × 4 | 1,709.3 | −408.2 [−543.0, −273.3] | −878.3 [−1,016.1, −740.6] | −869.4 [−1,003.7, −735.0] |

4B, requests per second:

| Tokens × callers | engine | exact | shared_context | batching |
| --- | --- | --- | --- | --- |
| 32 × 1 | 8.39 | +3.63 [+3.58, +3.68] | +13.18 [+12.70, +13.67] | +12.31 [+11.84, +12.78] |
| 32 × 4 | 7.76 | +4.54 [+3.05, +6.03] | +13.84 [+12.52, +15.17] | +13.97 [+12.53, +15.42] |
| 128 × 1 | 7.39 | +2.80 [+2.70, +2.90] | +11.66 [+11.45, +11.87] | +10.95 [+10.71, +11.19] |
| 128 × 4 | 6.85 | +3.40 [+2.35, +4.45] | +12.25 [+11.22, +13.28] | +12.17 [+11.13, +13.22] |
| 512 × 1 | 4.93 | +1.32 [+1.29, +1.34] | +8.98 [+8.94, +9.01] | +8.54 [+8.49, +8.59] |
| 512 × 4 | 4.85 | +1.42 [+1.22, +1.63] | +9.13 [+8.92, +9.34] | +9.06 [+8.85, +9.28] |
| 2,048 × 1 | 2.42 | +0.64 [+0.63, +0.66] | +2.43 [+2.42, +2.44] | +2.42 [+2.40, +2.43] |
| 2,048 × 4 | 2.41 | +0.67 [+0.58, +0.75] | +2.46 [+2.36, +2.56] | +2.44 [+2.35, +2.54] |

9B, p50 ms, engine and Δ against it (mean [95% interval]):

| Tokens × callers | engine | exact | shared_context | batching |
| --- | --- | --- | --- | --- |
| 32 × 1 | 161.3 | −44.4 [−45.1, −43.8] | −96.1 [−96.6, −95.6] | −94.1 [−94.6, −93.7] |
| 32 × 4 | 916.1 | −448.9 [−893.5, −4.3] | −660.2 [−1,104.3, −216.0] | −669.1 [−1,110.8, −227.4] |
| 128 × 1 | 191.1 | −41.3 [−41.6, −41.1] | −117.1 [−117.4, −116.8] | −115.1 [−115.5, −114.7] |
| 128 × 4 | 935.7 | −338.7 [−613.2, −64.2] | −641.8 [−915.6, −368.0] | −649.0 [−921.0, −377.0] |
| 512 × 1 | 278.3 | −46.6 [−46.9, −46.3] | −176.0 [−176.3, −175.7] | −173.8 [−174.1, −173.6] |
| 512 × 4 | 1,223.0 | −295.5 [−512.4, −78.7] | −814.8 [−1,031.4, −598.1] | −810.3 [−1,028.4, −592.1] |
| 2,048 × 1 | 561.9 | −95.8 [−96.6, −95.0] | −259.3 [−259.9, −258.7] | −257.3 [−258.0, −256.6] |
| 2,048 × 4 | 2,348.8 | −491.9 [−650.4, −333.4] | −1,139.8 [−1,297.6, −982.1] | −1,142.5 [−1,301.5, −983.5] |

9B, p95 ms:

| Tokens × callers | engine | exact | shared_context | batching |
| --- | --- | --- | --- | --- |
| 32 × 1 | 213.0 | −85.1 [−178.3, +8.0] | −146.6 [−252.4, −40.8] | −144.6 [−250.4, −38.8] |
| 32 × 4 | 1,034.0 | −548.6 [−1,110.8, +13.5] | −772.9 [−1,334.7, −211.2] | −722.6 [−1,258.0, −187.3] |
| 128 × 1 | 230.2 | −73.6 [−143.2, −4.0] | −155.4 [−235.8, −75.1] | −153.6 [−233.9, −73.2] |
| 128 × 4 | 1,003.3 | −402.4 [−758.3, −46.5] | −707.1 [−1,062.2, −352.1] | −620.8 [−1,000.2, −241.4] |
| 512 × 1 | 317.1 | −77.5 [−140.8, −14.2] | −212.6 [−281.9, −143.3] | −210.9 [−281.5, −140.2] |
| 512 × 4 | 1,301.2 | −362.7 [−663.6, −61.8] | −885.2 [−1,186.3, −584.0] | −811.6 [−1,131.4, −491.8] |
| 2,048 × 1 | 569.2 | −94.6 [−97.5, −91.8] | −263.1 [−271.7, −254.5] | −261.8 [−272.2, −251.3] |
| 2,048 × 4 | 2,506.1 | −627.8 [−1,019.2, −236.4] | −1,283.5 [−1,675.7, −891.3] | −1,293.3 [−1,683.7, −903.0] |

9B, requests per second:

| Tokens × callers | engine | exact | shared_context | batching |
| --- | --- | --- | --- | --- |
| 32 × 1 | 5.96 | +2.42 [+2.10, +2.74] | +9.35 [+8.84, +9.85] | +8.90 [+8.39, +9.41] |
| 32 × 4 | 5.38 | +3.10 [+1.87, +4.32] | +10.14 [+8.92, +11.36] | +10.13 [+8.91, +11.35] |
| 128 × 1 | 5.12 | +1.51 [+1.35, +1.66] | +8.36 [+8.10, +8.61] | +8.01 [+7.75, +8.27] |
| 128 × 4 | 4.69 | +2.00 [+1.18, +2.83] | +8.89 [+8.08, +9.70] | +8.87 [+8.04, +9.70] |
| 512 × 1 | 3.53 | +0.76 [+0.68, +0.84] | +6.21 [+6.17, +6.25] | +6.01 [+5.95, +6.08] |
| 512 × 4 | 3.36 | +0.94 [+0.57, +1.31] | +6.40 [+6.02, +6.78] | +6.40 [+6.02, +6.78] |
| 2,048 × 1 | 1.77 | +0.36 [+0.35, +0.37] | +1.52 [+1.51, +1.53] | +1.51 [+1.49, +1.52] |
| 2,048 × 4 | 1.70 | +0.45 [+0.33, +0.56] | +1.60 [+1.49, +1.71] | +1.61 [+1.50, +1.72] |

- With one caller every side's p95 stays within 1.2× of its p50, except the
  engine's at 32 tokens (1.27× and 1.31×): a tree shape seen for the first
  time pays a one-time setup
  (kernel specialization and library algorithm choices), and prompt lengths
  vary within a cell.
- With 4 callers the engine serves one request at a time in arrival order,
  so its p50 is about four service times; `exact` runs one request per
  forward as the engine does, and the approximate profiles run the packed
  trees, `batching` also several requests per forward.

4B on CPU (16 cores, 8 requests per length, concurrency 1, seconds;
informative: no rounds or intervals, no verdict):

| Tokens | exact | shared_context | engine |
| --- | --- | --- | --- |
| 128 | 30.1 | 19.9 | 30.4 |
| 512 | 55.7 | 36.4 | 57.3 |

## Vela-2.0-0.8B on ROCm

- **Date:** 2026-10-06, at `bd9cbeeb9` (staging `d9fbd0115` with the 0.8B's
  registry entry), in the router's ROCm image as above, with
  `MIOPEN_FIND_MODE=FAST`. The heap freeze after each load pass
  (`a1e1b4ccb`) came later; it can only shorten a request's garbage
  collection pauses.
- **Setup:** one MI325X, 8 host cores in the container's cpuset, threads
  capped at 8; node load at most 18 (other workloads on the node's other GPUs
  and cores). 10 interleaved rounds, side order rotated per round; each cell
  is 20 warm-up requests, then 30 requests at 1 or 4 concurrent callers. A
  first run read the same except for single slow rounds while models loaded
  on another GPU; this run replaces it (`vela2-reduced.json` keeps both).
- **Reading:** every profile is faster than the engine on every row, with
  every interval on the better side. With one caller `exact` is 22–30%
  faster at the median, and `shared_context` and `batching` are 1.6–2.3×
  faster; with 4 callers they serve 1.35–1.45× (`exact`) and 1.8–2.4× the
  engine's requests per second.

0.8B, p50 ms, engine and Δ against it (mean [95% interval]):

| Tokens × callers | engine | exact | shared_context | batching |
| --- | --- | --- | --- | --- |
| 32 × 1 | 54.3 | −11.8 [−12.4, −11.2] | −22.3 [−22.8, −21.8] | −21.3 [−21.8, −20.8] |
| 32 × 4 | 227.1 | −59.9 [−80.6, −39.2] | −101.6 [−122.5, −80.7] | −107.2 [−128.7, −85.8] |
| 128 × 1 | 56.1 | −12.7 [−13.2, −12.2] | −22.8 [−23.1, −22.5] | −22.6 [−23.1, −22.1] |
| 128 × 4 | 233.0 | −61.3 [−76.0, −46.6] | −103.9 [−118.2, −89.7] | −115.2 [−129.1, −101.4] |
| 512 × 1 | 78.5 | −23.5 [−24.0, −23.0] | −44.3 [−44.7, −43.9] | −44.0 [−44.4, −43.7] |
| 512 × 4 | 317.6 | −99.9 [−111.3, −88.6] | −183.9 [−195.1, −172.7] | −187.1 [−198.6, −175.5] |
| 2,048 × 1 | 168.2 | −50.6 [−51.1, −50.1] | −72.3 [−72.6, −72.0] | −72.9 [−73.5, −72.4] |
| 2,048 × 4 | 664.7 | −198.5 [−220.0, −177.0] | −286.9 [−308.0, −265.8] | −290.6 [−313.4, −267.8] |

0.8B, p95 ms:

| Tokens × callers | engine | exact | shared_context | batching |
| --- | --- | --- | --- | --- |
| 32 × 1 | 58.3 | −12.9 [−15.0, −10.8] | −25.3 [−31.3, −19.2] | −24.4 [−30.4, −18.4] |
| 32 × 4 | 247.4 | −78.1 [−122.7, −33.4] | −120.2 [−165.5, −75.0] | −79.3 [−123.9, −34.8] |
| 128 × 1 | 60.2 | −14.1 [−16.1, −12.0] | −26.0 [−31.1, −20.8] | −25.8 [−31.3, −20.3] |
| 128 × 4 | 239.6 | −64.5 [−83.3, −45.6] | −107.6 [−125.2, −90.1] | −55.5 [−77.9, −33.2] |
| 512 × 1 | 83.1 | −25.5 [−28.4, −22.6] | −47.8 [−52.6, −42.9] | −47.5 [−52.7, −42.4] |
| 512 × 4 | 322.4 | −102.5 [−116.9, −88.0] | −187.2 [−201.5, −173.0] | −160.4 [−174.7, −146.0] |
| 2,048 × 1 | 172.5 | −51.6 [−52.9, −50.3] | −73.5 [−78.4, −68.5] | −75.6 [−80.6, −70.6] |
| 2,048 × 4 | 684.6 | −214.9 [−251.0, −178.7] | −302.4 [−338.0, −266.7] | −301.8 [−337.9, −265.6] |

0.8B, requests per second:

| Tokens × callers | engine | exact | shared_context | batching |
| --- | --- | --- | --- | --- |
| 32 × 1 | 18.01 | +5.19 [+4.85, +5.53] | +13.01 [+12.48, +13.54] | +12.14 [+11.47, +12.80] |
| 32 × 4 | 17.61 | +6.21 [+4.73, +7.68] | +14.07 [+12.56, +15.59] | +14.44 [+12.85, +16.03] |
| 128 × 1 | 17.66 | +5.03 [+4.79, +5.26] | +12.28 [+11.95, +12.61] | +12.05 [+11.45, +12.66] |
| 128 × 4 | 17.21 | +5.94 [+4.94, +6.94] | +13.46 [+12.55, +14.37] | +14.44 [+13.49, +15.39] |
| 512 × 1 | 12.63 | +5.37 [+5.24, +5.50] | +16.49 [+16.32, +16.66] | +16.23 [+15.94, +16.53] |
| 512 × 4 | 12.58 | +5.72 [+5.25, +6.18] | +17.09 [+16.60, +17.57] | +18.00 [+17.45, +18.55] |
| 2,048 × 1 | 5.91 | +2.50 [+2.37, +2.62] | +4.46 [+4.39, +4.53] | +4.56 [+4.48, +4.64] |
| 2,048 × 4 | 5.99 | +2.57 [+2.37, +2.77] | +4.55 [+4.36, +4.74] | +4.65 [+4.42, +4.88] |

## Vela-2.0-0.8B on CPU

- **Date:** 2026-10-06, at `bd9cbeeb9`.
- **Setup:** 16 cores of an AMD EPYC 9575F on one NUMA node, in a
  `systemd-run` scope (effective cpuset logged), threads capped at 16,
  PyTorch 2.10's CPU build for both sides; node load at most 19. A decoder
  request takes 4–18 s on CPU, so the grid is smaller than on ROCm: one
  caller; at 32–512 tokens 2 warm-up requests, then 6 per round over 5
  interleaved rounds; at 2,048 tokens 1 warm-up request, then 3 per round
  over 3 rounds.
- **Reading:** `shared_context` and `batching` are 15–19% faster than the
  engine at the median at every length, with every interval on the better
  side. In this run `exact` is level at 128 tokens and slightly slower at 32
  and 512 (+2.5% [+1.6, +3.5] and +0.3% [+0.1, +0.5] at the median), with
  those intervals on the worse side. The re-time below finds `exact` level
  or better at both lengths and shows where that difference came from. The
  2,048-token row
  has 3 rounds, below the standard's 5, and gives no verdict; with 6
  requests per round no row reports a p95. The 0.3B is the size to serve on
  a CPU.
- **Later commits:** `a1e1b4ccb` freezes the heap after each load pass,
  which leaves fewer objects for a request's garbage collections to walk: it
  can only shorten pauses. The huge-page default (`129be34ea`) changes how
  CPU weight copies are laid out; the re-time below runs under it.

0.8B on CPU, p50 ms, engine and Δ against it (mean [95% interval]):

| Tokens | engine | exact | shared_context | batching |
| --- | --- | --- | --- | --- |
| 32 | 4,614 | +117 [+73, +162] | −786 [−891, −682] | −785 [−844, −726] |
| 128 | 6,619 | +76 [−178, +330] | −992 [−1,219, −766] | −1,042 [−1,313, −771] |
| 512 | 10,587 | +33 [+8, +58] | −2,032 [−2,127, −1,938] | −1,878 [−2,218, −1,539] |
| 2,048 | 17,288 | +553 [−844, +1,949] | −3,108 [−5,550, −667] | −3,121 [−5,964, −278] |

0.8B on CPU, requests per second:

| Tokens | engine | exact | shared_context | batching |
| --- | --- | --- | --- | --- |
| 32 | 0.216 | −0.005 [−0.007, −0.004] | +0.044 [+0.042, +0.046] | +0.044 [+0.041, +0.047] |
| 128 | 0.148 | +0.001 [−0.003, +0.004] | +0.029 [+0.023, +0.035] | +0.028 [+0.024, +0.033] |
| 512 | 0.094 | −0.001 [−0.001, +0.000] | +0.023 [+0.020, +0.025] | +0.022 [+0.019, +0.024] |
| 2,048 | 0.057 | −0.001 [−0.003, +0.000] | +0.014 [+0.002, +0.027] | +0.014 [+0.000, +0.028] |

### Vela-2.0-0.8B `exact` on CPU, re-timed

- **Date:** 2026-10-06, at `0d8d76cec` (with the huge-page default and the
  heap freeze; no other change to the CPU request path since the first run).
- **Setup:** the first run's tool and node (`tools/vela2_bench.py --sides
  reference,runtime`, both sides in one process, every forward on the CPU
  device thread, 16 threads, a `systemd-run` scope), 32 and 512 tokens, 2
  warm-up requests, then 6 per round over 10 interleaved rounds. Two
  placements:
  - the first run's cores (128–143, NUMA node 1), memory unbound: about a
    third of the process's memory landed on node 0;
  - cores 32–47 with `numactl --membind=0`: every page local.

  Each placement at libgomp's default spin count (300,000, which
  `vllm-srun serve` keeps for a process without ONNX Runtime models) and at
  10,000. Node load at most 16 in the local rounds, and 33 in the unbound
  ones, where another workstream's job ran on node 0.

p50 ms of the engine, and runtime − engine (mean [95% interval]):

| Memory | Spin | Tokens | engine | p50 Δ | p95 Δ | req/s Δ |
| --- | --- | --- | --- | --- | --- | --- |
| local | 300,000 | 32 | 3,309 | −28.3 [−34.7, −21.9] | −44.7 [−54.5, −34.9] | +0.003 [+0.003, +0.004] |
| local | 300,000 | 512 | 6,701 | −16.5 [−28.7, −4.4] | −4.8 [−17.3, +7.8] | +0.001 [+0.000, +0.001] |
| local | 10,000 | 32 | 3,348 | −15.5 [−29.3, −1.6] | −19.4 [−27.7, −11.1] | +0.001 [+0.001, +0.002] |
| local | 10,000 | 512 | 6,791 | −26.6 [−37.6, −15.6] | −71.7 [−91.2, −52.1] | +0.001 [+0.000, +0.001] |
| unbound | 300,000 | 32 | 5,297 | −2.6 [−18.5, +13.2] | −56.3 [−96.4, −16.3] | +0.000 [−0.000, +0.001] |
| unbound | 300,000 | 512 | 10,907 | −169.7 [−185.4, −154.1] | −127.4 [−155.6, −99.1] | +0.001 [+0.001, +0.001] |
| unbound | 10,000 | 32 | 5,711 | −115.9 [−138.1, −93.7] | −212.9 [−260.1, −165.7] | +0.005 [+0.004, +0.005] |
| unbound | 10,000 | 512 | 11,037 | −193.5 [−221.3, −165.7] | −211.0 [−236.1, −185.9] | +0.001 [+0.001, +0.002] |

- **Every cell is level or better.** On local memory `exact` is 0.2–0.9%
  faster than the engine at the median, at both lengths and both spin
  counts. The two level cells (unbound 32-token p50 and local 512-token p95,
  both at 300,000) have the better point estimate.
- **The same work.** Profiled on the device thread (torch.profiler, two
  requests per side and length, local memory), both sides run the same
  operations per request:
  - 762 matrix products (`aten::mm`, 4.0 s of the 6.3 s of operator time at
    32 tokens), 144 triangular solves and 1,516 batched products;
  - the engine makes 24 more copies;
  - the runtime's operator time is 2.0% below the engine's at 32 tokens and
    0.1% below it at 512, with the matrix products within 0.7%.

  Around the forward the runtime adds 1.6 ms at 32 tokens and 3.3 ms at
  512: planning (tokenization and rendering) 0.7 and 1.9 ms, the hand-offs to
  the device thread 0.4 ms, and the answers 0.6 and 0.9 ms.
- **Where the first run's difference came from: memory placement.** Node B's
  NUMA node 1 has little free memory beside its page cache, so a process on
  its cores takes part of its memory from node 0, and each process places
  the two sides' weights differently.
  - In the unbound profiles all of the runtime's weights (it loads first)
    were on node 1, and 2.0 of the engine's 3.2 GiB on node 0.
  - In that process the engine was 0.9% slower at 32 tokens and 1.4% slower
    at 512.
  - The first run also predates the huge-page default, without which the
    runtime's weight copies landed on 4 KiB pages differently in every
    process (`decision1-performance.md`).

  The forward is the engine's, operation for operation, so the cell stays
  closed: what separates the sides by a few percent is where a process's
  memory lands. `vela2-reduced.json` (`latency.cpu_08b_retime`) has the
  intervals and the placements.

## Against the Vela 1.0 path

Vela 1.0 answered the same router signals with seven classifier calls per
request (domain, jailbreak guard, PII, fact-check, feedback, modality,
safety; one ModernBERT 307M model each). `vela1` timed every call through
the router's legacy runtime at `61aa7eb2d` (`tools/legacy_parity.py legacy`)
on its 547-input corpus: the repository's E2E prompts plus long inputs
derived from them. The Vela 2.0 0.3B answers the same inputs, one request
each, at concurrency 1 (`tools/vela2_bench.py --prompts`); the legacy column
is the sum of a prompt's seven calls. One run per side: informative, no
verdict.

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
