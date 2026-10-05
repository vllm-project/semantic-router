# Vela 2.0 performance

On the exact profile (answers identical to the packages' engine,
`vela2-parity.md`) the `vela2` family is faster than the Vela 2.0 engine on
every measured row on ROCm (9–30% at the median with one caller, 1.2–1.6×
the requests per second with four) and level with it or better on CPU,
where both run the same operations on MKL. The opt-in approximate profiles
serve the 4B and 9B 1.8–2.8× faster than the engine on ROCm and 1.5× on
CPU, and the 0.3B up to 1.9× the engine's throughput with four callers. On
CPU, `max_speed` runs the 0.3B on a `float32-packed` copy of its linear
layers, 1.6–1.7× faster than `exact` at every length. One Vela 2.0 0.3B
request also replaces the seven Vela 1.0
classifier calls a router request made: on CPU it is 2.2× faster than their
sum at the median and 10× at p95, and 3.2× at the median under `max_speed`.

- **Dates:** each section gives its own (2026-10-04 or 2026-10-05).
- **Devices:** ROCm: one AMD Instinct MI325X (gfx942) per run, 8–16 host
  cores, on the router image's stack for the timed A/B sections (the
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

- **Date:** 2026-10-05, at `fb67e3bb3` (staging `be7366c49` merged: the
  per-device lock, and graph captures in thread-local mode; the 0.3B captures
  no graph).
- **Stack:** the router image's, stack B plus `causal-conv1d`: PyTorch
  `2.12.0+rocm7.2`, Triton 3.7.0, FLA 0.5.2, `causal-conv1d` 1.7.0 and the
  rest of `extproc-rocm`'s freeze (`venv-rocm72-cc`, manifest
  `875eeb85865e`), on one AMD Instinct MI325X (gfx942), both sides in one
  process.
- **Setup:** 16 host cores in a `systemd-run` scope (effective cpuset logged),
  threads capped at 16; node load at most 17.
- **Rounds:** 10 interleaved rounds, side order rotated per round; each cell
  is 20 warm-up requests, then 30 requests at 1 or 4 concurrent callers.
- **Reading:** `exact`, the default, is faster than the engine on every row,
  with every interval on the better side. `batching` (opt-in) is faster at
  2,048 tokens and with 4 callers. With 1 caller at 32–512 tokens it pays its
  2 ms collection window (p50 +0.3 to +1.2 ms, 1–13 fewer requests per
  second): the opt-in profile's documented trade-off, never the default path.
- **Replicate:** the same run on stack B without `causal-conv1d`
  (`venv-rocm72`, at `a9b9f2818`, the same GPU code; ModernBERT has no causal
  convolution) reads the same: `exact` faster on every row, `batching` behind
  with 1 caller at 32–128 tokens only (`vela2-reduced.json`).

p50 ms, engine and Δ against it (mean [95% interval]):

| Tokens × callers | engine | exact | batching |
| --- | --- | --- | --- |
| 32 × 1 | 9.0 | −0.8 [−1.0, −0.6] | +1.2 [+1.1, +1.4] |
| 32 × 4 | 37.1 | −7.2 [−7.9, −6.6] | −19.6 [−20.1, −19.1] |
| 128 × 1 | 10.1 | −1.1 [−1.8, −0.4] | +0.9 [+0.3, +1.5] |
| 128 × 4 | 42.1 | −9.3 [−10.0, −8.6] | −20.3 [−21.1, −19.4] |
| 512 × 1 | 14.4 | −1.9 [−2.0, −1.9] | +0.3 [+0.2, +0.4] |
| 512 × 4 | 59.5 | −13.5 [−14.6, −12.5] | −20.7 [−22.5, −19.0] |
| 2,048 × 1 | 35.9 | −8.3 [−8.4, −8.2] | −12.5 [−12.7, −12.4] |
| 2,048 × 4 | 143.4 | −38.8 [−41.1, −36.5] | −64.6 [−68.7, −60.5] |

p95 ms:

| Tokens × callers | engine | exact | batching |
| --- | --- | --- | --- |
| 32 × 1 | 9.4 | −0.9 [−1.0, −0.7] | +1.3 [+1.2, +1.4] |
| 32 × 4 | 38.7 | −7.8 [−9.5, −6.2] | −12.7 [−16.1, −9.3] |
| 128 × 1 | 10.7 | −1.3 [−1.9, −0.6] | +0.8 [+0.3, +1.4] |
| 128 × 4 | 43.0 | −9.1 [−9.9, −8.3] | −11.6 [−13.1, −10.2] |
| 512 × 1 | 14.8 | −1.9 [−1.9, −1.8] | +0.4 [+0.3, +0.5] |
| 512 × 4 | 60.8 | −13.0 [−14.4, −11.6] | −16.2 [−18.3, −14.2] |
| 2,048 × 1 | 36.6 | −7.4 [−7.7, −7.0] | −11.7 [−12.1, −11.4] |
| 2,048 × 4 | 145.4 | −36.2 [−43.3, −29.1] | −53.0 [−65.8, −40.2] |

Requests per second:

| Tokens × callers | engine | exact | batching |
| --- | --- | --- | --- |
| 32 × 1 | 111.43 | +12.26 [+9.41, +15.11] | −13.13 [−14.56, −11.71] |
| 32 × 4 | 106.72 | +25.53 [+22.48, +28.57] | +96.72 [+87.93, +105.50] |
| 128 × 1 | 98.68 | +12.37 [+7.25, +17.48] | −8.16 [−12.78, −3.54] |
| 128 × 4 | 94.02 | +25.77 [+23.76, +27.78] | +70.61 [+68.00, +73.22] |
| 512 × 1 | 69.59 | +11.12 [+10.71, +11.53] | −1.25 [−1.56, −0.94] |
| 512 × 4 | 66.35 | +18.37 [+16.95, +19.79] | +35.50 [+32.62, +38.38] |
| 2,048 × 1 | 27.86 | +8.26 [+7.71, +8.82] | +14.70 [+13.48, +15.92] |
| 2,048 × 4 | 27.81 | +10.03 [+9.17, +10.88] | +21.40 [+19.18, +23.62] |

## Vela-2.0-0.3B on CPU

- **Date:** 2026-10-05, at `d4c6d9a50`. Later commits change no CPU forward
  or batching path (only cancellation bookkeeping, load retries and the GPU
  device lock).
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

- **Date:** 2026-10-05, at `fb67e3bb3`, on the router image's stack as above
  (`venv-rocm72-cc`).
- **Setup:** one MI325X per model, the 4B and the 9B side by side on two
  GPUs, 8 host cores each in a `systemd-run` scope (effective cpuset logged),
  threads capped at 8; node load at most 25. 10 interleaved rounds, side
  order rotated per round; each cell is 20 warm-up requests, then 30
  requests at 1 or 4 concurrent callers.
- **Reading:** every profile is faster than the engine on every row of both
  models, with every interval on the better side but one: the 9B's `exact`
  p95 at 32 tokens × 4 callers, −552 ms [−1,113, +9], level to better at the
  10-round cap. With one caller `exact` is 21–30% faster at the median on
  the 4B and 17–27% on the 9B; `shared_context` and `batching` are 1.8–2.8×
  faster, and serve 1.9–3.0× the engine's requests per second with 4 callers.

4B, p50 ms, engine and Δ against it (mean [95% interval]):

| Tokens × callers | engine | exact | shared_context | batching |
| --- | --- | --- | --- | --- |
| 32 × 1 | 118.0 | −35.4 [−35.9, −35.0] | −70.6 [−71.2, −70.0] | −68.7 [−69.3, −68.1] |
| 32 × 4 | 598.0 | −270.3 [−489.6, −51.0] | −411.6 [−631.0, −192.3] | −413.3 [−634.5, −192.2] |
| 128 × 1 | 134.4 | −36.1 [−37.0, −35.3] | −81.0 [−81.7, −80.2] | −78.7 [−79.8, −77.6] |
| 128 × 4 | 623.9 | −232.4 [−379.4, −85.4] | −412.7 [−559.4, −266.1] | −412.5 [−559.6, −265.5] |
| 512 × 1 | 203.1 | −43.7 [−44.1, −43.4] | −131.0 [−131.5, −130.6] | −128.5 [−128.7, −128.3] |
| 512 × 4 | 901.0 | −227.0 [−375.9, −78.2] | −606.4 [−725.4, −487.3] | −580.3 [−715.6, −445.1] |
| 2,048 × 1 | 416.0 | −88.0 [−88.7, −87.3] | −203.1 [−203.6, −202.5] | −202.8 [−203.1, −202.4] |
| 2,048 × 4 | 1,681.3 | −371.2 [−439.6, −302.9] | −833.1 [−906.2, −759.9] | −836.5 [−908.3, −764.7] |

4B, p95 ms:

| Tokens × callers | engine | exact | shared_context | batching |
| --- | --- | --- | --- | --- |
| 32 × 1 | 148.5 | −58.5 [−107.3, −9.7] | −99.7 [−158.0, −41.5] | −97.8 [−156.3, −39.2] |
| 32 × 4 | 671.8 | −334.2 [−631.2, −37.2] | −480.6 [−778.3, −182.9] | −426.9 [−719.7, −134.0] |
| 128 × 1 | 145.9 | −42.3 [−52.6, −32.0] | −91.4 [−111.3, −71.4] | −89.2 [−109.2, −69.2] |
| 128 × 4 | 680.9 | −278.0 [−494.0, −61.9] | −467.0 [−679.4, −254.6] | −398.7 [−606.2, −191.2] |
| 512 × 1 | 231.7 | −54.5 [−81.8, −27.2] | −154.1 [−187.6, −120.5] | −152.2 [−185.1, −119.2] |
| 512 × 4 | 958.6 | −212.4 [−340.9, −83.9] | −639.3 [−797.2, −481.3] | −544.8 [−731.8, −357.9] |
| 2,048 × 1 | 456.0 | −101.0 [−119.1, −83.0] | −224.2 [−258.8, −189.5] | −224.8 [−258.4, −191.2] |
| 2,048 × 4 | 1,789.1 | −451.7 [−644.9, −258.4] | −932.9 [−1,120.0, −745.9] | −925.9 [−1,100.0, −751.8] |

4B, requests per second:

| Tokens × callers | engine | exact | shared_context | batching |
| --- | --- | --- | --- | --- |
| 32 × 1 | 8.23 | +3.65 [+3.54, +3.76] | +12.79 [+12.37, +13.21] | +11.96 [+11.55, +12.37] |
| 32 × 4 | 7.64 | +4.48 [+3.06, +5.90] | +13.66 [+12.41, +14.92] | +13.63 [+12.36, +14.90] |
| 128 × 1 | 7.38 | +2.72 [+2.66, +2.78] | +11.32 [+11.04, +11.60] | +10.56 [+10.25, +10.87] |
| 128 × 4 | 6.78 | +3.38 [+2.34, +4.42] | +12.08 [+11.03, +13.13] | +11.97 [+10.91, +13.04] |
| 512 × 1 | 4.81 | +1.31 [+1.27, +1.34] | +8.85 [+8.70, +9.01] | +8.46 [+8.26, +8.65] |
| 512 × 4 | 4.53 | +1.41 [+0.92, +1.90] | +8.89 [+7.95, +9.84] | +8.24 [+6.58, +9.91] |
| 2,048 × 1 | 2.37 | +0.63 [+0.61, +0.64] | +2.28 [+2.19, +2.36] | +2.27 [+2.18, +2.35] |
| 2,048 × 4 | 2.36 | +0.68 [+0.57, +0.79] | +2.35 [+2.21, +2.49] | +2.34 [+2.23, +2.45] |

9B, p50 ms, engine and Δ against it (mean [95% interval]):

| Tokens × callers | engine | exact | shared_context | batching |
| --- | --- | --- | --- | --- |
| 32 × 1 | 162.2 | −44.2 [−44.5, −43.9] | −95.2 [−95.4, −94.9] | −93.6 [−94.0, −93.2] |
| 32 × 4 | 921.7 | −448.6 [−889.5, −7.7] | −659.5 [−1,099.3, −219.8] | −690.8 [−1,128.1, −253.6] |
| 128 × 1 | 192.9 | −41.5 [−42.3, −40.8] | −117.7 [−118.8, −116.7] | −115.7 [−116.6, −114.7] |
| 128 × 4 | 936.2 | −334.4 [−601.1, −67.7] | −638.5 [−904.4, −372.5] | −640.8 [−907.0, −374.6] |
| 512 × 1 | 279.0 | −46.6 [−47.1, −46.2] | −175.5 [−175.9, −175.0] | −173.6 [−174.1, −173.2] |
| 512 × 4 | 1,230.6 | −298.5 [−509.8, −87.2] | −797.9 [−967.3, −628.4] | −815.4 [−1,020.2, −610.7] |
| 2,048 × 1 | 564.9 | −95.0 [−95.8, −94.2] | −253.4 [−254.2, −252.7] | −252.4 [−253.3, −251.6] |
| 2,048 × 4 | 2,341.5 | −470.2 [−615.3, −325.1] | −1,104.2 [−1,252.5, −955.9] | −1,103.8 [−1,252.5, −955.1] |

9B, p95 ms:

| Tokens × callers | engine | exact | shared_context | batching |
| --- | --- | --- | --- | --- |
| 32 × 1 | 215.1 | −44.0 [−45.8, −42.2] | −146.9 [−253.4, −40.4] | −144.9 [−251.6, −38.3] |
| 32 × 4 | 1,043.2 | −551.8 [−1,112.6, +9.1] | −758.8 [−1,327.7, −190.0] | −694.5 [−1,238.8, −150.2] |
| 128 × 1 | 258.0 | −79.4 [−150.3, −8.5] | −169.3 [−247.1, −91.6] | −167.3 [−245.2, −89.3] |
| 128 × 4 | 1,032.1 | −421.5 [−760.0, −83.1] | −729.0 [−1,066.7, −391.3] | −637.0 [−978.1, −295.8] |
| 512 × 1 | 316.9 | −75.5 [−135.2, −15.9] | −211.1 [−276.6, −145.6] | −207.9 [−274.4, −141.4] |
| 512 × 4 | 1,304.1 | −320.9 [−537.1, −104.8] | −863.0 [−1,113.1, −613.0] | −786.1 [−1,075.0, −497.2] |
| 2,048 × 1 | 609.7 | −102.9 [−113.6, −92.3] | −278.7 [−320.2, −237.1] | −284.4 [−340.3, −228.4] |
| 2,048 × 4 | 2,571.7 | −694.1 [−1,097.0, −291.3] | −1,328.9 [−1,735.1, −922.8] | −1,324.5 [−1,729.7, −919.2] |

9B, requests per second:

| Tokens × callers | engine | exact | shared_context | batching |
| --- | --- | --- | --- | --- |
| 32 × 1 | 5.92 | +2.20 [+2.09, +2.32] | +9.03 [+8.61, +9.44] | +8.61 [+8.21, +9.02] |
| 32 × 4 | 5.33 | +3.05 [+1.88, +4.22] | +9.73 [+8.60, +10.87] | +9.90 [+8.85, +10.95] |
| 128 × 1 | 4.91 | +1.45 [+1.25, +1.64] | +7.86 [+7.39, +8.32] | +7.46 [+6.94, +7.99] |
| 128 × 4 | 4.62 | +2.00 [+1.21, +2.79] | +8.76 [+8.00, +9.52] | +8.80 [+7.99, +9.61] |
| 512 × 1 | 3.52 | +0.76 [+0.68, +0.83] | +6.10 [+6.01, +6.20] | +5.93 [+5.83, +6.03] |
| 512 × 4 | 3.35 | +0.90 [+0.62, +1.17] | +5.98 [+5.61, +6.34] | +6.23 [+5.98, +6.49] |
| 2,048 × 1 | 1.75 | +0.35 [+0.34, +0.36] | +1.44 [+1.41, +1.46] | +1.43 [+1.41, +1.45] |
| 2,048 × 4 | 1.69 | +0.45 [+0.34, +0.56] | +1.54 [+1.43, +1.66] | +1.54 [+1.42, +1.65] |

- With one caller every side's p95 stays within 1.2× of its p50, except the
  9B's `exact` at 32 tokens (1.45×) and the engine's at 32 and 128 tokens
  (1.26–1.34×): a tree shape seen for the first time pays a one-time setup
  (kernel specialization and library algorithm choices), and prompt lengths
  vary within a cell.
- With 4 callers the engine serves one request at a time in arrival order,
  so its p50 is about four service times; `exact` runs one request per
  forward as the engine does, and the approximate profiles run the packed
  trees, `batching` also several requests per forward.

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
