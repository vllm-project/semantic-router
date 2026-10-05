# Decision 1.0 performance

The `decision1` family against each Decision 1.0 package's bundled runtime,
on the same node, inputs and devices.

- **On CPU, exact is level with the bundled runtime or better on every row but
  one: Kai's router requests are 7.6% slower at p50, wholly, and open** (CPU
  section). Every CPU row is an interleaved A/B with a 95% interval on the
  difference: single requests on uniform and mixed lengths for all seven
  models, their throughput, and router requests. Both sides run the same FP32
  math through the same MKL kernels, so exact can't be much faster; the single
  requests are level on all seven models, and the decoders' rates are a little
  better where the interval says so (Eos +0.23 requests/s [+0.07, +0.39] and
  Sol +0.09 [+0.01, +0.17] on mixed lengths, Nox +0.02 [+0.00, +0.04] at C =
  1).
- **On CPU the encoders' opt-in profiles win.** For six router signals about
  one prompt, `batching` answers 3.0–3.3× and `max_speed` 4.8–5.3× faster at
  p50 than the bundled runtime, and they serve 3.0–3.3× and 4.8–5.3× its
  rate one request at a time (up to 4.5× and 7.3× at C = 16). `max_speed`
  runs the encoders on `float32-packed` copies of their three stacks: single
  requests 1.2–1.4× faster at p50, up to 2.0× the bundled throughput, and no
  decision changes in 1,431 requests per encoder.
- **On ROCm, exact and faster, on the official-wheel stack:** in interleaved
  A/Bs (5 rounds, 95% intervals) in `be7366c49`, which has the packages of
  `af71d5e82` (the official PyTorch 2.12.0 wheel, whose attention rounds
  differently from the release's), no row is worse. Exact serves a single
  request 1.2–1.3× faster at p50 on the encoders and 1.5–3.1× on the decoders,
  end to end through the API; for six router signals about one prompt,
  `batching` cuts the encoders' p95 by 35–38% and `shared_context` the
  decoders' by 51–74%. These rows were not re-timed on the adopted router
  image `a580be6b9` (vLLM's ROCm PyTorch), on which exact answers
  byte-identically to the release (parity record).

- **CPU:** node C, 16 cores of an AMD EPYC 9575F (Zen 5) per run, each side
  in its own docker cgroup cpuset (the effective cpuset is logged), CPU
  PyTorch 2.10.0 with MKL (the router image's pin), Transformers 5.17 for the
  bundled side, 16 threads on both sides (`OMP_NUM_THREADS` and
  `MKL_NUM_THREADS` too).
- **ROCm:** one AMD Instinct MI325X (gfx942) per run, on 8 host cores of the
  GPU's NUMA node, in `mr-p24-lead/extproc-rocm72cc:be7366c49` with
  `transformers==5.17.0` added for the bundled side: the official PyTorch
  2.12.0 wheel from the rocm7.2 index, Triton 3.7.0, FLA 0.5.2 and
  causal-conv1d 1.7.0, the packages of the image tag `af71d5e82`. The adopted
  router image `a580be6b9` takes PyTorch, AOTriton and the ROCm 7.2.3
  libraries from vLLM's ROCm image instead; these rows were not re-timed on
  it.
- **Bundled side:** the package's runtime (Transformers remote code,
  `system_one`), a library that answers one request at a time, so its
  sequential rate is its throughput. On ROCm it runs the same FLA kernel
  choices as the native side.
- **Commits:** the CPU A/B is at `8fc0bf02b` (S1 and IP3a merged). Since then
  the CPU request path changed only by the scheduler's cancellation
  bookkeeping (`1ce0bafc9`: one `cancelled()` check per answered job, one
  `done()` check per planned job). The device lock, the per-model kernel
  choices and the thread-local graph capture are GPU-only, and load retries,
  placement by plugin name and the advertised limits run at load, so the run
  stands for the head. The ROCm rows' commits are in their section.
- **Raw results:** `decision1-performance.json`: every run without paths
  (the CPU A/B is set `bench-cpu-ab`), and under `intervals` each A/B row's
  means, difference, interval, rounds and verdict.

## How the CPU rows were measured

Every CPU row is an interleaved A/B on one node (the 16:53 standard):

- **Each side in its own fresh process,** on the same cgroup-confined cores:
  the bundled runtime and the runtime's exact profile (and, where a table
  shows it, an opt-in profile). Per round the sides run back to back, and the
  order alternates (rotates, with more sides) every round.
- **Rounds:** 5. Rows whose point estimate was worse with an interval
  straddling zero got rounds 6–10, the most the standard adds. The opt-in
  profiles' single-request rows are 2 rounds (informative; no gate row
  depends on them).
- **Difference:** runtime minus bundled within each round, with a 95% t
  interval over the rounds. **Better** or **worse** means the whole interval
  is on that side of zero. **Level** means it straddles zero; a level row
  whose point estimate is worse is recorded with its interval.
- **Where:** four 16-core lanes of one NUMA node (0–15, 16–31, 32–47, 48–63),
  one timed job per lane at a time; each row's rounds stay on one lane, with
  two exceptions (CPU section). Other workstreams ran on the node's other
  cores and GPUs. A round counts only if the node's 1-minute load, read at
  every side's start, was under 120; the highest reading was 99.

## Workloads

- **Single request:** one Choice question about a prompt. Uniform lengths are
  the first N typed-final prompts (about 250 tokens). Mixed lengths are a
  seeded sample of css15 prompts that fit the 1,024-token input (11–770
  tokens, median 43). On ROCm, N = 400 uniform. On CPU, both lengths at
  N = 400 for the encoders and 100 for Eos and Sol, and 20 uniform for Nox
  and Lux.
- **Router request:** the six router signals that Route's `QUESTIONS.json`
  declares (domain, modality, jailbreak, safety, pii, fact_check), as explicit
  questions about each public231 prompt. ROCm uses all 231. CPU uses the first
  30 (encoders), 20 (Eos, Sol), 10 (Nox) or 6 (Lux).
- **Many questions:** the public request of `tools/many_questions.py` (one
  ticket) at 16, 64 and 128 questions.
- **Throughput:** the requests in waves of C concurrent requests through
  `Runtime.call`, against the bundled runtime's sequential rate in the same
  round. "One at a time" is the runtime's own sequential rate.

Latencies are in ms, end to end through `Runtime.call` (the API's request
path) for the runtime and `system_one` for the bundled side. The encoders'
approximate profile is `batching` (bidirectional attention shares no prefix),
the decoders' `shared_context`.

## ROCm

Every row is the 16:53-standard A/B on the official-wheel ROCm stack, image
`mr-p24-lead/extproc-rocm72cc:be7366c49`: the official PyTorch 2.12.0 wheel from
the rocm7.2 index (HIP 7.2, AOTriton 0.11.2), Triton 3.7.0, FLA 0.5.2 and
`causal-conv1d` 1.7.0 built for gfx942. The image tag `af71d5e82` has the same
packages. Both sides run in that image, with `transformers==5.17.0` added for
the bundled side; pip added only Transformers and its pure-Python
dependencies. The runtime is `a1c7a284e`, one AMD Instinct MI325X (gfx942) per
run. **These rows were not re-timed on the router image adopted afterwards**
(`a580be6b9`), which takes PyTorch, AOTriton 0.13.50 and the ROCm 7.2.3
libraries from vLLM's ROCm image: its attention is the release's, not this
stack's.

- **Cores:** each GPU's host threads run in a docker cgroup cpuset of 8 cores
  on the GPU's own NUMA node that no other timed job used (GPU1 0–7, GPU2
  8–15, GPU5 144–151, GPU7 152–159); the effective cpuset is logged.
- **Sides:** the bundled runtime, the runtime's exact profile and its
  approximate profile, each in its own process, the order rotated every round.
  Five rounds per row, the standard's minimum: no row read worse, so none got
  rounds 6–10. From 11:44 to 12:01 UTC the lead's trial images were built on
  node C's other cores while GPU5 and GPU7 ran their many-question rounds 2
  and 3 (node load 16–34); those rounds are kept.
- **Numerics on this stack:** exact answers byte-identically across cold
  processes, and the bundled runtime and the runtime run the same kernels; the
  release image's answers differ by rounding. On the adopted image they are
  byte-identical to the release (parity record).

### Single requests

The first 400 typed-final prompts, one Choice question each:

| Model | Profile | p50 bundled → runtime | p50 Δ [95% CI] | p95 bundled → runtime | p95 Δ [95% CI] | Rounds | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | `exact` | 7.98 → 6.29 | −1.69 [−2.04, −1.35] | 16.9 → 11.1 | −5.8 [−15.7, +4.1] | 5 | level |
|  | `batching` | 7.98 → 6.97 | −1.01 [−1.56, −0.46] | 16.9 → 16.3 | −0.6 [−9.1, +7.8] | 5 | level |
| Lex-0.6B | `exact` | 7.52 → 6.15 | −1.37 [−1.94, −0.79] | 9.16 → 9.83 | +0.67 [−2.31, +3.65] | 5 | level |
|  | `batching` | 7.52 → 6.93 | −0.59 [−1.11, −0.07] | 9.16 → 8.67 | −0.49 [−3.24, +2.25] | 5 | level |
| Route-0.6B | `exact` | 7.41 → 6.15 | −1.26 [−1.72, −0.80] | 8.44 → 8.55 | +0.10 [−2.95, +3.15] | 5 | level |
|  | `batching` | 7.41 → 6.90 | −0.51 [−1.02, +0.00] | 8.44 → 7.25 | −1.19 [−1.71, −0.67] | 5 | level |
| Eos-0.8B | `exact` | 22.7 → 8.4 | −14.3 [−15.2, −13.4] | 25.6 → 12.1 | −13.5 [−21.6, −5.4] | 5 | better |
|  | `shared_context` | 22.7 → 8.1 | −14.6 [−15.0, −14.1] | 25.6 → 10.2 | −15.4 [−18.5, −12.2] | 5 | better |
| Sol-2B | `exact` | 23.9 → 7.7 | −16.2 [−19.5, −13.0] | 31.3 → 13.0 | −18.3 [−30.0, −6.5] | 5 | better |
|  | `shared_context` | 23.9 → 7.6 | −16.3 [−19.6, −13.1] | 31.3 → 11.0 | −20.3 [−33.7, −6.8] | 5 | better |
| Nox-4B | `exact` | 29.2 → 13.2 | −16.0 [−16.4, −15.5] | 32.1 → 15.6 | −16.5 [−21.6, −11.4] | 5 | better |
|  | `shared_context` | 29.2 → 13.3 | −15.9 [−16.3, −15.6] | 32.1 → 15.5 | −16.6 [−21.8, −11.4] | 5 | better |
| Lux-9B | `exact` | 31.4 → 20.3 | −11.1 [−19.8, −2.4] | 38.9 → 23.0 | −15.8 [−33.4, +1.7] | 5 | level |
|  | `shared_context` | 31.4 → 18.6 | −12.8 [−19.2, −6.5] | 38.9 → 26.6 | −12.3 [−40.8, +16.3] | 5 | level |

- Every exact p50 interval is on the runtime's side: 1.2–1.3× faster on the
  encoders (Kai 7.98 → 6.29 ms) and 1.5–3.1× on the decoders (Eos 22.7 → 8.4,
  Sol 23.9 → 7.7, Nox 29.2 → 13.2, Lux 31.4 → 20.3). The level rows are level
  on p95 only. The two positive point estimates, Lex's and Route's exact p95
  (+0.67 and +0.10 ms), are level.

### Router requests

The six router signals about each of the 231 public231 prompts:

| Model | Profile | p50 bundled → runtime | p50 Δ [95% CI] | p95 bundled → runtime | p95 Δ [95% CI] | Rounds | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | `exact` | 24.3 → 20.6 | −3.7 [−14.5, +7.1] | 50.5 → 45.1 | −5.4 [−19.0, +8.1] | 5 | level |
|  | `batching` | 24.3 → 13.8 | −10.4 [−15.4, −5.5] | 50.5 → 33.0 | −17.5 [−27.3, −7.8] | 5 | better |
| Lex-0.6B | `exact` | 22.6 → 17.6 | −5.0 [−7.8, −2.3] | 50.4 → 40.8 | −9.6 [−16.6, −2.5] | 5 | better |
|  | `batching` | 22.6 → 12.4 | −10.2 [−13.2, −7.2] | 50.4 → 31.3 | −19.1 [−24.9, −13.3] | 5 | better |
| Route-0.6B | `exact` | 26.6 → 20.7 | −5.9 [−10.8, −1.0] | 55.2 → 47.6 | −7.6 [−10.3, −4.8] | 5 | better |
|  | `batching` | 26.6 → 14.0 | −12.5 [−23.0, −2.0] | 55.2 → 35.4 | −19.8 [−33.6, −6.0] | 5 | better |
| Eos-0.8B | `exact` | 31.6 → 24.7 | −7.0 [−15.0, +1.1] | 156.9 → 148.3 | −8.7 [−33.4, +16.1] | 5 | level |
|  | `shared_context` | 31.6 → 24.0 | −7.6 [−13.0, −2.3] | 156.9 → 76.7 | −80.2 [−102.0, −58.4] | 5 | better |
| Sol-2B | `exact` | 46.3 → 36.2 | −10.1 [−25.8, +5.7] | 204.5 → 177.5 | −27.0 [−73.3, +19.2] | 5 | level |
|  | `shared_context` | 46.3 → 36.2 | −10.1 [−22.5, +2.3] | 204.5 → 72.1 | −132.4 [−163.7, −101.1] | 5 | level |
| Nox-4B | `exact` | 75.7 → 64.0 | −11.8 [−22.7, −0.8] | 461.5 → 393.0 | −68.6 [−144.9, +7.7] | 5 | level |
|  | `shared_context` | 75.7 → 58.1 | −17.7 [−21.8, −13.5] | 461.5 → 119.5 | −342.1 [−413.7, −270.4] | 5 | better |
| Lux-9B | `exact` | 123.3 → 105.1 | −18.2 [−53.7, +17.2] | 665.5 → 618.5 | −47.1 [−176.9, +82.8] | 5 | level |
|  | `shared_context` | 123.3 → 95.4 | −27.9 [−39.8, −16.0] | 665.5 → 178.4 | −487.1 [−590.6, −383.6] | 5 | better |

- Exact runs the released shapes, so on the encoders it does the bundled
  runtime's work: every question type present runs its stack over all six
  rows. `batching` runs each stack over its own rows only, packed, and every
  layer stack replays its own bucket graphs.
- `shared_context` computes the prompt once for all six questions. The
  decoders' long prompts are where the bundled runtime's p95 comes from.
- Exact is level or better on every model, with p50 15–22% below the bundled
  runtime's. `batching` cuts the encoders' p95 by 35–38% and `shared_context`
  the decoders' by 51–74% (Lux 665.5 → 178.4 ms); Sol's `shared_context` row
  is level only on p50.

### Throughput (requests/s)

Single requests in waves of C through `Runtime.call`, against the bundled
runtime's sequential rate in the same round:

| Model | Profile | Bundled | C = 1 | C = 4 | C = 16 | Rounds | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | `exact` | 102.3 | 148.2, +45.8 [+23.5, +68.2] | 152.6, +50.3 [+28.1, +72.5] | 159.1, +56.8 [+33.1, +80.5] | 5 | better |
|  | `batching` | 102.3 | 123.6, +21.2 [−7.5, +49.9] | 299.9, +197.6 [+153.1, +242.1] | 575.3, +472.9 [+408.3, +537.6] | 5 | level |
| Lex-0.6B | `exact` | 126.2 | 155.7, +29.4 [+18.3, +40.6] | 154.9, +28.7 [+14.7, +42.7] | 159.1, +32.9 [+8.8, +57.0] | 5 | better |
|  | `batching` | 126.2 | 132.4, +6.2 [−22.1, +34.5] | 298.9, +172.7 [+75.5, +269.9] | 595.5, +469.2 [+379.4, +559.1] | 5 | level |
| Route-0.6B | `exact` | 129.9 | 148.5, +18.6 [+4.4, +32.9] | 145.5, +15.6 [−7.4, +38.6] | 144.0, +14.1 [−19.0, +47.2] | 5 | level |
|  | `batching` | 129.9 | 127.5, −2.4 [−29.0, +24.3] | 277.9, +148.0 [+37.2, +258.7] | 532.4, +402.5 [+221.8, +583.3] | 5 | level |
| Eos-0.8B | `exact` | 43.5 | 115.8, +72.4 [+59.5, +85.3] | 118.6, +75.1 [+62.9, +87.4] | 119.5, +76.1 [+65.4, +86.7] | 5 | better |
|  | `shared_context` | 43.5 | 116.0, +72.6 [+61.3, +83.9] | 121.5, +78.1 [+75.4, +80.8] | 118.5, +75.1 [+58.7, +91.5] | 5 | better |
| Sol-2B | `exact` | 40.3 | 126.3, +86.0 [+68.5, +103.6] | 126.6, +86.3 [+70.8, +101.9] | 128.2, +87.9 [+74.6, +101.2] | 5 | better |
|  | `shared_context` | 40.3 | 108.6, +68.3 [+48.0, +88.6] | 112.4, +72.1 [+54.0, +90.2] | 123.0, +82.7 [+68.2, +97.2] | 5 | better |
| Nox-4B | `exact` | 33.8 | 71.2, +37.4 [+28.0, +46.7] | 75.1, +41.3 [+40.7, +41.9] | 74.6, +40.8 [+38.4, +43.1] | 5 | better |
|  | `shared_context` | 33.8 | 74.8, +41.0 [+40.7, +41.3] | 72.5, +38.7 [+29.0, +48.4] | 69.8, +36.0 [+24.7, +47.3] | 5 | better |
| Lux-9B | `exact` | 31.6 | 51.6, +20.0 [+12.0, +28.1] | 53.6, +22.0 [+16.5, +27.5] | 50.0, +18.4 [+7.0, +29.8] | 5 | better |
|  | `shared_context` | 31.6 | 51.3, +19.8 [+11.4, +28.1] | 50.7, +19.1 [+11.7, +26.5] | 52.0, +20.4 [+17.5, +23.3] | 5 | better |

Router requests:

| Model | Profile | Bundled | C = 1 | C = 4 | C = 16 | Rounds | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | `exact` | 44.4 | 47.5, +3.1 [−8.3, +14.5] | 53.6, +9.2 [+8.7, +9.7] | 50.4, +6.0 [−3.8, +15.8] | 5 | level |
|  | `batching` | 44.4 | 78.2, +33.8 [+32.9, +34.7] | 120.0, +75.6 [+72.7, +78.6] | 155.8, +111.4 [+108.0, +114.9] | 5 | better |
| Lex-0.6B | `exact` | 37.2 | 53.1, +15.8 [+8.1, +23.5] | 53.7, +16.4 [+8.5, +24.3] | 54.2, +17.0 [+8.3, +25.7] | 5 | better |
|  | `batching` | 37.2 | 78.1, +40.9 [+32.2, +49.5] | 119.9, +82.7 [+74.8, +90.6] | 157.2, +120.0 [+111.4, +128.5] | 5 | better |
| Route-0.6B | `exact` | 44.3 | 53.0, +8.7 [+7.6, +9.8] | 52.1, +7.8 [+3.7, +11.8] | 51.2, +6.8 [−0.9, +14.5] | 5 | level |
|  | `batching` | 44.3 | 77.0, +32.6 [+31.5, +33.8] | 116.5, +72.2 [+67.6, +76.7] | 155.0, +110.6 [+103.3, +118.0] | 5 | better |
| Eos-0.8B | `exact` | 20.1 | 22.6, +2.5 [+0.5, +4.4] | 22.8, +2.7 [+0.2, +5.1] | 23.5, +3.4 [+1.5, +5.3] | 5 | better |
|  | `shared_context` | 20.1 | 30.4, +10.3 [+7.6, +13.0] | 32.3, +12.2 [+10.1, +14.3] | 32.7, +12.6 [+10.2, +15.0] | 5 | better |
| Sol-2B | `exact` | 14.0 | 16.9, +2.9 [−0.7, +6.6] | 17.7, +3.7 [−0.0, +7.4] | 19.1, +5.2 [+3.1, +7.2] | 5 | level |
|  | `shared_context` | 14.0 | 27.7, +13.7 [+11.8, +15.6] | 26.4, +12.4 [+10.3, +14.5] | 24.3, +10.4 [+7.3, +13.5] | 5 | better |
| Nox-4B | `exact` | 7.06 | 8.42, +1.36 [+0.70, +2.02] | 8.51, +1.45 [+0.82, +2.08] | 8.50, +1.44 [+0.77, +2.12] | 5 | better |
|  | `shared_context` | 7.06 | 15.07, +8.0 [+7.8, +8.3] | 14.91, +7.9 [+7.0, +8.7] | 14.79, +7.7 [+6.2, +9.2] | 5 | better |
| Lux-9B | `exact` | 5.00 | 5.67, +0.67 [+0.66, +0.69] | 5.69, +0.70 [+0.68, +0.71] | 5.53, +0.54 [+0.25, +0.82] | 5 | better |
|  | `shared_context` | 5.00 | 11.51, +6.5 [+6.4, +6.6] | 11.26, +6.3 [+5.3, +7.2] | 11.65, +6.7 [+6.6, +6.7] | 5 | better |

- Exact runs one request per forward, so concurrency adds little to it.
- `batching` coalesces the questions of concurrent requests, and the encoders
  pack them without padding. `shared_context` doesn't coalesce requests.
- At C = 1, exact serves 1.1–1.4× the bundled rate on the encoders and
  1.6–3.1× on the decoders. On single requests `batching` reaches 4.1–5.6× at
  C = 16, and on router requests 3.5–4.2×; `shared_context` serves router
  requests at 1.5–2.3× from C = 1. The `batching` single-request rows are
  level only at C = 1, where it pays the batching window.

### Many questions about one input

The public request of `tools/many_questions.py` (one ticket) at 16, 64 and 128
questions, 20 runs per side and round:

| Model | Questions | Profile | p50 bundled → runtime | p50 Δ [95% CI] | p95 bundled → runtime | p95 Δ [95% CI] | Rounds | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | 16 | `exact` | 48.7 → 44.5 | −4.2 [−10.4, +2.1] | 51.0 → 48.2 | −2.8 [−14.9, +9.3] | 5 | level |
|  | 16 | `batching` | 48.7 → 30.6 | −18.1 [−21.4, −14.8] | 51.0 → 34.7 | −16.4 [−20.3, −12.5] | 5 | better |
|  | 64 | `exact` | 127.0 → 91.7 | −35.3 [−67.2, −3.5] | 138.7 → 124.3 | −14.4 [−95.9, +67.1] | 5 | level |
|  | 64 | `batching` | 127.0 → 90.1 | −36.9 [−90.4, +16.6] | 138.7 → 96.5 | −42.2 [−112.8, +28.4] | 5 | level |
|  | 128 | `exact` | 218.1 → 174.0 | −44.1 [−84.7, −3.5] | 240.1 → 181.4 | −58.7 [−101.7, −15.7] | 5 | better |
|  | 128 | `batching` | 218.1 → 154.5 | −63.5 [−82.3, −44.8] | 240.1 → 161.5 | −78.6 [−94.4, −62.7] | 5 | better |
| Lex-0.6B | 16 | `exact` | 48.0 → 39.4 | −8.6 [−9.7, −7.5] | 49.1 → 41.1 | −8.0 [−9.5, −6.6] | 5 | better |
|  | 16 | `batching` | 48.0 → 30.9 | −17.1 [−21.4, −12.7] | 49.1 → 33.1 | −16.0 [−23.0, −9.1] | 5 | better |
|  | 64 | `exact` | 119.1 → 96.2 | −22.9 [−29.0, −16.9] | 130.3 → 106.1 | −24.2 [−44.5, −3.8] | 5 | better |
|  | 64 | `batching` | 119.1 → 89.8 | −29.3 [−71.5, +12.9] | 130.3 → 92.7 | −37.6 [−80.3, +5.2] | 5 | level |
|  | 128 | `exact` | 217.6 → 172.9 | −44.7 [−84.2, −5.2] | 268.9 → 183.0 | −85.9 [−142.5, −29.2] | 5 | better |
|  | 128 | `batching` | 217.6 → 141.1 | −76.5 [−106.6, −46.5] | 268.9 → 144.9 | −124.0 [−183.3, −64.8] | 5 | better |
| Route-0.6B | 16 | `exact` | 61.6 → 41.4 | −20.3 [−54.5, +14.0] | 68.2 → 44.1 | −24.0 [−72.4, +24.3] | 5 | level |
|  | 16 | `batching` | 61.6 → 29.5 | −32.1 [−65.3, +1.0] | 68.2 → 32.5 | −35.7 [−80.7, +9.2] | 5 | level |
|  | 64 | `exact` | 115.0 → 95.6 | −19.5 [−28.4, −10.6] | 150.9 → 99.6 | −51.3 [−130.3, +27.7] | 5 | level |
|  | 64 | `batching` | 115.0 → 76.9 | −38.1 [−39.8, −36.5] | 150.9 → 82.2 | −68.6 [−146.6, +9.4] | 5 | level |
|  | 128 | `exact` | 207.4 → 218.9 | +11.5 [−55.2, +78.1] | 222.6 → 228.6 | +6.0 [−79.1, +91.1] | 5 | level |
|  | 128 | `batching` | 207.4 → 146.0 | −61.5 [−77.5, −45.4] | 222.6 → 151.3 | −71.3 [−98.3, −44.2] | 5 | better |
| Eos-0.8B | 16 | `exact` | 57.7 → 44.7 | −13.1 [−18.9, −7.3] | 62.4 → 47.2 | −15.2 [−23.6, −6.8] | 5 | better |
|  | 16 | `shared_context` | 57.7 → 45.4 | −12.4 [−19.5, −5.3] | 62.4 → 50.2 | −12.2 [−28.9, +4.5] | 5 | level |
|  | 64 | `exact` | 201.8 → 175.2 | −26.6 [−40.0, −13.2] | 215.6 → 184.4 | −31.2 [−50.0, −12.3] | 5 | better |
|  | 64 | `shared_context` | 201.8 → 75.5 | −126.3 [−129.0, −123.6] | 215.6 → 81.9 | −133.7 [−153.0, −114.4] | 5 | better |
|  | 128 | `exact` | 402.0 → 360.1 | −42.0 [−92.3, +8.3] | 510.9 → 370.3 | −140.6 [−437.7, +156.4] | 5 | level |
|  | 128 | `shared_context` | 402.0 → 138.5 | −263.6 [−273.1, −254.1] | 510.9 → 153.7 | −357.2 [−642.4, −72.1] | 5 | better |
| Sol-2B | 16 | `exact` | 81.0 → 59.0 | −22.0 [−34.3, −9.7] | 84.8 → 61.5 | −23.3 [−37.5, −9.0] | 5 | better |
|  | 16 | `shared_context` | 81.0 → 35.9 | −45.0 [−59.1, −31.0] | 84.8 → 39.3 | −45.5 [−61.8, −29.3] | 5 | better |
|  | 64 | `exact` | 302.4 → 224.6 | −77.7 [−124.2, −31.3] | 310.4 → 236.4 | −73.9 [−106.5, −41.3] | 5 | better |
|  | 64 | `shared_context` | 302.4 → 88.1 | −214.3 [−260.5, −168.1] | 310.4 → 94.0 | −216.4 [−259.2, −173.6] | 5 | better |
|  | 128 | `exact` | 567.6 → 462.9 | −104.7 [−166.7, −42.7] | 580.5 → 476.9 | −103.6 [−170.0, −37.1] | 5 | better |
|  | 128 | `shared_context` | 567.6 → 237.6 | −330.0 [−549.2, −110.9] | 580.5 → 292.7 | −287.8 [−632.8, +57.2] | 5 | level |
| Nox-4B | 16 | `exact` | 165.1 → 119.0 | −46.2 [−82.6, −9.7] | 181.8 → 120.4 | −61.3 [−124.4, +1.8] | 5 | level |
|  | 16 | `shared_context` | 165.1 → 55.6 | −109.6 [−145.8, −73.3] | 181.8 → 57.4 | −124.4 [−186.3, −62.5] | 5 | better |
|  | 64 | `exact` | 612.1 → 490.0 | −122.1 [−250.3, +6.1] | 668.5 → 557.6 | −110.9 [−264.3, +42.6] | 5 | level |
|  | 64 | `shared_context` | 612.1 → 167.5 | −444.6 [−555.9, −333.3] | 668.5 → 169.3 | −499.1 [−648.3, −350.0] | 5 | better |
|  | 128 | `exact` | 1,143 → 994 | −149 [−266, −31] | 1,344 → 1,099 | −244 [−597, +109] | 5 | level |
|  | 128 | `shared_context` | 1,143 → 327 | −816 [−852, −779] | 1,344 → 335 | −1,008 [−1,505, −512] | 5 | better |
| Lux-9B | 16 | `exact` | 215.4 → 187.7 | −27.7 [−31.6, −23.8] | 218.4 → 189.5 | −28.9 [−30.1, −27.7] | 5 | better |
|  | 16 | `shared_context` | 215.4 → 76.4 | −139.0 [−160.8, −117.3] | 218.4 → 78.5 | −139.9 [−164.5, −115.4] | 5 | better |
|  | 64 | `exact` | 854.4 → 722.4 | −131.9 [−237.4, −26.5] | 878.1 → 828.1 | −50.0 [−312.1, +212.2] | 5 | level |
|  | 64 | `shared_context` | 854.4 → 247.7 | −606.7 [−715.8, −497.5] | 878.1 → 292.7 | −585.4 [−778.2, −392.6] | 5 | better |
|  | 128 | `exact` | 1,642 → 1,426 | −216 [−232, −200] | 1,751 → 1,584 | −168 [−516, +181] | 5 | level |
|  | 128 | `shared_context` | 1,642 → 442 | −1,200 [−1,206, −1,195] | 1,751 → 469 | −1,283 [−1,512, −1,054] | 5 | better |

- Exact is level or better at every size. At 128 questions `shared_context`
  answers 2.4–3.7× faster at p50 on the decoders (Eos 402 → 139 ms, Lux 1,642
  → 442) and `batching` 1.4–1.5× on the encoders. The only positive point
  estimates, Route's exact p50 and p95 at 128 questions (+11.5 ms [−55.2,
  +78.1] and +6.0 ms [−79.1, +91.1]), are level.

### What per-model kernel choices cost (review P1-4)

Since `be29b9be8` every FLA autotuner routes its lookups to the scoped model's
configurations (design §11), and every device call runs inside the model's
scope. A graph replay launches nothing from Python, so the cost falls on
eager forwards only. These runs are on the packages' release image, before the
stack change; they compare two runtime commits, not the runtime with the
bundled runtime.

- **Per launch** (microbenchmark, the two cache lookups Triton makes per
  autotuned launch): 188 ns inside a pinned scope against 57 ns for a plain
  dict, and 129 ns to enter and leave a scope once per device call. An eager
  Eos forward makes about 126 autotuned launches, so about 17 µs of about
  20 ms.
- **Eager forwards on ROCm:** the decoder loaded through `Decision1Family` and
  the native engine with graphs off (fused layers on), pinned as each commit
  pins, then `run` on the exact batches of 400 typed-final requests (one
  Choice question each, synchronized per request; one untimed pass, one timed
  pass). One process per run, staging `7511ad785` against `64d6046ad`,
  alternating first each round, on one MI325X of node C:

  | Model | Metric | `7511ad785` | `64d6046ad` | Difference, 95% CI | Rounds |
  | --- | --- | --- | --- | --- | --- |
  | Eos-0.8B | p50 ms | 19.68 | 19.76 | +0.40% [−1.96, +2.76] | 10 |
  |  | requests/s | 50.37 | 49.32 | −2.09% [−6.78, +2.60] | 10 |
  | Lux-9B | p50 ms | 19.75 | 20.09 | +1.70% [+0.93, +2.48] | 10 |
  |  | requests/s | 49.81 | 47.09 | −5.45% [−10.54, −0.36] | 10 |

  Eos is level: its intervals straddle zero after ten rounds, the most the
  16:53 standard adds. **Lux is slower,** wholly: p50 by 0.34 ms in every one of
  the ten rounds, and its throughput also by a few tail spikes of the newer
  commit (its mean rose in four rounds). A first 10-round Eos series at
  `8fc0bf02b`, with the slower lookups, gave p50 +0.22% [−0.68, +1.11].
- **How much of that is the routing:** in one process, alternating per
  request between the routed caches and plain dicts that hold the same
  resolved configurations (300 pairs each, the same 100 requests), the routing
  costs a median 0.10 ms per Lux forward (19.84 against 19.76 ms) and 0.08 ms
  per Eos forward (19.17 against 19.04 ms): 0.4–0.5%. Every lookup hits its
  memo (168 per Lux request, one key per kernel). The rest of the
  cross-process difference is not in the lookups.
- **Why the routing stays:** a per-scope swap of the autotuners' caches would
  make the lookups free, but it is safe only while one process serves a single
  GPU, and resolving a new key would then go through FLA's config files and its
  environment variable again. On the gate rows the cost is smaller still,
  because the exact decoders replay graphs for every shape they have seen
  twice.

## CPU

Every row is the 16:53-standard A/B at `8fc0bf02b` described above. The
Sol, Nox and Lux rows are new in this record (review P2-23), and every row
replaces the paired and two-round rows of IP2.

### Single requests, exact

Latency, ms:

| Model | Lengths | p50 bundled → runtime | p50 Δ [95% CI] | p95 bundled → runtime | p95 Δ [95% CI] | Rounds | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | uniform | 93.1 → 92.9 | −0.2 [−2.8, +2.4] | 125.5 → 129.3 | +3.7 [−5.2, +12.6] | 10 | level |
| Kai-0.6B | mixed | 55.8 → 55.7 | −0.2 [−1.7, +1.4] | 214.3 → 211.6 | −2.7 [−9.0, +3.6] | 10 | level |
| Lex-0.6B | uniform | 93.0 → 92.1 | −0.9 [−2.3, +0.5] | 124.7 → 126.2 | +1.5 [−9.4, +12.5] | 10 | level |
| Lex-0.6B | mixed | 55.7 → 56.9 | +1.1 [−0.9, +3.1] | 216.1 → 217.8 | +1.8 [−7.0, +10.5] | 10 | level |
| Route-0.6B | uniform | 95.8 → 94.4 | −1.4 [−3.9, +1.0] | 136.9 → 132.2 | −4.8 [−15.7, +6.2] | 10 | level |
| Route-0.6B | mixed | 59.2 → 57.4 | −1.9 [−5.2, +1.4] | 231.3 → 219.2 | −12.1 [−24.5, +0.3] | 10 | level |
| Eos-0.8B | uniform | 520.2 → 507.1 | −13.1 [−59.5, +33.3] | 706.3 → 680.5 | −25.9 [−87.4, +35.7] | 10 | level |
| Eos-0.8B | mixed | 386.3 → 365.7 | −20.7 [−69.3, +28.0] | 1,251.3 → 1,085.9 | −165.4 [−526.3, +195.5] | 5 | level |
| Sol-2B | uniform | 1,152 → 1,133 | −19 [−97, +59] | 1,421 → 1,435 | +14 [−128, +157] | 10 | level |
| Sol-2B | mixed | 847.8 → 791.7 | −56.1 [−175.5, +63.4] | 2,743.7 → 2,263.3 | −480.4 [−969.0, +8.2] | 5 | level |
| Nox-4B | uniform | 3,066 → 2,864 | −202 [−406, +3] | 3,361 → 3,174 | −187 [−613, +239] | 5 | level |
| Lux-9B | uniform | 5,144 → 5,112 | −32 [−376, +312] | 5,522 → 5,734 | +212 [−286, +709] | 10 | level |

Throughput, requests/s: the bundled runtime's sequential rate, then the
runtime's rate and its difference from that rate in the same round:

| Model | Lengths | Bundled | One at a time | C = 1 | C = 4 | C = 16 | Rounds | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | uniform | 10.0 | 10.0, −0.1 [−0.3, +0.2] | 10.0, −0.0 [−0.3, +0.2] | 10.0, −0.0 [−0.1, +0.1] | 10.2, +0.2 [+0.0, +0.3] | 10 | level |
| Kai-0.6B | mixed | 12.3 | 12.5, +0.2 [−0.1, +0.5] | 11.9, −0.4 [−0.8, +0.1] | 12.2, −0.2 [−0.6, +0.3] | 12.3, −0.1 [−0.4, +0.3] | 10 | level |
| Lex-0.6B | uniform | 10.0 | 10.1, +0.0 [−0.3, +0.3] | 9.9, −0.2 [−0.4, +0.1] | 10.2, +0.2 [−0.0, +0.4] | 10.3, +0.2 [−0.1, +0.5] | 10 | level |
| Lex-0.6B | mixed | 12.4 | 12.2, −0.2 [−0.5, +0.1] | 12.0, −0.4 [−0.8, +0.1] | 12.2, −0.1 [−0.6, +0.3] | 12.6, +0.2 [−0.3, +0.7] | 10 | level |
| Route-0.6B | uniform | 9.69 | 9.88, +0.19 [−0.19, +0.57] | 9.68, −0.01 [−0.47, +0.46] | 9.96, +0.27 [−0.06, +0.61] | 10.22, +0.53 [+0.27, +0.79] | 10 | level |
| Route-0.6B | mixed | 11.7 | 12.0, +0.3 [−0.2, +0.8] | 12.0, +0.3 [−0.2, +0.8] | 12.2, +0.4 [−0.2, +1.1] | 12.3, +0.6 [+0.1, +1.2] | 10 | level |
| Eos-0.8B | uniform | 1.84 | 1.88, +0.04 [−0.09, +0.16] | 1.86, +0.01 [−0.12, +0.15] | 1.89, +0.04 [−0.08, +0.17] | – | 10 | level |
| Eos-0.8B | mixed | 1.87 | 2.10, +0.23 [+0.07, +0.39] | 2.14, +0.26 [+0.02, +0.50] | 2.13, +0.26 [+0.03, +0.49] | – | 5 | better |
| Sol-2B | uniform | 0.86 | 0.86, +0.00 [−0.05, +0.06] | 0.87, +0.01 [−0.03, +0.06] | 0.88, +0.02 [−0.01, +0.05] | – | 10 | level |
| Sol-2B | mixed | 0.89 | 0.98, +0.09 [+0.01, +0.17] | 1.02, +0.12 [+0.06, +0.19] | 0.99, +0.09 [+0.00, +0.19] | – | 5 | better |
| Nox-4B | uniform | 0.33 | 0.35, +0.02 [−0.01, +0.04] | 0.35, +0.02 [+0.00, +0.04] | 0.37, +0.04 [+0.02, +0.06] | – | 5 | level |
| Lux-9B | uniform | 0.20 | 0.20, −0.00 [−0.01, +0.01] | 0.20, +0.00 [−0.01, +0.01] | 0.20, +0.00 [−0.01, +0.02] | – | 10 | level |

- Exact runs one request per forward on CPU (its rows are not
  batch-invariant; parity record), so concurrency adds little to it. The
  decoders were measured at C = 1 and 4 only.
- Every exact single-request row is level or better: 10 rounds for the
  encoders and the uniform Eos, Sol and Lux rows, 5 for the rest, none of
  which read worse at 5. The better cells are rates: Eos and Sol on mixed
  lengths, Nox at C = 1 and 4, and Kai and Route at C = 16.

### Router requests

Six router signals about one prompt. The decoders run exact only (their
`shared_context` shares little on these short prompts: within 1% of the
bundled runtime for Eos at IP2).

| Model | Profile | p50 bundled → runtime | p50 Δ [95% CI] | p95 bundled → runtime | p95 Δ [95% CI] | Rounds | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | `exact` | 867.8 → 933.9 | +66.1 [+5.1, +127.1] | 1,137.6 → 1,212.2 | +74.6 [+11.2, +138.1] | 10 | **worse** |
| Kai-0.6B | `batching` | 850.6 → 264.2 | −586.4 [−662.4, −510.3] | 1,091.1 → 309.5 | −781.7 [−897.3, −666.0] | 5 | better |
| Kai-0.6B | `max_speed` | 850.6 → 161.5 | −689.1 [−785.5, −592.6] | 1,091.1 → 183.9 | −907.2 [−1,064.5, −750.0] | 5 | better |
| Lex-0.6B | `exact` | 892.9 → 949.6 | +56.7 [−55.6, +169.0] | 1,148.3 → 1,195.9 | +47.6 [−70.4, +165.5] | 10 | level |
| Lex-0.6B | `batching` | 913.8 → 276.6 | −637.3 [−666.6, −607.9] | 1,165.4 → 363.9 | −801.5 [−868.6, −734.3] | 5 | better |
| Lex-0.6B | `max_speed` | 913.8 → 173.0 | −740.8 [−756.7, −725.0] | 1,165.4 → 221.9 | −943.5 [−1,061.6, −825.4] | 5 | better |
| Route-0.6B | `exact` | 864.8 → 935.0 | +70.3 [−20.1, +160.7] | 1,121.3 → 1,237.9 | +116.6 [−26.0, +259.2] | 10 | level |
| Route-0.6B | `batching` | 864.5 → 287.7 | −576.8 [−652.0, −501.6] | 1,084.3 → 370.1 | −714.2 [−801.0, −627.5] | 5 | better |
| Route-0.6B | `max_speed` | 864.5 → 180.3 | −684.2 [−747.3, −621.2] | 1,084.3 → 243.4 | −841.0 [−946.3, −735.7] | 5 | better |
| Eos-0.8B | `exact` | 6,085 → 6,347 | +262 [−288, +811] | 6,662 → 6,922 | +260 [−315, +835] | 10 | level |
| Sol-2B | `exact` | 11,554 → 11,068 | −486 [−1,177, +204] | 12,857 → 11,985 | −872 [−2,141, +396] | 10 | level |
| Nox-4B | `exact` | 28,793 → 29,426 | +633 [−319, +1,585] | 31,355 → 32,216 | +861 [−1,213, +2,934] | 10 | level |
| Lux-9B | `exact` | 46,652 → 47,042 | +391 [−774, +1,555] | 50,100 → 50,094 | −5 [−1,846, +1,835] | 10 | level |

Throughput, requests/s, as above:

| Model | Profile | Bundled | One at a time | C = 1 | C = 4 | C = 16 | Rounds | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | `exact` | 1.12 | 1.05, −0.07 [−0.11, −0.02] | 1.05, −0.06 [−0.13, −0.00] | 1.07, −0.05 [−0.13, +0.03] | 1.08, −0.04 [−0.11, +0.04] | 10 | **worse** |
| Kai-0.6B | `batching` | 1.15 | 3.73, +2.59 [+2.24, +2.93] | 3.60, +2.45 [+2.25, +2.66] | 4.70, +3.55 [+3.31, +3.78] | 5.18, +4.03 [+3.74, +4.31] | 5 | better |
| Kai-0.6B | `max_speed` | 1.15 | 6.06, +4.91 [+4.65, +5.16] | 5.69, +4.54 [+4.15, +4.94] | 7.19, +6.04 [+4.93, +7.16] | 7.82, +6.67 [+5.59, +7.75] | 5 | better |
| Lex-0.6B | `exact` | 1.09 | 1.04, −0.05 [−0.14, +0.03] | 1.03, −0.06 [−0.14, +0.02] | 1.03, −0.06 [−0.12, +0.01] | 1.05, −0.04 [−0.12, +0.05] | 10 | level |
| Lex-0.6B | `batching` | 1.07 | 3.49, +2.43 [+2.13, +2.72] | 3.58, +2.52 [+2.10, +2.93] | 4.35, +3.29 [+3.17, +3.41] | 4.76, +3.69 [+3.50, +3.89] | 5 | better |
| Lex-0.6B | `max_speed` | 1.07 | 5.56, +4.49 [+4.04, +4.94] | 5.23, +4.17 [+3.71, +4.63] | 7.32, +6.25 [+5.57, +6.92] | 7.77, +6.70 [+5.58, +7.82] | 5 | better |
| Route-0.6B | `exact` | 1.11 | 1.04, −0.07 [−0.17, +0.03] | 1.03, −0.08 [−0.16, +0.00] | 1.05, −0.06 [−0.14, +0.01] | 1.04, −0.07 [−0.15, +0.01] | 10 | level |
| Route-0.6B | `batching` | 1.12 | 3.37, +2.25 [+2.07, +2.43] | 3.42, +2.30 [+2.11, +2.49] | 4.31, +3.19 [+2.98, +3.39] | 5.04, +3.91 [+3.59, +4.24] | 5 | better |
| Route-0.6B | `max_speed` | 1.12 | 5.39, +4.27 [+3.83, +4.70] | 5.37, +4.25 [+3.48, +5.02] | 6.98, +5.86 [+4.29, +7.43] | 7.43, +6.31 [+4.17, +8.45] | 5 | better |
| Eos-0.8B | `exact` | 0.17 | 0.16, −0.01 [−0.02, +0.00] | 0.16, −0.00 [−0.02, +0.01] | – | – | 10 | level |
| Sol-2B | `exact` | 0.0865 | 0.0906, +0.0042 [−0.0013, +0.0096] | 0.0888, +0.0024 [−0.0011, +0.0059] | – | – | 10 | level |
| Nox-4B | `exact` | 0.0347 | 0.0341, −0.0007 [−0.0017, +0.0004] | 0.0338, −0.0010 [−0.0021, +0.0002] | – | – | 10 | level |
| Lux-9B | `exact` | 0.0213 | 0.0212, −0.0001 [−0.0006, +0.0003] | 0.0212, −0.0001 [−0.0008, +0.0005] | – | – | 10 | level |

- **Exact does the bundled runtime's work,** and answers it byte for byte
  (parity record). On the encoders every question type present runs its
  stack over all six rows; on the decoders the six questions are one padded
  batch of six rows, as the bundled runtime runs them.
- **One exact router row is worse: Kai's.** At 10 rounds its p50 is
  +66.1 ms [+5.1, +127.1] (867.8 → 933.9 ms, +7.6%) and its p95 +74.6 ms
  [+11.2, +138.1]; one at a time it serves −0.07 requests/s [−0.11, −0.02],
  at C = 1 −0.06 [−0.13, −0.00], and at C = 4 and 16 it is level. The
  runtime was slower at p50 in nine of the ten rounds: +116, +35, −10, +7,
  +25 ms (rounds 1–5), then +82, +279, +86, +39, +2 ms (6–10). Lex
  (+56.7 ms [−55.6, +169.0]) and Route (+70.3 ms [−20.1, +160.7]) lean the
  same way, about 7%, and are level; the four decoders are level, Sol's
  point estimate on the better side. The encoders' single requests, one
  question through the same stacks, are level, so the gap is in how a
  six-question request runs; where it goes is not found yet. This is open.
- **Kai, a second series.** A separate 10-round A/B on 16 cores with no other
  timed work on their NUMA node (node C 16–31, the first 30 router prompts,
  C = 1, the node's 1-minute load at most 44) reads it worse too: p50 +65.3 ms
  [+10.4, +120.2] (646.2 → 711.5 ms), one at a time −0.11 requests/s [−0.19,
  −0.03], C = 1 −0.11 [−0.20, −0.02], and p95 +33.5 ms [−7.0, +74.0], level.
  The bundled runtime's p50 stayed within 636–652 ms in every round; the
  runtime's was 651–677 ms in seven rounds (1–5% slower) and 807–834 ms in
  three (24–29% slower), so some fresh runtime processes run these requests in
  a slower mode.
- **Nox, a second series.** Its C = 1 rate read wholly worse at 5 and 7
  rounds in the four-lane series, then level at 8–10. Three checks looked for
  a runtime cost and found none:
  - in one process, each request on every path back to back (6 requests),
    the family's forward took 0.86 s less than the bundled runtime's 28.1 s,
    and `Runtime.call` 0.13 s more;
  - in separate processes on 16 quiet cores (144–159, NUMA node 1, 3
    requests), the runtime answered in 24.5 s through `Runtime.call` against
    the bundled runtime's 25.8 s, with 14.7 s of `aten::mm` against 16.7 s;
  - a separate 5-round A/B on those cores (the first 6 router prompts, C = 1,
    no other timed CPU work on that NUMA node): level, p50 +1,020 ms [−850,
    +2,890] on the bundled runtime's 21.8 s, p95 +529 ms [−1,189, +2,246], and
    both rates −0.00 requests/s [−0.00, +0.00].
- **Where the rounds ran:** each round's two sides on the same cores; each
  row on one lane, except Eos's router rows (rounds 1–5 on 16–31, 6–10 on
  48–63) and Lux's uniform rows (1–5 on 48–63, 6–10 on 0–15), which moved to
  the first free lane.
- **`batching`** runs each question type's stack over its own rows only,
  packed. **`max_speed`** does the same on the `float32-packed` copies of
  the three stacks (reduced copies below), with no decision changes (parity
  record).

### Opt-in profiles, single requests (encoders)

Two rounds of `max_speed` and `batching` on each encoder's lane right after
its A/B rounds (C = 1 / 4 / 16 through `Runtime.call`), against the bundled
runtime's mean over those two rounds. Means:

| Model | Lengths | Profile | p50 bundled → runtime | p95 bundled → runtime | req/s bundled → C = 1 / 4 / 16 | Rounds |
| --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | uniform | `max_speed` | 94.1 → 68.1 | 122.5 → 94.5 | 10.0 → 14.6 / 16.1 / 17.7 | 2 |
| Kai-0.6B | uniform | `batching` | 94.1 → 104.2 | 122.5 → 138.8 | 10.0 → 9.2 / 11.1 / 12.1 | 2 |
| Kai-0.6B | mixed | `max_speed` | 54.8 → 40.2 | 211.4 → 164.6 | 12.5 → 17.1 / 19.2 / 20.0 | 2 |
| Kai-0.6B | mixed | `batching` | 54.8 → 67.7 | 211.4 → 244.0 | 12.5 → 11.5 / 11.7 / 13.5 | 2 |
| Lex-0.6B | uniform | `max_speed` | 94.1 → 69.0 | 124.7 → 92.4 | 10.0 → 14.4 / 15.2 / 18.9 | 2 |
| Lex-0.6B | uniform | `batching` | 94.1 → 107.6 | 124.7 → 138.9 | 10.0 → 9.3 / 11.3 / 12.3 | 2 |
| Lex-0.6B | mixed | `max_speed` | 56.4 → 39.9 | 211.4 → 157.6 | 12.3 → 17.2 / 17.7 / 22.4 | 2 |
| Lex-0.6B | mixed | `batching` | 56.4 → 68.8 | 211.4 → 259.5 | 12.3 → 11.4 / 11.6 / 14.0 | 2 |
| Route-0.6B | uniform | `max_speed` | 96.1 → 68.5 | 124.0 → 88.2 | 9.8 → 14.5 / 17.4 / 19.6 | 2 |
| Route-0.6B | uniform | `batching` | 96.1 → 106.1 | 124.0 → 156.2 | 9.8 → 9.2 / 11.6 / 11.8 | 2 |
| Route-0.6B | mixed | `max_speed` | 57.3 → 46.5 | 217.8 → 186.7 | 12.0 → 15.6 / 20.3 / 19.3 | 2 |
| Route-0.6B | mixed | `batching` | 57.3 → 62.7 | 217.8 → 237.5 | 12.0 → 10.8 / 12.1 / 14.8 | 2 |

- **`max_speed`** runs the `float32-packed` copies of the three stacks
  (reduced copies below): single requests 1.2–1.4× faster at p50 and p95,
  and 1.6–2.0× the bundled rate at C = 16.
- **`batching`** holds the queue for the batching window to coalesce
  concurrent requests, so one request at a time pays the window: p50 +5 to
  +14 ms, and 0.9× the bundled rate at C = 1. On Route four of these cells
  read wholly worse over the two rounds: one at a time −1.05 req/s [−1.41,
  −0.69] (uniform) and −0.98 [−1.02, −0.94] (mixed), C = 1 −0.67 [−1.19,
  −0.15] (uniform), and the mixed p95 +19.8 ms [+13.4, +26.1]. It gains with
  concurrency (1.1–1.2× at C = 16); on CPU, `max_speed` is faster at every
  concurrency. Neither profile is a default; `exact` is.

### What changed on CPU since `c22bb15cd`

Each change measured in alternating rounds of fresh processes on the same
cores, before the A/B above:

- **Weights in process memory** (`69ae2d0c5`, Where a request's time goes):
  exact went from 0.84–0.92× of the bundled rate on short prompts (one round
  at `d3d1d7e68`) to level.
- **oneDNN keeps 8,192 compiled primitives** (`7292622f3`). A packed linear
  sees a new row count for almost every coalesced batch, and at the default
  1,024 entries it recompiled on most calls: 620–664 µs per call across
  1,088 distinct row counts against 300–360 µs across 16. Before, 4 of 12
  `max_speed` runs at C = 4 or 16 fell to 8.8–14 requests/s, below one
  request at a time; after, none of 18 did.
- **Coalesced batches stay within 1,024 padded tokens** (`2774d97dc`): at
  C = 16 on uniform lengths, 16.9 → 24.3 (Kai) and 16.2 → 24.4 (Lex)
  requests/s, 1–7% lower at C = 4; 1,536 and 2,048 were lower or erratic.
- **Type heads read coalesced rows by length group** (`eb8fe8582`): at C = 16
  on mixed lengths, 18.8 → 25.5 (Kai) and 19.2 → 28.4 (Lex) requests/s. On
  MI325X it gained nothing and lost up to 9% at C = 4, so GPUs keep one
  padded read.

## Reduced copies under `max_speed`

Under `max_speed` an encoder may run approximate batches on a copy of its
linear layers (design §5.4). A package consents only where its records show
at least 99% label agreement with exact and a faster path. Each copy below
replaced the three stacks' linear layers: dynamic int8, BF16, or FP32
weights that oneDNN pre-packs (`float32-packed`). Norms, softmax and heads
stayed FP32. Both versions ran in one process, interleaved per request, on the
exact profile's batches. Agreement means the same label (the choice, Noul class
or Score level) as the FP32 path.

| Copy (device, questions) | Kai | Lex | Route | p50 vs FP32 |
| --- | --- | --- | --- | --- |
| Dynamic int8, per-tensor weights (CPU, 500) | 51.8% | 52.4% | 63.8% | 1.53–1.57× faster |
| Dynamic int8, per-channel weights (CPU, 500) | 52.0% | 48.0% | 59.8% | 1.55–1.66× faster |
| BF16 autocast (CPU with AVX-512 BF16, 2,987) | 98.26% | 99.26% | 98.02% | 1.26–1.34× faster |
| BF16 autocast (MI325X, 10,605) | 98.75% | 99.33% | 98.60% | 1.16× slower |
| `float32-packed`: oneDNN pre-packed FP32 linears (CPU, 2,987) | 100% | 100% | 100% | 1.52–1.53× faster |

- **int8 breaks the models:** max |Δp| reaches 0.56–0.87, with or without
  per-channel weights. The likely cause is 8-bit activations against
  ModernBERT's activation outliers.
- **BF16's weak spot is Score:** 93.7–95.0% agreement on Kai and Route, where
  many questions are near-ties. Max |Δp| stays at 0.027–0.056.
- **GPU BF16 is slower:** a single-request encoder forward is launch-bound, so
  the casts autocast adds cost more than the BF16 matrix units save.
- **`float32-packed` keeps every decision:** max |Δp| 2.4e-5, while its
  linears run through oneDNN's pre-packed kernel instead of MKL's.
- **So Kai, Lex and Route consent to `float32-packed` on CPU, and to no GPU
  copy** (`BuiltinModel.reduced`). The exact profile keeps MKL, so its answers
  stay byte-identical to the bundled runtime.
- **The served copy (`c22bb15cd`):** the engine loads one copy per layer stack
  (1.32 GB of pre-packed linears for the three), and `max_speed` runs
  `batching` on it. Through the runtime it changes no decision in 1,431
  requests per encoder against the bundled runtime (parity record) and serves
  single requests 1.3–1.5× and router requests 4.1–5.5× faster than the bundled
  runtime (CPU sections). A host that cannot run the copy (no oneDNN) serves
  `max_speed` without it, and the receipt says why.

## Where a request's time goes

From the paired single-request runs, p50:

- **Planning** (validation, rendering, tokenization) takes 0.6 ms on CPU (Kai,
  typed-final).
- **The scheduler** adds 0.05–0.95 ms on ROCm over running the model directly
  on the same thread. On CPU the difference is within the noise.
- **The API layer** adds up to 0.9 ms on ROCm and 0.3–1.4 ms on CPU for the
  encoders. Small requests are planned on the event loop.
- **The family's own work** (planning plus `run` on the released batches) is
  faster than the bundled runtime's `system_one` on the same requests: by
  4–22 ms on ROCm. On CPU, where both run the same FP32 math through the same
  MKL kernels, it ranges from 3.4% faster (Route) to 0.4% slower (Eos).
- **A short CPU request** (Kai, one Noul question about a 7-word prompt, 16
  cores, every path on the device thread of one process with one OpenMP team):
  planning 0.22 ms, the family's `run` 31.05 ms, through the scheduler 31.67
  ms, through `Runtime.call` 32.98 ms, while the bundled `system_one` took
  26.95 ms. The profile showed why: the same 88 `aten::mm` calls took 222 µs
  each against the bundled runtime's 169 µs, because the native backbone's
  FP32 weights were views of the checkpoint's file mapping. Since `69ae2d0c5`
  they are copied into process memory once at load, with identical answers.
  End to end, in fresh processes alternating on the same cores (3 rounds,
  p50), the request went from 30.1–31.7 ms to 25.5–26.2 ms, against
  24.7–25.4 ms for the bundled runtime called directly on its main thread.
  The remaining ~0.9 ms is the API layer.

## What makes it fast

- **Encoders:**
  - one embedding feeds the three stacks (branches of one backbone);
  - the additive attention masks are built once per forward;
  - `batching` runs each question type's stack over that type's rows only,
    packed back to back with no padding (`run_approximate`);
  - on GPUs every stack replays its own bucket graphs (approximate profiles);
  - `max_speed` on CPU runs a copy of every stack whose linears oneDNN
    pre-packed once at load;
  - on CPUs, coalesced batches stay within 1,024 padded tokens, and the type
    heads read coalesced rows by length group instead of padded to the longest;
  - the weights live in the process's memory, not in the checkpoint's file
    mapping;
  - on CPU, every forward runs on one device thread, so one thread pool serves
    them all.
- **Decoders:**
  - every layer runs the fused gfx942 kernels on the BF16 stream, bit for bit;
  - HIP graphs for exact shapes;
  - Eos's FP64-accumulating convolution only where the bundled runtime uses
    it;
  - pinned FLA kernel choices;
  - `shared_context` computes a request's shared prompt once;
  - the 2 GiB guard keeps every GPU forward within FLA's 32-bit offsets.
- **Runtime:** small requests are planned on the event loop, and only the
  coalescing profiles hold the queue for the batching window.

## Reproduce

```bash
python3 tools/decision1_bench.py paired --package PACKAGE_DIR --model vllm-sr/Decision-1.0-Kai-0.6B \
  --cache-dir HF_CACHE --device rocm:0|cpu [--threads 16] [--profile batching|max_speed] \
  --prompts typed-final:PROMPTS.jsonl:400 [--router QUESTIONS.json] --output paired.json
# An A/B round: `reference` and `native`, each in its own fresh process on the same cgroup
# cpuset, the order alternating every round; at least 5 rounds.
python3 tools/decision1_bench.py native --model vllm-sr/Decision-1.0-Kai-0.6B --cache-dir HF_CACHE \
  --device rocm:0|cpu --profile exact|batching|shared_context|max_speed --prompts typed-final:PROMPTS.jsonl:400 \
  --concurrency 1 4 16 --output native.json
python3 tools/decision1_bench.py reference --package PACKAGE_DIR --repo vllm-sr/Decision-1.0-Kai-0.6B \
  --device cuda:0|cpu --prompts many64:MANY64.jsonl:20 --warmup 4 --output reference.json
```
