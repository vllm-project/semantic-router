# Decision 1.0 performance

The `decision1` family against each Decision 1.0 package's bundled runtime,
on the same node, inputs and devices.

- **On CPU, exact is level with the bundled runtime or better on every row of
  the tables but one, Kai's router requests (7.6% slower at p50, wholly). With
  the huge-page default (`129be34ea`) the slow mode behind it is gone;
  re-timed alone, Kai's, Lex's and Route's router requests are 3.3%, 2.6% and
  2.1% slower at p50 and Kai's single-request p95 2.8%, wholly, and open**
  (CPU section). libgomp's default spin count, which `vllm-srun serve` now
  keeps for these models, takes Kai's p50 gap from 3.3% to 2.1%. It leaves
  Lex's and Route's at 3.5% and 6.6%, because the bundled runtime gains
  from it too.
  Every CPU row is an interleaved A/B with a 95% interval on the difference:
  single requests on uniform and mixed lengths for all seven models, their
  throughput, and router requests. Both sides run the same FP32 math through
  the same MKL kernels, so exact can't be much faster; the single requests are
  level on all seven models, and the decoders' rates are a little better where
  the interval says so (Eos +0.23 requests/s [+0.07, +0.39] and Sol +0.09
  [+0.01, +0.17] on mixed lengths, Nox +0.02 [+0.00, +0.04] at C = 1).
- **On CPU the encoders' opt-in profiles win.** For six router signals about
  one prompt, `batching` answers 3.0–3.3× and `max_speed` 4.8–5.3× faster at
  p50 than the bundled runtime, and they serve 3.0–3.3× and 4.8–5.3× its
  rate one request at a time (up to 4.5× and 7.3× at C = 16). `max_speed`
  runs the encoders on `float32-packed` copies of their three stacks: single
  requests 1.2–1.4× faster at p50, up to 2.0× the bundled throughput, and no
  decision changes in 1,431 requests per encoder.
- **On ROCm, exact and faster, in the router's image:** in interleaved A/Bs (5
  rounds, 95% intervals) in `a580be6b9` (vLLM's ROCm PyTorch, on which the
  runtime answers byte-identically to the release), no row is worse. Exact
  serves a single request 1.5–1.7× faster at p50 on the encoders and 1.3–3.1×
  on the decoders, end to end through the API; for six router signals about
  one prompt, `batching` cuts the encoders' p95 by 43–48% and `shared_context`
  the decoders' by 46–76%.

- **CPU:** node C, 16 cores of an AMD EPYC 9575F (Zen 5) per run, each side
  in its own docker cgroup cpuset (the effective cpuset is logged), CPU
  PyTorch 2.10.0 with MKL (the router image's pin), Transformers 5.17 for the
  bundled side, 16 threads on both sides (`OMP_NUM_THREADS` and
  `MKL_NUM_THREADS` too).
- **ROCm:** one AMD Instinct MI325X (gfx942) per run, on 8 host cores of the
  GPU's NUMA node, in the router's ROCm image
  `Dockerfile.extproc` at `a580be6b9` (`ACCELERATOR=rocm`): PyTorch 2.12.0+git6bbd260 with
  AOTriton 0.13.50 and the ROCm 7.2.3 libraries from vLLM's ROCm image, Triton
  3.7.0, FLA 0.5.2 and causal-conv1d 1.7.0, with Transformers 5.17.0 added for
  the bundled side.
- **Bundled side:** the package's runtime (Transformers remote code,
  `system_one`), a library that answers one request at a time, so its
  sequential rate is its throughput. On ROCm it runs the same FLA kernel
  choices as the native side.
- **Commits:** the CPU A/B is at `8fc0bf02b` (S1 and IP3a merged). Since then
  the CPU request path changed only by the scheduler's cancellation
  bookkeeping (`1ce0bafc9`: one `cancelled()` check per answered job, one
  `done()` check per planned job). The device lock, the per-model kernel
  choices and the thread-local graph capture are GPU-only, and load retries,
  placement by plugin name and the advertised limits run at load. `a1e1b4ccb`
  also freezes the heap after each load pass, which leaves fewer objects for a
  request's garbage collections to walk: it can only shorten pauses. So the
  run stands for the head, except where the huge-page default (`129be34ea`)
  changes how CPU memory is allocated: the rows re-timed under it are in the
  CPU section. The ROCm rows' commits are in their section.
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

Every row is the 16:53-standard A/B in the router's ROCm image,
`Dockerfile.extproc` at `a580be6b9` (`ACCELERATOR=rocm`): PyTorch
2.12.0+git6bbd260 (AOTriton 0.13.50) and the ROCm 7.2.3 libraries from vLLM's
ROCm image, Triton 3.7.0, FLA 0.5.2 and `causal-conv1d` 1.7.0. Both sides run
in that image, with Transformers 5.17.0 and `regex` appended to `PYTHONPATH`
for the bundled side. The runtime is `1bd99cd37`, one AMD Instinct MI325X
(gfx942) per run. On this image the runtime's answers are byte-identical to
the release (parity record). These rows replace the same rows timed on the
official-wheel stack (`be7366c49`, the packages of `af71d5e82`, runtime
`a1c7a284e`), which were level or better everywhere too; their runs and
intervals stay in `decision1-performance.json` (`intervals` keys
`rocm-official-wheel*`). They ran before the huge-page default (`129be34ea`),
which changes only CPU-side allocations on a GPU.

- **Cores:** each GPU's host threads run in a docker cgroup cpuset of 8 cores
  on the GPU's own NUMA node that no other timed job used (GPU1 0–7, GPU2
  8–15, GPU5 144–151, GPU7 152–159); the effective cpuset is logged.
- **Sides:** the bundled runtime, the runtime's exact profile and its
  approximate profile, each in its own process, the order rotated every round.
  Five rounds per row, the standard's minimum: no row read worse, so none got
  rounds 6–10. Two sets of rounds ran again after the queue, on their own GPUs
  and host cores, and their first runs are not used: GPU5's round 1 of Kai's
  many-question rows, which a CPU job of the re-timing's own overlapped; and
  every round with a side that started from 17:54 to 17:58 UTC, while another
  job loaded an image on node C (many questions, Nox-4B and Lux-9B round 5;
  router throughput, Eos-0.8B rounds 4–5, Kai-0.6B rounds 2–3, Lex-0.6B round
  2, Lux-9B and Route-0.6B rounds 1–2, and Sol-2B round 4).

### Single requests

The first 400 typed-final prompts, one Choice question each:

| Model | Profile | p50 bundled → runtime | p50 Δ [95% CI] | p95 bundled → runtime | p95 Δ [95% CI] | Rounds | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | `exact` | 9.03 → 6.10 | −2.92 [−3.12, −2.72] | 14.2 → 8.1 | −6.1 [−14.2, +2.0] | 5 | level |
|  | `batching` | 9.03 → 6.68 | −2.34 [−2.53, −2.16] | 14.2 → 7.0 | −7.2 [−14.2, −0.2] | 5 | better |
| Lex-0.6B | `exact` | 10.4 → 6.1 | −4.2 [−8.0, −0.5] | 14.9 → 10.8 | −4.1 [−13.4, +5.2] | 5 | level |
|  | `batching` | 10.4 → 6.8 | −3.6 [−7.4, +0.2] | 14.9 → 8.6 | −6.3 [−11.2, −1.4] | 5 | level |
| Route-0.6B | `exact` | 8.94 → 6.06 | −2.88 [−3.10, −2.65] | 11.4 → 7.9 | −3.5 [−8.0, +0.9] | 5 | level |
|  | `batching` | 8.94 → 6.76 | −2.17 [−2.41, −1.94] | 11.4 → 11.6 | +0.1 [−6.3, +6.6] | 5 | level |
| Eos-0.8B | `exact` | 23.0 → 8.1 | −14.9 [−15.8, −14.0] | 30.4 → 11.4 | −19.0 [−26.5, −11.4] | 5 | better |
|  | `shared_context` | 23.0 → 9.6 | −13.4 [−17.8, −9.0] | 30.4 → 16.1 | −14.4 [−32.3, +3.6] | 5 | level |
| Sol-2B | `exact` | 23.5 → 7.6 | −15.9 [−17.8, −13.9] | 28.6 → 12.6 | −16.0 [−26.0, −6.0] | 5 | better |
|  | `shared_context` | 23.5 → 7.6 | −15.9 [−17.7, −14.1] | 28.6 → 11.2 | −17.4 [−25.0, −9.7] | 5 | better |
| Nox-4B | `exact` | 29.3 → 14.9 | −14.5 [−19.5, −9.4] | 35.4 → 18.6 | −16.8 [−33.6, +0.0] | 5 | level |
|  | `shared_context` | 29.3 → 14.9 | −14.4 [−19.1, −9.8] | 35.4 → 17.2 | −18.1 [−23.6, −12.6] | 5 | better |
| Lux-9B | `exact` | 29.5 → 22.2 | −7.3 [−13.3, −1.3] | 36.7 → 29.8 | −7.0 [−21.2, +7.3] | 5 | level |
|  | `shared_context` | 29.5 → 18.5 | −10.9 [−11.3, −10.6] | 36.7 → 21.1 | −15.6 [−27.5, −3.7] | 5 | better |

- Every exact p50 interval is on the runtime's side: 1.5–1.7× faster on the
  encoders (Kai 9.03 → 6.10 ms) and 1.3–3.1× on the decoders (Eos 23.0 → 8.1,
  Sol 23.5 → 7.6, Nox 29.3 → 14.9, Lux 29.5 → 22.2). The level verdicts come
  from p95 intervals that straddle zero, and from Lex's `batching` p50 (−3.6
  ms [−7.4, +0.2]). The one positive point estimate, Route's `batching` p95
  (+0.1 ms), is level.

### Router requests

The six router signals about each of the 231 public231 prompts:

| Model | Profile | p50 bundled → runtime | p50 Δ [95% CI] | p95 bundled → runtime | p95 Δ [95% CI] | Rounds | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | `exact` | 27.0 → 22.7 | −4.3 [−9.2, +0.5] | 63.6 → 48.9 | −14.7 [−26.3, −3.1] | 5 | level |
|  | `batching` | 27.0 → 15.6 | −11.4 [−16.6, −6.2] | 63.6 → 33.2 | −30.4 [−44.0, −16.9] | 5 | better |
| Lex-0.6B | `exact` | 30.0 → 32.2 | +2.2 [−4.8, +9.2] | 67.6 → 73.3 | +5.7 [−24.4, +35.8] | 5 | level |
|  | `batching` | 30.0 → 15.3 | −14.7 [−20.0, −9.3] | 67.6 → 36.6 | −30.9 [−53.9, −7.9] | 5 | better |
| Route-0.6B | `exact` | 28.5 → 21.0 | −7.5 [−11.7, −3.3] | 61.9 → 52.3 | −9.6 [−32.3, +13.1] | 5 | level |
|  | `batching` | 28.5 → 14.1 | −14.4 [−20.9, −7.9] | 61.9 → 35.3 | −26.6 [−44.6, −8.7] | 5 | better |
| Eos-0.8B | `exact` | 26.9 → 24.5 | −2.4 [−9.8, +5.0] | 144.1 → 140.2 | −3.8 [−33.6, +25.9] | 5 | level |
|  | `shared_context` | 26.9 → 23.4 | −3.5 [−8.4, +1.5] | 144.1 → 78.5 | −65.5 [−72.2, −58.9] | 5 | level |
| Sol-2B | `exact` | 42.2 → 27.8 | −14.4 [−22.5, −6.4] | 209.4 → 168.3 | −41.1 [−100.4, +18.2] | 5 | level |
|  | `shared_context` | 42.2 → 34.9 | −7.4 [−13.4, −1.3] | 209.4 → 71.1 | −138.2 [−164.6, −111.9] | 5 | better |
| Nox-4B | `exact` | 82.6 → 68.5 | −14.1 [−38.5, +10.3] | 488.7 → 356.0 | −132.6 [−187.8, −77.4] | 5 | level |
|  | `shared_context` | 82.6 → 59.9 | −22.6 [−45.4, +0.1] | 488.7 → 117.9 | −370.8 [−418.9, −322.7] | 5 | level |
| Lux-9B | `exact` | 121.0 → 107.3 | −13.7 [−57.4, +30.0] | 660.0 → 570.5 | −89.4 [−220.3, +41.4] | 5 | level |
|  | `shared_context` | 121.0 → 92.7 | −28.3 [−38.2, −18.3] | 660.0 → 168.2 | −491.8 [−584.9, −398.7] | 5 | better |

- Exact runs the released shapes, so on the encoders it does the bundled
  runtime's work: every question type present runs its stack over all six
  rows. `batching` runs each stack over its own rows only, packed, and every
  layer stack replays its own bucket graphs.
- `shared_context` computes the prompt once for all six questions. The
  decoders' long prompts are where the bundled runtime's p95 comes from.
- Exact is level or better on every model, with p50 9–34% below the bundled
  runtime's, except Lex's: +2.2 ms [−4.8, +9.2] at p50 and +5.7 ms [−24.4,
  +35.8] at p95, level, the section's only positive point estimates.
  `batching` cuts the encoders' p95 by 43–48% and `shared_context` the
  decoders' by 46–76% (Lux 660.0 → 168.2 ms).

### Throughput (requests/s)

Single requests in waves of C through `Runtime.call`, against the bundled
runtime's sequential rate in the same round:

| Model | Profile | Bundled | C = 1 | C = 4 | C = 16 | Rounds | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | `exact` | 100.1 | 157.4, +57.3 [+37.3, +77.4] | 154.7, +54.6 [+22.2, +87.0] | 157.2, +57.1 [+18.0, +96.1] | 5 | better |
|  | `batching` | 100.1 | 140.3, +40.2 [+13.0, +67.3] | 312.4, +212.3 [+154.8, +269.8] | 500.2, +400.1 [+319.5, +480.7] | 5 | better |
| Lex-0.6B | `exact` | 95.3 | 135.4, +40.1 [+0.9, +79.4] | 162.0, +66.7 [+47.6, +85.9] | 164.7, +69.4 [+46.8, +92.1] | 5 | better |
|  | `batching` | 95.3 | 131.3, +36.0 [+5.4, +66.7] | 319.6, +224.3 [+171.3, +277.3] | 508.1, +412.8 [+340.5, +485.2] | 5 | better |
| Route-0.6B | `exact` | 105.3 | 160.0, +54.7 [+43.8, +65.5] | 154.1, +48.8 [+28.0, +69.5] | 154.4, +49.1 [+25.3, +72.9] | 5 | better |
|  | `batching` | 105.3 | 146.8, +41.5 [+37.1, +45.9] | 336.8, +231.5 [+221.7, +241.4] | 509.0, +403.7 [+336.5, +470.9] | 5 | better |
| Eos-0.8B | `exact` | 41.1 | 115.6, +74.4 [+59.0, +89.9] | 117.6, +76.4 [+65.5, +87.4] | 111.3, +70.2 [+44.6, +95.7] | 5 | better |
|  | `shared_context` | 41.1 | 121.7, +80.6 [+76.3, +84.9] | 124.6, +83.5 [+81.6, +85.4] | 119.0, +77.8 [+57.6, +98.1] | 5 | better |
| Sol-2B | `exact` | 41.5 | 129.5, +88.0 [+81.1, +94.9] | 133.4, +91.9 [+88.2, +95.7] | 127.6, +86.1 [+68.1, +104.1] | 5 | better |
|  | `shared_context` | 41.5 | 129.1, +87.6 [+83.0, +92.2] | 131.8, +90.3 [+84.2, +96.4] | 134.8, +93.3 [+90.2, +96.4] | 5 | better |
| Nox-4B | `exact` | 32.9 | 63.9, +31.0 [+16.5, +45.4] | 74.0, +41.1 [+38.6, +43.6] | 75.2, +42.2 [+40.4, +44.0] | 5 | better |
|  | `shared_context` | 32.9 | 61.3, +28.3 [+12.0, +44.7] | 74.3, +41.4 [+38.8, +44.0] | 69.3, +36.3 [+23.7, +49.0] | 5 | better |
| Lux-9B | `exact` | 32.3 | 50.4, +18.1 [+11.2, +24.9] | 48.5, +16.2 [+8.0, +24.4] | 53.2, +20.9 [+18.0, +23.8] | 5 | better |
|  | `shared_context` | 32.3 | 50.8, +18.5 [+16.5, +20.4] | 44.4, +12.1 [+4.5, +19.7] | 50.3, +18.0 [+13.2, +22.8] | 5 | better |

Router requests:

| Model | Profile | Bundled | C = 1 | C = 4 | C = 16 | Rounds | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | `exact` | 32.4 | 43.5, +11.1 [+2.1, +20.2] | 38.9, +6.5 [−6.4, +19.5] | 38.0, +5.7 [−5.6, +16.9] | 5 | level |
|  | `batching` | 32.4 | 72.1, +39.7 [+30.5, +49.0] | 106.4, +74.0 [+65.4, +82.6] | 132.6, +100.3 [+90.6, +110.0] | 5 | better |
| Lex-0.6B | `exact` | 34.0 | 40.3, +6.4 [−0.8, +13.5] | 44.0, +10.0 [+6.2, +13.8] | 44.2, +10.3 [+6.5, +14.0] | 5 | level |
|  | `batching` | 34.0 | 68.0, +34.0 [+27.0, +41.1] | 101.1, +67.1 [+54.3, +80.0] | 125.0, +91.1 [+74.0, +108.1] | 5 | better |
| Route-0.6B | `exact` | 33.2 | 41.0, +7.7 [−3.5, +19.0] | 39.4, +6.1 [−6.9, +19.2] | 37.4, +4.2 [−3.2, +11.5] | 5 | level |
|  | `batching` | 33.2 | 62.3, +29.1 [+14.4, +43.7] | 88.7, +55.5 [+28.4, +82.6] | 112.6, +79.4 [+46.9, +111.9] | 5 | better |
| Eos-0.8B | `exact` | 19.3 | 22.4, +3.1 [−1.0, +7.3] | 24.6, +5.3 [+3.1, +7.5] | 23.6, +4.3 [−1.6, +10.2] | 5 | level |
|  | `shared_context` | 19.3 | 33.7, +14.5 [+11.5, +17.5] | 32.7, +13.4 [+9.1, +17.8] | 33.1, +13.9 [+8.7, +19.0] | 5 | better |
| Sol-2B | `exact` | 14.9 | 17.0, +2.1 [−1.1, +5.2] | 18.5, +3.6 [+2.9, +4.3] | 16.5, +1.6 [−4.6, +7.8] | 5 | level |
|  | `shared_context` | 14.9 | 24.9, +10.0 [+5.9, +14.1] | 25.9, +11.0 [+6.7, +15.4] | 27.2, +12.4 [+9.8, +14.9] | 5 | better |
| Nox-4B | `exact` | 7.26 | 8.49, +1.24 [+0.77, +1.71] | 8.52, +1.26 [+0.80, +1.73] | 8.57, +1.31 [+0.83, +1.79] | 5 | better |
|  | `shared_context` | 7.26 | 15.07, +7.8 [+7.7, +7.9] | 14.68, +7.4 [+6.1, +8.7] | 15.37, +8.1 [+8.0, +8.2] | 5 | better |
| Lux-9B | `exact` | 4.93 | 5.51, +0.58 [+0.35, +0.80] | 5.44, +0.51 [+0.06, +0.95] | 5.71, +0.78 [+0.59, +0.97] | 5 | better |
|  | `shared_context` | 4.93 | 11.44, +6.5 [+6.2, +6.8] | 10.77, +5.8 [+4.9, +6.7] | 11.21, +6.3 [+5.3, +7.3] | 5 | better |

- Exact runs one request per forward, so concurrency adds little to it.
- `batching` coalesces the questions of concurrent requests, and the encoders
  pack them without padding. `shared_context` doesn't coalesce requests.
- At C = 1, exact serves single requests at 1.4–1.6× the bundled rate on the
  encoders and 1.6–3.1× on the decoders, and router requests at 1.1–1.3×. At
  C = 16 `batching` reaches 4.8–5.3× on single requests and 3.4–4.1× on router
  requests, and `shared_context` serves router requests at 1.7–2.3× from
  C = 1. The five level rows, the encoders', Eos's and Sol's exact router rows,
  each straddle zero at one to three of the concurrencies.

### Many questions about one input

The public request of `tools/many_questions.py` (one ticket) at 16, 64 and 128
questions, 20 runs per side and round:

| Model | Questions | Profile | p50 bundled → runtime | p50 Δ [95% CI] | p95 bundled → runtime | p95 Δ [95% CI] | Rounds | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | 16 | `exact` | 59.5 → 49.0 | −10.5 [−15.8, −5.2] | 69.9 → 57.0 | −12.9 [−32.2, +6.5] | 5 | level |
|  | 16 | `batching` | 59.5 → 37.0 | −22.5 [−28.7, −16.4] | 69.9 → 48.0 | −21.9 [−65.7, +21.8] | 5 | level |
|  | 64 | `exact` | 144.1 → 121.7 | −22.3 [−52.5, +7.8] | 146.1 → 162.2 | +16.1 [−93.5, +125.8] | 5 | level |
|  | 64 | `batching` | 144.1 → 100.4 | −43.6 [−61.3, −25.9] | 146.1 → 106.7 | −39.4 [−69.5, −9.3] | 5 | better |
|  | 128 | `exact` | 270.0 → 199.1 | −70.9 [−88.4, −53.3] | 339.6 → 210.4 | −129.2 [−238.3, −20.1] | 5 | better |
|  | 128 | `batching` | 270.0 → 197.8 | −72.2 [−108.9, −35.5] | 339.6 → 228.8 | −110.8 [−226.5, +5.0] | 5 | level |
| Lex-0.6B | 16 | `exact` | 64.5 → 47.0 | −17.6 [−30.7, −4.5] | 74.7 → 47.4 | −27.3 [−44.6, −10.0] | 5 | better |
|  | 16 | `batching` | 64.5 → 33.4 | −31.2 [−44.4, −18.0] | 74.7 → 33.7 | −41.0 [−58.6, −23.4] | 5 | better |
|  | 64 | `exact` | 144.5 → 110.0 | −34.5 [−35.6, −33.4] | 164.2 → 110.7 | −53.5 [−85.8, −21.2] | 5 | better |
|  | 64 | `batching` | 144.5 → 94.1 | −50.4 [−51.0, −49.8] | 164.2 → 102.1 | −62.1 [−102.4, −21.8] | 5 | better |
|  | 128 | `exact` | 292.0 → 199.7 | −92.3 [−174.4, −10.2] | 346.5 → 221.5 | −125.0 [−307.3, +57.4] | 5 | level |
|  | 128 | `batching` | 292.0 → 192.6 | −99.4 [−199.1, +0.3] | 346.5 → 218.9 | −127.6 [−321.4, +66.2] | 5 | level |
| Route-0.6B | 16 | `exact` | 64.9 → 56.8 | −8.0 [−21.2, +5.2] | 72.0 → 61.5 | −10.5 [−30.3, +9.2] | 5 | level |
|  | 16 | `batching` | 64.9 → 36.7 | −28.2 [−47.4, −9.0] | 72.0 → 37.7 | −34.3 [−56.5, −12.1] | 5 | better |
|  | 64 | `exact` | 180.6 → 111.4 | −69.2 [−116.2, −22.2] | 204.3 → 122.6 | −81.7 [−153.2, −10.2] | 5 | better |
|  | 64 | `batching` | 180.6 → 110.4 | −70.3 [−119.1, −21.4] | 204.3 → 163.1 | −41.2 [−196.9, +114.4] | 5 | level |
|  | 128 | `exact` | 262.3 → 199.0 | −63.3 [−64.2, −62.4] | 282.1 → 202.1 | −80.0 [−101.5, −58.5] | 5 | better |
|  | 128 | `batching` | 262.3 → 206.5 | −55.8 [−103.2, −8.3] | 282.1 → 219.1 | −63.0 [−118.3, −7.7] | 5 | better |
| Eos-0.8B | 16 | `exact` | 56.0 → 47.8 | −8.1 [−12.9, −3.4] | 66.0 → 49.8 | −16.1 [−37.5, +5.3] | 5 | level |
|  | 16 | `shared_context` | 56.0 → 48.0 | −8.0 [−26.5, +10.5] | 66.0 → 51.0 | −14.9 [−34.4, +4.6] | 5 | level |
|  | 64 | `exact` | 226.0 → 193.4 | −32.6 [−70.1, +4.8] | 250.3 → 231.5 | −18.8 [−113.5, +75.9] | 5 | level |
|  | 64 | `shared_context` | 226.0 → 77.5 | −148.5 [−197.2, −99.8] | 250.3 → 82.7 | −167.6 [−236.0, −99.2] | 5 | better |
|  | 128 | `exact` | 406.9 → 330.6 | −76.3 [−94.0, −58.7] | 476.0 → 435.2 | −40.9 [−346.3, +264.6] | 5 | level |
|  | 128 | `shared_context` | 406.9 → 140.5 | −266.4 [−295.9, −237.0] | 476.0 → 144.0 | −332.0 [−430.7, −233.4] | 5 | better |
| Sol-2B | 16 | `exact` | 73.1 → 57.0 | −16.1 [−16.3, −15.9] | 73.8 → 58.3 | −15.5 [−16.4, −14.6] | 5 | better |
|  | 16 | `shared_context` | 73.1 → 37.4 | −35.7 [−38.9, −32.5] | 73.8 → 44.7 | −29.1 [−36.2, −22.0] | 5 | better |
|  | 64 | `exact` | 335.0 → 266.1 | −68.9 [−197.9, +60.0] | 378.7 → 290.6 | −88.2 [−250.0, +73.7] | 5 | level |
|  | 64 | `shared_context` | 335.0 → 85.2 | −249.8 [−388.6, −111.0] | 378.7 → 90.2 | −288.5 [−494.4, −82.6] | 5 | better |
|  | 128 | `exact` | 608.0 → 442.2 | −165.8 [−278.8, −52.8] | 626.2 → 511.2 | −115.0 [−293.6, +63.5] | 5 | level |
|  | 128 | `shared_context` | 608.0 → 157.9 | −450.1 [−562.8, −337.5] | 626.2 → 164.2 | −462.0 [−583.8, −340.3] | 5 | better |
| Nox-4B | 16 | `exact` | 156.0 → 118.7 | −37.3 [−71.1, −3.5] | 183.3 → 120.7 | −62.6 [−117.5, −7.7] | 5 | better |
|  | 16 | `shared_context` | 156.0 → 54.9 | −101.1 [−135.1, −67.1] | 183.3 → 58.7 | −124.5 [−177.8, −71.3] | 5 | better |
|  | 64 | `exact` | 570.1 → 470.7 | −99.3 [−99.9, −98.7] | 589.0 → 609.3 | +20.3 [−186.7, +227.3] | 5 | level |
|  | 64 | `shared_context` | 570.1 → 184.5 | −385.6 [−434.4, −336.8] | 589.0 → 204.8 | −384.2 [−431.6, −336.8] | 5 | better |
|  | 128 | `exact` | 1,140 → 939 | −201 [−219, −183] | 1,298 → 1,222 | −76 [−631, +479] | 5 | level |
|  | 128 | `shared_context` | 1,140 → 315 | −825 [−827, −823] | 1,298 → 370 | −928 [−1,238, −618] | 5 | better |
| Lux-9B | 16 | `exact` | 205.6 → 215.8 | +10.1 [−54.6, +74.9] | 218.1 → 225.3 | +7.2 [−61.1, +75.4] | 5 | level |
|  | 16 | `shared_context` | 205.6 → 83.1 | −122.5 [−141.0, −104.1] | 218.1 → 95.9 | −122.3 [−143.9, −100.6] | 5 | better |
|  | 64 | `exact` | 817.4 → 712.2 | −105.2 [−108.0, −102.5] | 841.0 → 947.8 | +106.8 [−189.4, +402.9] | 5 | level |
|  | 64 | `shared_context` | 817.4 → 275.4 | −542.0 [−616.6, −467.5] | 841.0 → 295.5 | −545.4 [−647.4, −443.5] | 5 | better |
|  | 128 | `exact` | 1,639 → 1,412 | −226 [−232, −220] | 1,975 → 1,579 | −396 [−950, +159] | 5 | level |
|  | 128 | `shared_context` | 1,639 → 437 | −1,202 [−1,207, −1,196] | 1,975 → 477 | −1,498 [−1,949, −1,046] | 5 | better |

- Exact is level or better at every size. At 128 questions `shared_context`
  answers 2.9–3.9× faster at p50 on the decoders (Eos 407 → 141 ms, Lux 1,639
  → 437) and `batching` 1.3–1.5× on the encoders. Positive point estimates
  appear only in five exact cells, all level: Kai's and Nox's p95 at 64
  questions, Lux's p50 and p95 at 16 and its p95 at 64.

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
- **Before the huge-page default (below), one exact router row is worse:
  Kai's.** At 10 rounds its p50 is
  +66.1 ms [+5.1, +127.1] (867.8 → 933.9 ms, +7.6%) and its p95 +74.6 ms
  [+11.2, +138.1]; one at a time it serves −0.07 requests/s [−0.11, −0.02],
  at C = 1 −0.06 [−0.13, −0.00], and at C = 4 and 16 it is level. The
  runtime was slower at p50 in nine of the ten rounds: +116, +35, −10, +7,
  +25 ms (rounds 1–5), then +82, +279, +86, +39, +2 ms (6–10). Lex
  (+56.7 ms [−55.6, +169.0]) and Route (+70.3 ms [−20.1, +160.7]) lean the
  same way, about 7%, and are level; the four decoders are level, Sol's
  point estimate on the better side. The encoders' single requests, one
  question through the same stacks, are level. The cause, and what is left
  of the gap, follow.
- **Kai, a second series.** A separate 10-round A/B on 16 cores with no other
  timed work on their NUMA node (node C 16–31, the first 30 router prompts,
  C = 1, the node's 1-minute load at most 44) reads it worse too: p50 +65.3 ms
  [+10.4, +120.2] (646.2 → 711.5 ms), one at a time −0.11 requests/s [−0.19,
  −0.03], C = 1 −0.11 [−0.20, −0.02], and p95 +33.5 ms [−7.0, +74.0], level.
  The bundled runtime's p50 stayed within 636–652 ms in every round; the
  runtime's was 651–677 ms in seven rounds (1–5% slower) and 807–834 ms in
  three (24–29% slower), so some fresh runtime processes run these requests in
  a slower mode.
- **The slow mode, and the huge-page default (`129be34ea`).** The runtime
  copies a model's weights into process memory at load (`69ae2d0c5`), on
  4 KiB pages whose physical placement differs from process to process; the
  bundled runtime reads its weights from the checkpoint's page-cache pages,
  the same physical pages in every process. In diagnostic runs on node D (16
  cores, the first 30 router prompts, C = 1), Kai's runtime processes ran
  650–680 ms or 820–870 ms at p50, slow in 4 of 7 and in 2 of 6; the bundled
  runtime never was. The OpenMP spin count, a second OpenMP team and the
  address layout (ASLR off) did not decide the mode. With
  `THP_MEM_ALLOC_ENABLE=1`, which puts PyTorch's CPU allocations of 2 MiB or
  more on madvised transparent huge pages, 6 of 6 processes ran 659–682 ms.
  Importing the package now sets it (design §12). Re-timed under it, on node
  C's NUMA node 0 with the bundled side's huge pages off as before, 10 rounds,
  load at most 49:

  | Row | With the default, runtime − bundled [95% CI] | Before it |
  | --- | --- | --- |
  | Kai router, p50 | +21.4 ms [+11.3, +31.6] (646.5 → 667.9), worse | +66.1 [+5.1, +127.1] four lanes; +65.3 [+10.4, +120.2] alone |
  | Kai router, p95 | +17.0 ms [−17.1, +51.2], level | +74.6 [+11.2, +138.1]; +33.5 [−7.0, +74.0] |
  | Kai router, one at a time | −0.04 requests/s [−0.06, −0.01], worse | −0.07 [−0.11, −0.02]; −0.11 [−0.19, −0.03] |
  | Kai router, C = 1 | −0.04 requests/s [−0.06, −0.01], worse | −0.06 [−0.13, −0.00]; −0.11 [−0.20, −0.02] |
  | Lex router, p50 | +18.4 ms [+10.9, +26.0] (700.8 → 719.2), worse | +56.7 [−55.6, +169.0], level |
  | Lex router, p95 | +52.7 ms [−2.7, +108.0], level | +47.6 [−70.4, +165.5], level |
  | Lex router, one at a time | −0.04 requests/s [−0.06, −0.01], worse | −0.05 [−0.14, +0.03], level |
  | Lex router, C = 1 | −0.03 requests/s [−0.05, −0.00], worse | −0.06 [−0.14, +0.02], level |
  | Route router, p50 | +13.4 ms [+6.2, +20.6] (646.6 → 660.0), worse | +70.3 [−20.1, +160.7], level |
  | Route router, p95 | +34.0 ms [−8.5, +76.4], level | +116.6 [−26.0, +259.2], level |
  | Route router, one at a time | −0.04 requests/s [−0.07, −0.01], worse | −0.07 [−0.17, +0.03], level |
  | Route router, C = 1 | −0.03 requests/s [−0.06, −0.01], worse | −0.08 [−0.16, +0.00], level |
  | Kai single, p50 | −0.05 ms [−1.21, +1.11], level | −0.21 [−2.78, +2.35], level |
  | Kai single, p95 | +2.3 ms [+0.1, +4.6] (83.9 → 86.2), worse | +3.75 [−5.15, +12.65], level |

  Kai's router row ran alone on 16–31; then Lex's router row (32–47) and Kai's
  single requests (48–63, C = 1 / 4 / 16 level or better) ran side by side. No
  runtime process fell into the slow mode. Lex's router row and Kai's
  single-request p95 read worse now because the slow mode's variance left
  their intervals, not because they got slower. On node D, Lex's runtime read
  +76.8 ms [−15.8, +169.4] without huge pages, 2 of 6 processes slow, and
  +10.3 [+5.5, +15.1] with them; Kai's single requests with huge pages on
  against off were p50 −0.07 ms [−0.67, +0.52] and p95 +0.42 [−2.49, +3.32].
  Outside Decision 1.0, 5 rounds on against off: Vela Embedding p50 −0.09 ms
  [−0.41, +0.23] and Vela Domain −0.24 [−0.69, +0.21], p95 level for both; a
  process's RSS grows by 1.5–23 MiB (0.1–1.3%). Re-timed under the default:
  Decision 1.0's Kai, Lex and Route on CPU (Route's router row alone on 16–31
  after the others, load at most 27), and the Vela Embedding and Vela Domain
  spot checks. Not re-timed under it: the decoders' CPU rows, Vela 2.0 (the
  0.8B's `exact` cells included) and Decision 2.0 on CPU, and the ROCm rows,
  where it changes only CPU-side allocations (the weights and forwards run on
  the GPU).
- **A longer OpenMP spin narrows what is left, but does not close it.** On
  node D with huge pages, Kai's router row read +11.7 ms [+2.4, +21.1] on the
  runtime's `GOMP_SPINCOUNT=10000` and +1.9 [−4.1, +7.9] on libgomp's default
  300,000. The bundled side of that diagnostic kept 10,000:
  `tools/decision1_bench.py` imported the runtime package, which then set
  it, on both sides of every series here. `vllm-srun serve` now picks the
  spin count per process (design §12): libgomp's default unless the process
  serves an ONNX Runtime model on the CPU. The A/B below runs both sides at
  each value, as each runs by itself.

  **Router requests at each spin count** (#4611, at `8b92b620f`): node C,
  16-core lanes of NUMA node 0 with memory bound to it (Kai 32–47, Lex
  48–63, Route 0–15). 10 interleaved rounds, fresh processes, the first 30
  public231 prompts, C = 1, and both spin counts in every round. Runtime −
  bundled:

  | Row | Both at 10,000 | Both at libgomp's default |
  | --- | --- | --- |
  | Kai, p50 | +25.9 ms [+13.8, +38.0] (789.6 → 815.5) | +16.4 [+9.6, +23.2] (780.5 → 796.9) |
  | Kai, p95 | +54.0 [−21.4, +129.4] | +50.5 [−17.3, +118.3] |
  | Kai, one at a time | −0.055 requests/s [−0.071, −0.039] | −0.048 [−0.068, −0.027] |
  | Kai, C = 1 | −0.032 [−0.047, −0.017] | −0.053 [−0.079, −0.027] |
  | Lex, p50 | +29.9 ms [+15.0, +44.7] (787.3 → 817.2) | +27.3 [+15.6, +38.9] (775.6 → 802.8) |
  | Lex, p95 | +58.6 [+14.9, +102.2] | +64.1 [+21.3, +106.8] |
  | Lex, one at a time | −0.052 [−0.068, −0.036] | −0.060 [−0.082, −0.039] |
  | Lex, C = 1 | −0.039 [−0.060, −0.018] | −0.053 [−0.075, −0.031] |
  | Route, p50 | +24.7 ms [+11.1, +38.2] (792.6 → 817.3) | +50.6 [+12.6, +88.5] (766.0 → 816.6) |
  | Route, p95 | +28.0 [−16.9, +73.0] | +112.6 [−6.4, +231.5] |
  | Route, one at a time | −0.054 [−0.077, −0.030] | −0.107 [−0.182, −0.032] |
  | Route, C = 1 | −0.031 [−0.048, −0.015] | −0.081 [−0.168, +0.006] |

  - **Every p50 and one-at-a-time row is worse at both spin counts.** The
    level cells are the three p95 rows at 10,000, Kai's and Route's at the
    default, and Route's C = 1 rate at the default.
  - **Both sides gain from the default spin.** Default − 10,000 at p50: the
    runtime −18.6 ms [−32.5, −4.7] (Kai), −14.3 [−29.7, +1.1] (Lex) and
    −0.7 [−11.7, +10.2] (Route); the bundled runtime −9.1 [−14.5, −3.7],
    −11.7 [−20.3, −3.1] and −26.7 [−60.0, +6.6].
  - **Against the bundled runtime as it runs** (libgomp's default), the
    runtime's p50 went from +35, +42 and +51 ms on 10,000 to +16, +27 and
    +51 ms for Kai, Lex and Route.
  - **So the rest of the gap is not the spin count:** 2–7% of a router
    request on these rows, open.
  - The ONNX Runtime side of the choice is in `embed-performance.md`.
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
