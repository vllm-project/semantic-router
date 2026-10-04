# Decision 1.0 performance

The `decision1` family against each Decision 1.0 package's bundled runtime,
on the same node, inputs and devices.

- **On ROCm, exact and faster:** the exact profile answers byte-identically
  (parity record) and serves a single request 1.35–1.42× faster at p50 on the
  encoders and 1.6–2.8× faster on the decoders, end to end through the API.
- **The router's requests go further on the opt-in profiles.** For six router
  signals about one prompt, `batching` halves the encoders' p50 and p95 (Kai
  30.2 → 15.6 ms and 64.9 → 33.4 ms), with every layer stack replaying its own
  GPU graphs. `shared_context` cuts the decoders' p95 by 53–76% (Lux
  575 → 135 ms). 128 questions about one input run 2.9–3.7× faster than the
  bundled runtime. At 16 concurrent single requests, `batching` serves 2.9–7.6×
  the bundled runtime's throughput.
- **On CPU, exact matches the bundled runtime, and the opt-in profiles win:**
  both sides run the same FP32 math through the same MKL kernels. Each in its
  own process, exact serves 0.99–1.04× the bundled throughput with
  0.99–1.04× its p50. Six router signals run 3.3–4.1× faster on the encoders
  with `batching`. `max_speed` runs the encoders on `float32-packed` copies of
  their three stacks: single requests 1.3–1.5× faster at p50 and p95, the six
  router signals 4.1–5.5× faster, 1.4–2.0× the bundled throughput, and no
  decision changes in 1,431 requests per encoder.

- **Date:** 2026-10-04.
- **ROCm:** one AMD Instinct MI325X (gfx942) per run, in the packages' release
  image (PyTorch 2.12 ROCm, Transformers 5.17, Triton 3.7.1, FLA 0.5.2,
  causal-conv1d 1.7.0), on 8 host cores of the GPU's NUMA node.
- **CPU:** 16 cores of an AMD EPYC 9575F (Zen 5), CPU PyTorch 2.10 with
  MKL, Transformers 5.17, 16 threads on both sides.
- **Bundled side:** the package's runtime (Transformers remote code,
  `system_one`), a library that answers one request at a time, so its
  sequential rate is its throughput. On ROCm it runs the same FLA kernel
  choices as the native side.
- **Commits:** the paired runs at `0bc5c6a75` unless a table names another:
  `6537747d8` fixes the bench tool for requests that plan no items, and from
  `a3d2593a0` only the coalescing profiles hold the queue for the batching
  window, so `shared_context` no longer waits for it. The throughput and
  many-question runs are at `d5b985e43`, with the same `decision1` runtime code
  as `0bc5c6a75`. From `c22bb15cd` every encoder layer stack has its own GPU
  graphs and reduced copy: the encoders' ROCm `batching` rows and the CPU
  `max_speed` router rows are at that commit. The CPU unpaired rows are at
  `69ae2d0c5` and `7292622f3` (weights in process memory, oneDNN's primitive
  cache, and the CPU-only batch budget and type-head reads before them).
- **Raw results:** `decision1-performance.json`, every run without paths; from
  `c22bb15cd` with what the engine ran (graphs per layer stack, the copy).
- **Shared node:** other workstreams' jobs ran on other cores of the same
  node. The single-request and router latencies are therefore paired:
  `tools/decision1_bench.py paired` loads both runtimes in one process and
  times each request on every path back to back, in rotating order, all on
  the scheduler's device thread. Load changes then hit both sides alike.

## Workloads

- **Single request:** the first 400 typed-final prompts, each one request
  with one Choice question.
- **Router request:** the six router signals that Route's `QUESTIONS.json`
  declares (domain, modality, jailbreak, safety, pii, fact_check), as explicit
  questions about each public231 prompt: all 231 on ROCm, the first 100 on CPU.
- **Many questions:** the public request of `tools/many_questions.py` (one
  ticket) at 16, 64 and 128 questions; p50 of 20 runs.
- **Throughput:** the 400 single requests in waves of C concurrent requests
  through the scheduler.

Latencies are in ms. "Runtime" is end to end through `Runtime.call`, the
API's request path; "Scheduler" is planning plus the scheduler without the
API layer. The paired Δ is the median over requests of runtime minus
bundled.

## ROCm

### Single request, exact

| Model | Bundled p50 | Bundled p95 | Runtime p50 | Runtime p95 | Scheduler p50 | Paired Δ p50 | Speedup p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | 13.4 | 17.5 | 9.9 | 11.2 | 9.9 | -3.6 | 1.35× |
| Lex-0.6B | 12.0 | 19.5 | 8.5 | 13.7 | 8.9 | -3.3 | 1.42× |
| Route-0.6B | 12.0 | 13.5 | 8.8 | 10.1 | 9.1 | -3.1 | 1.37× |
| Eos-0.8B | 24.7 | 33.5 | 9.8 | 19.0 | 9.2 | -14.9 | 2.53× |
| Sol-2B | 24.6 | 31.0 | 8.9 | 14.3 | 8.4 | -15.2 | 2.76× |
| Nox-4B | 36.2 | 45.3 | 15.4 | 16.4 | 14.6 | -20.8 | 2.35× |
| Lux-9B | 33.4 | 39.6 | 20.7 | 21.6 | 19.8 | -12.7 | 1.61× |

### Router request

| Model | Profile | Bundled p50 | Bundled p95 | Runtime p50 | Runtime p95 | Paired Δ p50 | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | `exact` | 29.9 | 59.7 | 26.8 | 57.4 | -2.9 | `6537747d8` |
|  | `batching` | 30.2 | 64.9 | 15.6 | 33.4 | -14.6 | `c22bb15cd` |
| Lex-0.6B | `exact` | 29.7 | 58.5 | 26.6 | 55.7 | -2.8 | `6537747d8` |
|  | `batching` | 30.6 | 65.3 | 15.8 | 35.2 | -15.0 | `c22bb15cd` |
| Route-0.6B | `exact` | 29.7 | 58.3 | 26.6 | 61.1 | -2.7 | `6537747d8` |
|  | `batching` | 30.6 | 64.5 | 15.6 | 33.9 | -15.0 | `c22bb15cd` |
| Eos-0.8B | `exact` | 30.1 | 181.4 | 24.0 | 123.4 | -4.8 | `0bc5c6a75` |
|  | `shared_context` | 30.7 | 188.2 | 23.6 | 89.0 | -4.7 | `a3d2593a0` |
| Sol-2B | `exact` | 38.4 | 198.9 | 30.1 | 162.4 | -8.2 | `0bc5c6a75` |
|  | `shared_context` | 38.2 | 185.0 | 29.8 | 61.7 | -8.1 | `a3d2593a0` |
| Nox-4B | `exact` | 76.8 | 411.9 | 64.8 | 382.1 | -13.0 | `0bc5c6a75` |
|  | `shared_context` | 73.2 | 401.0 | 60.0 | 105.9 | -12.9 | `a3d2593a0` |
| Lux-9B | `exact` | 107.9 | 575.1 | 93.1 | 523.2 | -13.8 | `0bc5c6a75` |
|  | `shared_context` | 103.4 | 574.7 | 88.0 | 135.2 | -14.2 | `a3d2593a0` |

- Exact runs the released shapes, so on the encoders it does the bundled
  runtime's work: every question type present runs its stack over all six
  rows. `batching` runs each stack over its own rows only, packed, and from
  `c22bb15cd` replays a bucket graph per stack (at `6537747d8` only the Noul
  stack had graphs: p50 17.5–17.9 ms, p95 37.0–39.6 ms).
- `shared_context` computes the prompt once for all six questions; the
  decoders' long prompts are where the bundled runtime's p95 comes from.

### Many questions about one input (p50)

| Model | Questions | Bundled | Exact | Approximate | Approximate vs bundled |
| --- | --- | --- | --- | --- | --- |
| Kai-0.6B | 16 | 67.7 | 60.2 | 43.1 | 1.6× |
|  | 64 | 171.1 | 136.0 | 113.5 | 1.5× |
| Lex-0.6B | 16 | 66.6 | 59.4 | 43.5 | 1.5× |
|  | 64 | 170.4 | 134.9 | 109.9 | 1.6× |
| Route-0.6B | 16 | 66.3 | 59.7 | 54.2 | 1.2× |
|  | 64 | 172.8 | 133.9 | 109.3 | 1.6× |
| Eos-0.8B | 16 | 62.0 | 47.7 | 47.9 | 1.3× |
|  | 64 | 210.0 | 175.4 | 87.1 | 2.4× |
|  | 128 | 422.3 | 344.0 | 145.8 | 2.9× |
| Sol-2B | 16 | 76.6 | 60.4 | 45.7 | 1.7× |
|  | 64 | 298.8 | 230.9 | 91.7 | 3.3× |
|  | 128 | 593.9 | 456.7 | 167.6 | 3.5× |
| Nox-4B | 16 | 148.8 | 123.6 | 63.3 | 2.3× |
|  | 64 | 587.8 | 493.5 | 175.6 | 3.3× |
|  | 128 | 1,181 | 981.8 | 328.5 | 3.6× |
| Lux-9B | 16 | 212.0 | 184.8 | 86.2 | 2.5× |
|  | 64 | 844.0 | 727.1 | 249.3 | 3.4× |
|  | 128 | 1,687 | 1,438 | 453.4 | 3.7× |

- The encoders' approximate profile is `batching` (bidirectional attention
  shares no prefix); the decoders' is `shared_context`.

### Throughput (requests/s)

Single requests:

| Model | Bundled (sequential) | Exact C = 1 | Exact C = 4 | Exact C = 16 | Approximate C = 1 | Approximate C = 4 | Approximate C = 16 | Approximate profile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | 57.1 | 101.8 | 104.2 | 114.8 | 82.5 | 211.6 | 368.3 | `batching` |
| Lex-0.6B | 65.7 | 89.5 | 91.8 | 126.5 | 93.4 | 247.0 | 398.0 | `batching` |
| Route-0.6B | 80.5 | 122.8 | 125.3 | 142.4 | 94.1 | 238.5 | 365.8 | `batching` |
| Eos-0.8B | 36.8 | 104.3 | 90.8 | 84.0 | 89.5 | 198.8 | 279.9 | `batching` |
| Sol-2B | 35.8 | 106.2 | 108.3 | 90.4 | 92.4 | 174.0 | 225.1 | `batching` |
| Nox-4B | 25.8 | 62.9 | 65.3 | 66.4 | 60.2 | 100.1 | 130.7 | `batching` |
| Lux-9B | 30.4 | 47.1 | 48.6 | 50.2 | 41.3 | 66.5 | 88.5 | `batching` |

Router requests:

| Model | Bundled (sequential) | Exact C = 1 | Exact C = 4 | Exact C = 16 | Approximate C = 1 | Approximate C = 4 | Approximate C = 16 | Approximate profile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | 32.1 | 33.3 | 34.3 | 34.5 | 46.9 | 78.0 | 96.1 | `batching` |
| Lex-0.6B | 31.8 | 35.0 | 36.3 | 37.1 | 49.2 | 81.9 | 100.9 | `batching` |
| Route-0.6B | 32.9 | 34.1 | 36.2 | 37.1 | 50.2 | 86.8 | 104.8 | `batching` |
| Eos-0.8B | 16.1 | 22.6 | 22.9 | 23.4 | 26.4 | 28.1 | 29.4 | `shared_context` |
| Sol-2B | 14.3 | 17.9 | 17.8 | 18.1 | 19.6 | 24.7 | 25.1 | `shared_context` |
| Nox-4B | 7.0 | 8.4 | 8.5 | 8.5 | 13.7 | 14.4 | 14.5 | `shared_context` |
| Lux-9B | 4.9 | 5.6 | 5.6 | 5.7 | 10.8 | 11.2 | 11.2 | `shared_context` |

- Exact runs one request per forward, so concurrency adds little (Eos and Sol
  lose some to host contention at C = 16).
- `batching` coalesces the questions of concurrent requests, and the encoders
  pack them without padding.
- The bundled rates come from the same session's separate runs, so they can
  differ from the paired latencies above by the node's load at the time.

The encoders' `batching` at `c22bb15cd`, where every layer stack replays its
own graphs, against the bundled runtime's sequential rate in the same session
(router requests: the paired run's):

| Model | Workload | Bundled (sequential) | `batching` C = 1 | C = 4 | C = 16 |
| --- | --- | --- | --- | --- | --- |
| Kai-0.6B | single | 77.4 | 123.1 | 280.8 | 448.3 |
|  | router | 32.3 | 59.7 | 93.4 | 112.4 |
| Lex-0.6B | single | 77.7 | 126.6 | 263.1 | 408.7 |
|  | router | 31.6 | 60.5 | 90.5 | 111.8 |
| Route-0.6B | single | 75.9 | 127.6 | 247.3 | 404.2 |
|  | router | 32.0 | 61.4 | 89.2 | 113.1 |

- One `batching` request alone is now 1.6–1.7× the bundled rate (8.1 vs
  12.4 ms at p50 for Kai), and 16 concurrent ones 5.3–5.8×.
- Against the table above (`d5b985e43`, the Noul stack's graphs only, another
  session), the Choice and Score stacks' graphs help most where launches
  dominate: one request at a time, +36–49% for single requests and +22–27% for
  router requests; at C = 16, +3–22% and +8–17%.

## CPU

### Single request, exact

| Model | Bundled p50 | Bundled p95 | Runtime p50 | Runtime p95 | Scheduler p50 | Paired Δ p50 | Speedup p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | 111.8 | 152.8 | 108.7 | 132.1 | 108.3 | -2.6 | 1.03× |
| Lex-0.6B | 115.4 | 169.4 | 116.2 | 166.2 | 114.8 | +1.2 | 0.99× |
| Route-0.6B | 111.2 | 150.0 | 107.2 | 142.4 | 107.0 | -2.7 | 1.04× |
| Eos-0.8B | 792.4 | 934.0 | 788.7 | 968.5 | 788.8 | -1.7 | 1.00× |
| Sol-2B | 1,348 | 1,587 | 1,305 | 1,571 | 1,318 | -10.6 | 1.03× |
| Nox-4B | 3,068 | 3,361 | 2,932 | 3,193 | 2,941 | -123.0 | 1.05× |
| Lux-9B | 5,263 | 5,420 | 5,236 | 5,493 | 5,275 | -5.1 | 1.01× |

- Both sides run the same FP32 math through the same MKL kernels, so exact
  matches the bundled runtime: 0.99–1.05× at p50, and p95 from 14% lower
  (Kai) to 4% higher (Eos).
- Again at `a3d2593a0` (with `GOMP_SPINCOUNT=10000`, on the quieter cores
  32–47): Kai bundled 106.0 / 119.6, runtime 107.2 / 125.0, paired Δ +1.1 ms.
- These paired runs load the bundled model on the process's main thread,
  which gives the process a second OpenMP thread team. With two teams on the
  cores, every parallel region waits for a wake-up (the case `accel/cpu.py`
  describes), so both sides run about 1.6× slower than alone. The deltas stay
  fair; the absolute latencies are high. The encoders' unpaired rows below
  run each side in its own process.

### Router request

| Model | Profile | Bundled p50 | Bundled p95 | Runtime p50 | Runtime p95 | Paired Δ p50 | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | `exact` | 1,054 | 1,351 | 986.7 | 1,261 | -51.1 | `0bc5c6a75` |
|  | `batching` | 1,339 | 1,654 | 325.4 | 458.9 | -1,008.6 | `0bc5c6a75` |
|  | `max_speed` | 1,070 | 1,235 | 193.1 | 247.7 | -868.2 | `c22bb15cd` |
| Lex-0.6B | `exact` | 1,064 | 1,397 | 1,061 | 1,293 | -20.6 | `0bc5c6a75` |
|  | `batching` | 1,102 | 1,390 | 279.1 | 384.6 | -812.3 | `0bc5c6a75` |
|  | `max_speed` | 788.9 | 1,009 | 188.6 | 246.7 | -597.5 | `c22bb15cd` |
| Route-0.6B | `exact` | 1,022 | 1,467 | 1,003 | 1,328 | -39.7 | `0bc5c6a75` |
|  | `batching` | 1,147 | 1,397 | 350.0 | 437.5 | -801.3 | `0bc5c6a75` |
|  | `max_speed` | 780.8 | 941.1 | 189.3 | 233.3 | -591.9 | `c22bb15cd` |
| Eos-0.8B | `exact` | 6,694 | 7,113 | 6,668 | 7,072 | -9.1 | `c22bb15cd` |
|  | `shared_context` | 6,255 | 6,666 | 6,315 | 6,785 | +27.1 | `0bc5c6a75` |

- On the encoders, exact does the bundled runtime's work and is 2–5% faster.
  `batching` runs each question type's stack over its own rows only, packed,
  and serves the six signals 3.3–4.1× faster. `max_speed` does the same on the
  `float32-packed` copies of the three stacks: 4.1–5.5× faster, with no
  decision changes (parity record). Its bundled rows come from a later session
  on other cores, hence their own bundled numbers.
- Eos (the first 30 prompts): exact, the default, matches the bundled runtime
  (paired Δ −9 ms on 6.7 s), and `shared_context` is within 1% of it. These
  prompts are short next to the six router questions, so little is shared; on
  ROCm, the long prompts are where it pays (p95 above).

### Unpaired single requests and throughput (encoders)

Each side ran in its own fresh process, on the same 16 cores per model (Kai
32–47, Lex 48–63, Route 0–15), each model's runs right after the bundled
runtime's sequential pass on those cores. Two kinds of prompt lengths:
uniform (the 400 typed-final requests, about 250 tokens each) and mixed (a
seeded sample of 400 css15 prompts that fit the 1,024-token input: 11–770
tokens, median 43). The bundled, exact and `batching` rows are two rounds at
`69ae2d0c5`; `max_speed` is three rounds at `7292622f3`. Means.

Throughput, requests/s, C = 1 / 4 / 16:

| Model | Lengths | Bundled (sequential) | Exact | `max_speed` | `batching` |
| --- | --- | --- | --- | --- | --- |
| Kai-0.6B | uniform | 11.3 | 11.3 / 11.3 / 11.5 | 17.3 / 20.3 / 21.8 | 11.1 / 13.5 / 14.3 |
|  | mixed | 13.5 | 13.8 / 13.9 / 14.0 | 20.1 / 22.6 / 24.8 | 13.5 / 15.9 / 17.5 |
| Lex-0.6B | uniform | 11.2 | 11.3 / 11.3 / 11.5 | 17.4 / 21.1 / 22.6 | 10.8 / 13.4 / 14.3 |
|  | mixed | 14.0 | 13.8 / 13.9 / 14.0 | 21.3 / 24.6 / 27.7 | 13.4 / 16.4 / 18.1 |
| Route-0.6B | uniform | 11.2 | 11.3 / 11.5 / 11.3 | 17.0 / 20.0 / 22.2 | 10.8 / 13.5 / 14.5 |
|  | mixed | 13.8 | 13.8 / 14.0 / 13.8 | 19.3 / 22.1 / 27.0 | 13.3 / 17.0 / 19.0 |

Single requests, ms:

| Model | Lengths | Bundled p50 / p95 | Exact p50 / p95 | `max_speed` p50 / p95 |
| --- | --- | --- | --- | --- |
| Kai-0.6B | uniform | 85.3 / 97.1 | 85.3 / 98.1 | 56.7 / 62.7 |
|  | mixed | 50.4 / 197.0 | 48.6 / 191.0 | 35.7 / 138.1 |
| Lex-0.6B | uniform | 85.9 / 95.4 | 84.7 / 96.3 | 57.8 / 64.5 |
|  | mixed | 49.2 / 191.7 | 49.5 / 190.4 | 37.3 / 140.0 |
| Route-0.6B | uniform | 86.2 / 98.7 | 85.7 / 94.4 | 57.8 / 65.7 |
|  | mixed | 49.7 / 195.5 | 49.9 / 194.1 | 36.7 / 143.3 |

- **Exact matches the bundled runtime:** 0.99–1.04× its rate at every
  concurrency and 0.99–1.04× its p50, with p95 from 4% lower to 1% higher.
  It does the same FP32 work, one request per forward (CPU rows are not
  batch-invariant; see the parity record).
- **`max_speed`** serves 1.4–2.0× the bundled rate and answers single
  requests 1.3–1.5× faster at p50 and p95.
- **`batching`** (FP32 weights) serves 1.0–1.4×: on CPU, packing saves the
  padding of coalesced requests; the packed linears of `max_speed` are what
  pays.
- **Four CPU-only changes since `c22bb15cd`,** each measured in alternating
  rounds:
  - **Weights in process memory** (`69ae2d0c5`; Where a request's time goes):
    exact went from 0.84–0.92× of the bundled rate on short prompts (one round
    at `d3d1d7e68`) to the table above.
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
python3 tools/decision1_bench.py native --model vllm-sr/Decision-1.0-Kai-0.6B --cache-dir HF_CACHE \
  --device rocm:0|cpu --profile exact|batching|shared_context|max_speed --prompts typed-final:PROMPTS.jsonl:400 \
  --concurrency 1 4 16 --output native.json
python3 tools/decision1_bench.py reference --package PACKAGE_DIR --repo vllm-sr/Decision-1.0-Kai-0.6B \
  --device cuda:0|cpu --prompts many64:MANY64.jsonl:20 --warmup 4 --output reference.json
```
