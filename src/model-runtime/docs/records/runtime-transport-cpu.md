# Router-to-runtime transport on CPU: where a runtime call's time goes

The router's model-backed signals call the model runtime over HTTP/JSON on a
Unix socket. The standalone-mode latency record found 89% of a one-signal
request inside that call, without saying how much of it the transport took.
The runtime now reports its own time for every request in a `Server-Timing`
header, and the router records, for each call, the part of the exchange
outside the runtime (`vsr_model_runtime_transport_seconds`) and the runtime's
phases (`vsr_model_runtime_server_seconds`). This record splits the calls of
the default signals on CPU with those metrics and states the fast-path
decision they support.

- **Date:** 2026-10-07.
- **Machine:** AMD EPYC 9575F, CPU only, the machine of the router-latency
  record. A resident Kubernetes workload ran on other cores; the 1-minute
  load stayed between 3 and 25 (the program voids timings taken above 120).
- **Commits:**
  - This change: its code at `1219b9a56` on `main` `246dde1fe`, the same code
    as the merged commit.
  - Vela 2.0 0.3B: #4639's branch at `43bbb349b`, which makes the 0.3B the
    default of the built-in signals (`max_speed` on CPU), merged with this
    change for the measurement only (`c172bc376`). Its router, runtime and
    E2E code is that of #4702, which merged it. Both commits are on the
    branch `xunzhuo/runtime-server-timing-measured`.
  - Baseline: `main` at `3bff7d56d`.
  - Each router was built in CI's Go image (`golang:1.25-bookworm`) and each
    runtime installed from the same tree into a Python 3.12 environment with
    the router image's pins: PyTorch 2.10.0 (CPU) and ONNX Runtime 1.30.0.
- **Configurations**, from the router-latency record's
  [`router-latency-cpu.yaml`](router-latency-cpu.yaml):
  - *Vela 1.0:* that file as it is. Domain, prompt guard, PII, fact-check and
    feedback run on the Vela 1.0 models, each an implicit `model_runtime`
    deployment, plus a keyword signal. These were the defaults until #4702,
    and the file's catalog block restores them. The feedback detector reads
    follow-up turns only, so single prompts never call it.
  - *Vela 2.0 0.3B:* the same signals and decisions without the model
    catalog, so all of them run on their default, one 0.3B deployment: the
    defaults since #4702.
  - *Prompt guard alone:* the issue's case, the jailbreak signal on Vela 1.0
    Guard and nothing else.
- **Inputs:** the router-latency record's 539 prompts (median 59 characters,
  95th percentile 456, longest 4,920), written by
  [`tools/router_latency.py`](../../tools/router_latency.py) `corpus`
  (SHA-256 `6fb21cdbfbf4f0df…`).
- **Method:** the router-latency record's.
  - `POST /api/v1/routing/preview` runs every signal and the decision, with no
    upstream call. Each pass sends 20 warm-up requests, then the corpus once
    sequentially and once each at concurrency 4 and 16. Both result caches are
    off (the router's and the runtime's), so every request computes.
  - Each router and its runtime processes run in their own
    `systemd-run --scope -p AllowedCPUs=32-43` (12 vCPUs, memory bound to
    NUMA node 0); the effective cpuset of every scope read 32–43. The driver
    is pinned to 4 other vCPUs.
  - Five rounds, the arms alternating their order.
  - Each pass is bracketed by scrapes of the router's `/metrics`. For each
    deployment, the deltas give its calls, their mean duration in the router
    (`vsr_model_runtime_request_duration_seconds`), the mean transport of the
    exchanges that carried them and the runtime's mean phases. The tables
    give the median of the five rounds.
  - Raw per-round results: [`runtime-transport-cpu.json`](runtime-transport-cpu.json).

## Result

A call's time in the router splits into four parts:

- the router's own share: waiting for the other calls of its bundle, fusing
  them and splitting the answers;
- **transport**: the router's encoding and decoding, the connection and the
  HTTP exchange outside the runtime's handler;
- the runtime's **serving**: reading and decoding the body, encoding the
  answer, and its own overhead (event-loop and thread hand-offs);
- **inference**: tokenizing, waiting for the model, the forwards and
  assembling the answers.

Every one of the 78,165 runtime calls this change made in the rounds reported
its time.

**Vela 1.0, sequential** (requests: p50 16.3 ms, mean 26.2 ms, 38.2 per
second). Mean ms per call; each request makes one call per signal, and the
four run in parallel:

| Signal | Call | Router | Transport | Serving | Tokenize | Queue | Forward | Post | Transport share | Serving share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Domain | 21.48 | 0.15 | 0.47 | 0.33 | 0.24 | 0.18 | 20.06 | 0.05 | 2.2% | 1.6% |
| Fact-check | 21.62 | 0.13 | 0.46 | 0.32 | 0.24 | 0.19 | 20.20 | 0.05 | 2.1% | 1.5% |
| PII | 24.33 | 0.15 | 0.46 | 0.32 | 0.25 | 0.25 | 22.73 | 0.17 | 1.9% | 1.3% |
| Prompt guard | 23.82 | 0.11 | 0.46 | 0.33 | 0.24 | 0.21 | 22.39 | 0.09 | 1.9% | 1.4% |

**Vela 2.0 0.3B, the defaults since #4702, sequential** (requests: p50
79.1 ms, 11.9 per second): one call per request carries every signal's
question.

| Deployment | Call | Router | Transport | Serving | Tokenize | Queue | Forward | Post | Transport share | Serving share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vela 2.0 0.3B | 83.01 | 0.17 | 0.48 | 0.30 | 0.45 | 2.16 | 79.20 | 0.35 | 0.6% | 0.4% |

**Prompt guard alone, sequential** (requests: p50 8.8 ms, mean 13.0 ms, 76.5
per second):

| Signal | Call | Router | Transport | Serving | Tokenize | Queue | Forward | Post | Transport share | Serving share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prompt guard | 12.60 | 0.04 | 0.26 | 0.17 | 0.17 | 0.06 | 11.83 | 0.05 | 2.1% | 1.4% |

- **Inference is 95–99% of every call; transport is 0.6–2.2%.** The forward
  alone is 93–94% of a Vela 1.0 call and 95% of the 0.3B's. The standalone
  record's Guard request spent 12.4 ms in its runtime call; on this corpus a
  Guard call takes 12.6 ms, of which 0.26 ms is transport and 0.17 ms the
  runtime's serving.
- **Under load the parts outside inference grow, and their share falls.**
  At concurrency 4 and 16 the CPU is saturated: transport rises to 0.9–1.3 ms
  and serving to 0.7–1.4 ms per Vela 1.0 call, while the calls wait 21–227 ms
  for their model. Transport stays below 1.8% of every call:

  | Pass | Vela 1.0 transport share (per signal) | Vela 1.0 queue | 0.3B transport share | Prompt guard transport share |
  | --- | --- | --- | --- | --- |
  | Sequential | 1.9–2.2% | 0.18–0.25 ms | 0.6% | 2.1% |
  | Concurrency 4 | 1.4–1.7% | 21–33 ms | 0.3% | 1.3% |
  | Concurrency 16 | 0.5–1.0% | 68–227 ms | 0.1% | 0.3% |

- **The phases expose where a model's own time goes.** The 0.3B's 2.2 ms of
  queue on a sequential pass is its `max_speed` profile holding the 2 ms
  batching window (`--batch-window-ms`) for a request that has nothing to
  batch with; Vela 1.0 on `exact` waits 0.2 ms.

## The timing's own cost

- **In the request path:** the runtime reads the clock six more times per
  request, keeps one small object per request and one per model group, and
  writes one header: about 3 µs per request in isolation. The router reads the
  header and records eight histogram samples per call without allocating:
  about 1 µs (`BenchmarkServerTimingReadAndRecord`). Both were measured on one
  core of an AMD Ryzen AI 7 PRO 350.
- **End to end**, `main` against this change in the same five rounds on the
  Vela 1.0 configuration (difference with its 95% interval):

  | Pass | p50 ms | p95 ms | Requests/s |
  | --- | --- | --- | --- |
  | Sequential | 16.29 → 16.33, +0.18 [−0.29, +0.65] | 57.91 → 58.14, −0.55 [−2.23, +1.12] | 38.55 → 38.15, −0.35 [−0.94, +0.23] |
  | Concurrency 4 | 58.54 → 58.10, −0.43 [−3.22, +2.37] | 228.68 → 228.99, +3.72 [−13.92, +21.37] | 49.26 → 49.65, +0.36 [−0.33, +1.05] |
  | Concurrency 16 | 275.60 → 282.49, +4.29 [−1.53, +10.11] | 719.06 → 623.86, −68.59 [−141.53, +4.35] | 51.26 → 52.29, +0.55 [−1.02, +2.11] |

  Every interval of p50, p95 and throughput spans zero. The sequential p99,
  the sixth-slowest of 539 requests, came out +10.90 [+1.00, +20.81] ms;
  `main`'s own p99 ranged from 105.8 to 124.7 ms over the rounds. Those slowest
  requests are the longest prompts, so they were measured directly: the
  corpus's 27 longest prompts (456 characters and up), ten times each in a
  shuffled order, sequentially, in five more rounds alternating `main` and
  this change:

  | Long prompts, sequential | `main` → this change | Difference [95% CI] |
  | --- | --- | --- |
  | p50 | 75.04 → 74.80 ms | −0.10 [−0.32, +0.12] |
  | Mean | 135.17 → 134.26 ms | −0.94 [−2.20, +0.31] |
  | p99 | 631.26 → 632.10 ms | −8.59 [−25.89, +8.71] |
  | Requests/s | 7.39 → 7.44 | +0.05 [−0.02, +0.12] |

  The long prompts are not slower, so the corpus p99 above is the noise of a
  few requests per round. Their calls spend 0.4–0.6% in transport.

## Decision: no fast path

The router keeps calling the runtime over HTTP/JSON on its Unix socket.

- A fast path can only remove the transport and the runtime's serving: about
  0.8 ms of a 21–24 ms Vela 1.0 call (3–4%), 0.8 ms of the 0.3B's 83 ms (1%)
  and 0.4 ms of the 12.6 ms Guard call (3.4%). The signals of a request call
  in parallel, so a request would gain at most about 0.8 ms, under 5% of its
  p50 and about 3% of its mean.
- The rest is inference: the forwards are 93–95% of each call. Faster
  requests on CPU come from the models' side, such as the 0.3B's CPU cost and
  its batching window ([#4668](https://github.com/vllm-project/semantic-router/issues/4668)),
  and from GPU placement.
- A fast path would add what the standalone-mode design declined to take on
  (cgo, or a second wire protocol with its own framing and failure handling)
  for a few percent.

**When to revisit:** when, for a deployment on the request path, the
transport plus the runtime's `parse`, `serialize` and `other` phases exceed a
fifth of its `vsr_model_runtime_request_duration_seconds`. That is most likely
on a GPU, where forwards are a few milliseconds; this record covers the CPU.

**What a fast path would be then,** cheapest first:

1. **Cheaper HTTP on the same socket:** a binary bundle body (MessagePack,
   as vLLM's front end uses with its engine core) and a plain ASGI handler
   instead of the routing framework. It trims encoding and the runtime's
   serving, and changes neither process isolation nor failure handling.
2. **A shared-memory ring per runtime process:** request and response slots
   in a memory segment the router creates when it starts the process, an
   `eventfd` doorbell each way, and a fixed binary layout for a bundle's texts
   and questions. It removes the HTTP exchange and the runtime's event-loop
   hand-off, and a runtime that crashes still cannot take the router down.
3. **Not in process:** running PyTorch inside the router stays out, for the
   design doc's reasons (cgo, the Python GIL against the Go scheduler, GPU
   faults).
