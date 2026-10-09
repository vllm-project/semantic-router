# Opt-in profiles: accuracy and speed on ROCm (MI325X)

> **Phase 1 record** (Decision 2.0, [#4481](https://github.com/vllm-project/semantic-router/pull/4481) and its follow-ups). The Phases 2–4 records are the `decision1-*`, `vela1-*`, `vela2-*`, `embed-*`, `stores-*`, `router-latency-cpu*`, `router-latency-rocm*`, `rocm-router-image` and `removal-footprint*` files.

The `exact` profile, the default, answers bit-identically to the released
packages ([parity record](rocm-mi325x-parity.md)). The other profiles may move
an answer by rounding and are opt-in.

Each profile reports three things:

- the decisions it changes, against the exact path's own noise floor;
- the Decision Index delta: a paired bootstrap over scoring cases, whose 95%
  CI must contain 0 or be positive (Index values stay private);
- the speed it buys.

- **Date:** 2026-10-03.
- **Device:** one MI325X (gfx942), the packages' release images, fast path on.
- **Speed figures:** from the [perf record](rocm-mi325x-perf.md).

## Noise floor

The exact path is deterministic within a process, but it is not
batch-invariant: GEMM kernels depend on the row count, so the same question
asked alone or inside its request's batch can move by rounding. That
batch-composition effect is the reference.

Measured with the released runtime, it changes about as many decisions as
the profiles below. On typed-final with 16 questions per request, it changed
18 / 2,000 decisions (Kai) and 10 / 2,000 (Eos). On css15 it changed
35 / 6,547 and 29 / 6,547 (shared-context study,
`src/training/decision2/v2/serving/records/shared-ctx-2026-10-02.md`).

Across processes, the Qwen3.5 sizes' gated-delta kernels (FLA) pick their
Triton block configurations by timing. Two processes can therefore differ
by rounding (|Δp| up to about 0.02) unless they share an autotune cache. The
parity records share one; the Index comparisons below are paired within one
process.

## `shared_context`

Many questions about one input: the shared prefix runs once as one packed
tree. Attention merges the prefix part and each question's own part by
log-sum-exp, and the gated-delta layers hand over the prefix state.

- **Answers:** identical to the released runtime's `share_context=True` for
  all six sizes (207 / 207 requests, parity record). Its accuracy is therefore
  that switch's study (`shared-ctx-2026-10-02.md`):
  - the decisions that change are near-ties (top-two margin < 0.015 but one),
    about as many as the noise floor;
  - accuracy deltas on the four panels have 95% CIs containing 0;
  - the Index delta is indistinguishable from 0 at Kai and Eos (CIs contain 0).
- **Policy:** `available()` sets a break-even threshold per backbone. Below
  it, the profile keeps the exact path: the request must save that many tokens,
  (questions − 1) × shared prefix ≥ threshold.
- **Speed** (128 questions about one ~240-token input):

| Model | Exact | `shared_context` | Speed-up |
| --- | --- | --- | --- |
| Kai-0.6B | 199 ms | 112 ms | 1.8× |
| Eos-0.8B | 232 ms | 137 ms | 1.7× |
| Sol-2B | 373 ms | 171 ms | 2.2× |
| Nox-4B | 826 ms | 315 ms | 2.6× |
| Lux-9B | 1,320 ms | 448 ms | 2.9× |
| Vega-27B | 5,286 ms | 1,562 ms | 3.4× |

## `batching`

Questions of concurrent requests share padded batches: rows sorted by length,
packed up to 65,536 padded tokens, after a 2 ms collection window.

- **Decisions changed** at 16 concurrent requests, the four panels in panel
  order (`batchfid`, the released runtime's batches):

| Model | typed-final | css15 | public231 | mlx-diag | Accuracy Δ, 95% CI |
| --- | --- | --- | --- | --- | --- |
| Kai-0.6B | 16 / 2,000 | 38 / 6,547 | 0 / 231 | 4 / 2,275 | typed-final −0.35 pt [−0.65, −0.05]; the others contain 0 |
| Eos-0.8B | 9 / 2,000 | 29 / 6,547 | 1 / 231 | 1 / 2,275 | all contain 0 (typed-final +0.20 [−0.05, +0.49]) |
| Sol-2B | 15 / 2,000 | 24 / 6,547 | 0 / 231 | 8 / 2,275 | all contain 0 |

  These counts match the noise floor (Kai 18 and 35, Eos 10 and 29 on
  typed-final and css15). Kai's typed-final CI just excludes 0, but its Index
  delta shows no loss (below).

- **Index**, all 120,226 requests, answered in groups of 16 consecutive
  requests:
  - Kai: the released runtime's batching against its exact path. The harness's
    exact sample reproduced the base run on 900 / 900 requests. The paired
    bootstrap's 95% CI contains 0.
  - Eos: the native `batching` profile against the native exact path in the same
    process.
    - 999 of 282,335 decisions changed (0.35%), and no request changed status.
      The exact path's own batch-composition noise on the multi-question Index
      sample is 0.30% (Eos) and 0.41% (Kai).
    - The paired bootstrap's 95% CI contains 0, but the point estimate leans
      negative (P(Δ ≤ 0) ≈ 0.9). Batching touches every request, including the
      single-question ones the shared-context switch leaves alone, so its CI is
      about twice as wide.
    - Recommendation: opt-in for throughput, not a default. Kai shows no lean.
- **Throughput** (requests/s; pre-rendered single-question requests in waves
  of C, steady state):

| Model | Exact | `batching` C = 16 | `batching` C = 64 |
| --- | --- | --- | --- |
| Kai-0.6B | 234 | 681 | 837 |
| Eos-0.8B | 197 | 606 | 721 |
| Sol-2B | 151 | 365 | 415 |
| Nox-4B | 84 | 165 | 177 |
| Lux-9B | 58 | 101 | 108 |
| Vega-27B | 15 | 26 | 28 |

- A lone request pays the 2 ms window (Kai 154 vs 230 requests/s at C = 1).
- Above 4,096 padded tokens batches run without graphs. They are GPU-bound
  there, and concurrent traffic would otherwise capture a graph for nearly
  every new large shape.

## `max_speed`

Shared-context trees for the multi-question requests that pass the
shared_context policy; cross-request batching for the rest; approximate
kernels allowed (none registered yet, so the kernels are the exact ones).

- **Decisions:** the union of the two profiles above.
- **Index** (Eos, native, same process as the exact path):
  - 1,007 of 282,335 decisions changed (0.36%), and no request changed status;
  - the paired bootstrap's 95% CI contains 0 and leans negative, as for
    `batching`: most Index requests are below the shared-context break-even,
    so they are batched.
- **Speed:** `shared_context`'s on multi-question requests and `batching`'s
  throughput on the rest.

## Candidates measured and not shipped

- **Merged projection GEMMs** (q/k/v, gate/up, the gated-delta input
  projections as one GEMM each): 1.4–2.3× faster GEMMs at ≤ 512 tokens, but not
  bit-identical. They wait for a `max_speed` kernel registration with their own
  accuracy record.
- **Efficient SDPA with in-kernel GQA:** unavailable on ROCm with a mask.
