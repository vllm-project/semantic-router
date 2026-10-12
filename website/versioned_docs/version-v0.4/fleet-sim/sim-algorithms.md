---
title: Simulation Model
---

# Simulation model

Fleet Sim combines a fast queueing approximation with a request-level
discrete-event simulator (DES). This page explains what those models calculate
and where their results need external validation.

## Model flow

For a typical `optimize` run, Fleet Sim:

1. splits the workload CDF into short and long sub-distributions;
2. estimates service-time distributions from a GPU profile;
3. sizes each pool with an M/G/c queueing approximation;
4. ranks candidates by modeled hourly cost; and
5. runs DES for up to `--verify-top` candidates.

`simulate` skips the search and runs DES for the pool sizes supplied by the
caller. `pareto` evaluates CDF breakpoints analytically. `disagg` uses a
separate phase-level analytical model and does not run the request-level DES.

## Workload model

### CDF sampling

A CDF point `[t, f]` means fraction `f` of requests have at most `t` total
tokens. Samples are drawn uniformly inside each CDF interval. By default, a
sampled total is divided into 80% input and 20% output tokens.

The synthetic workload also assigns one of three categories:

| Category | Default fraction | Used by |
| --- | ---: | --- |
| `prose` | 0.60 | Compress-and-route eligibility |
| `code` | 0.25 | Treated as unsafe to compress |
| `rag` | 0.15 | Compress-and-route eligibility |

Arrivals follow a Poisson process at rate `lambda`. The random seed makes a
given command reproducible, but a single seed is not a confidence interval.

The standalone HTTP service converts an uploaded trace to a total-token CDF
before running a job. Original inter-arrival times, input/output ratios, and
route ordering are not replayed by that path. The Python library also contains
`TraceWorkload` for programmatic timestamped replay.

### Planning implication

The length router uses `input_tokens + output_tokens`. A live router does not
know the final output length before generation, so this is a capacity-planning
oracle rather than a directly deployable routing rule. When evaluating a real
policy, use the token estimate available at request time and measure its error.

## GPU profiles

Every pool uses a profile that supplies iteration latency, prefill latency,
KV-cache capacity, maximum concurrency, and hourly cost.

### Manual profiles

A `ManualProfile` contains measured or estimated constants:

| Field | Meaning |
| --- | --- |
| `W` | Base iteration latency in seconds |
| `H` | Per-active-sequence latency at `calibration_ctx` |
| `calibration_ctx` | Sequence length at which `H` was calibrated |
| `chunk` | Prefill tokens processed per iteration |
| `blk_size` | Tokens in one KV-cache block |
| `total_kv_blks` | KV-cache block budget |
| `max_slots` | Calibrated concurrency limit |
| `cost_per_hr` | Modeled hourly cost for one profile unit |

At active concurrency `n` and mean sequence length `L`, the iteration model is:

```text
H_effective = H * L / calibration_ctx
iteration_time = W + H_effective * n
```

The number of slots is the smaller of the KV-cache limit and the scaled
`max_slots` limit for the pool's maximum context. Manual profiles use the same
iteration model for prefill and decode because they contain no hardware FLOP
specification.

### Computed profiles

A `ComputedProfile` is built from a hardware specification, model
architecture, and serving configuration such as tensor parallelism, dtype,
chunk size, and KV utilization. It derives `W`, `H`, and the KV block budget.

For prefill, it estimates projection, attention, and feed-forward FLOPs and
returns the slower of compute time and memory time. This roofline calculation
is useful for sensitivity analysis, but it omits many runtime effects and must
still be calibrated against a load test.

### What a profile represents

A profile describes the full **model + hardware + parallel layout + serving
configuration** combination. A GPU name alone is not enough. Changing model
size, quantization, tensor parallelism, maximum context, chunking, or vLLM
version can invalidate `W`, `H`, capacity, cost, and power values together.

## Service-time model

For a request with input length `L_in`, output length `L_out`, and prefill chunk
`C`, the model uses:

```text
prefill_iterations = ceil(L_in / C)
prefill_time = prefill_iterations * prefill_iteration_time
decode_time = L_out * decode_iteration_time
raw_service_time = prefill_time + decode_time
TTFT = queue_wait + prefill_time
```

The DES schedules an effective completion time derived at full slot
concurrency, while it also records a physical completion estimate for output
token timing. It does not simulate every scheduler iteration or token event.

## Analytical sizing

Fleet Sim samples 3,000 service times from each pool's normalized CDF and
calculates:

- mean service time;
- squared coefficient of variation (`CV^2`);
- KV-cache slots per profile unit; and
- mean prefill time.

It estimates profile throughput as:

```text
mu_gpu = slots / mean_service_time
```

Pool waiting time uses an Erlang-C probability with the Kimura M/G/c P99
approximation. Each KV slot is treated as a queueing server. The selected pool
size must meet both the P99 wait target and a default utilization cap of 0.85.
Reported analytical TTFT adds mean prefill time to the estimated P99 wait.

This approximation assumes a stationary Poisson arrival stream and an
independent service-time distribution. Token-based routing, bursty traffic,
and shared GPU scheduling can violate those assumptions, which is why selected
candidates should be checked with DES and a real load test.

## Discrete-event simulation

The DES advances between arrivals and modeled completions. Each pool contains
identical instances and uses shortest queue by default. An instance admits a
request only when it has both a free logical slot and enough KV blocks.

If KV admission would exceed the block budget, the instance preempts the
longest active request and puts it back at the head of the queue. The model
restarts its service calculation when admitted again; it does not preserve
token-by-token progress.

The primary DES metrics are P50/P99 TTFT, P99 queue wait, completion throughput,
SLO compliance, and mean utilization. Percentiles and SLO compliance are
calculated over **completed requests only**. Always compare `total_completed`
with the requested simulation count; queue rejection or an incomplete drain
can otherwise make the latency percentiles look healthier than the entire
arrival population.

The CLI does not currently remove an initial warm-up segment from reported
metrics. Run enough requests, inspect multiple seeds, and compare steady-state
load tests when tail latency matters.

## Routing models

| Router | Behavior | Availability |
| --- | --- | --- |
| `LengthRouter` | Sends a request to the smallest fitting pool or a configured short/long split | CLI and library |
| `CompressAndRouteRouter` | Compresses eligible requests in `(B_short, gamma * B_short]` and sends them to the short pool | `optimize`, `simulate` through `--gamma`, and library |
| `SpilloverRouter` | Sends short traffic to the long pool when short-pool pressure crosses a threshold | Library |
| `LeastLoadedRouter` | Selects the pool with the lowest active-plus-queued load relative to slots | Multi-pool CLI and library |
| `ModelRouter` | Maps `request.model_id` to a pool, with a configured or first-pool fallback | Multi-pool CLI and library |
| `SemanticRouter` | Calls a user-supplied classifier function, falling back when its output is unknown | Library; CLI JSON can select it but supplies no classifier function |
| `RandomRouter` | Uniformly selects a pool | CLI baseline and library |

`compare-routers` compares only length, compress-and-route with a fixed
`gamma=1.5`, and random routing. It does not call the live vLLM Semantic Router.

### Compress-and-route assumptions

The DES treats `prose`, `rag`, and `mixed` as safe categories and `code` as
unsafe. Eligible input is shortened to the short-pool budget. The analytical
optimizer uses an effective compression probability, defaulting to `0.75`, for
traffic in the borderline band.

These are simulator assumptions. They do not measure semantic preservation,
category error, or the real compressor's latency. Validate quality separately
and replace the safe fraction with workload evidence.

## Threshold search

`pareto` uses every non-terminal CDF breakpoint with a nontrivial short-traffic
fraction as a candidate `B_short`. For each point it analytically sizes the two
pools and marks points not dominated in both modeled cost and worst-pool P99.

The output is a set of trade-offs, not an automatic production threshold. A
threshold should also be robust to estimation error, workload drift, and the
context capacity of the short pool.

## Disaggregated prefill and decode

The `disagg` command estimates independent prefill and decode throughput, then
sweeps worker counts. System throughput is the smaller of the two phase rates.
It applies fixed degradation factors (`0.90` for prefill and `0.92` for decode)
and multiplies base prefill time by `1.80` for its TTFT estimate.

Those constants are built-in assumptions. The model does not simulate KV
transfer size, topology, network contention, placement, or phase queues. Use
measured transfer and phase behavior before deciding that a disaggregated
design meets TTFT or TPOT targets.

## Known boundaries

Fleet Sim currently does not model:

- token-level continuous batching or exact vLLM scheduler behavior;
- tensor-parallel collectives, network topology, host overhead, or kernel
  launch effects;
- prefix-cache hit distributions, speculative decoding, quantization kernels,
  or adapter switching unless represented indirectly in a calibrated profile;
- non-Poisson bursts in the CLI CDF workflows;
- model answer quality or routing-classifier error;
- live failures, repair queues, rollout capacity, or autoscaler reaction time;
- rejection as part of the SLO-compliance denominator; or
- hard rejection when a request exceeds every configured pool's maximum
  context—the length router sends it to the largest pool.

Use the simulator to compare explicit assumptions. Use production-shaped load
tests to accept a deployment.

For energy calculations built on top of this performance model, continue with
the [power model](./power-model).
