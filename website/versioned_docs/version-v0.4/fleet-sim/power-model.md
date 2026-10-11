---
title: Power Model
---

# Power model

Fleet Sim can add an estimated power curve to a calibrated performance
profile. This supports two planning questions:

- how output tokens per joule change with pool composition and utilization;
- how a concurrency cap may trade modeled board power for queueing latency.

Power results are estimates. The DES can check the latency side of a scenario,
but it does not measure or independently validate the power curve.

## Power-model variants

### Manual profile

A `ManualProfile` can define:

- `power_idle_w`: modeled board power at the low-concurrency endpoint;
- `power_nominal_w`: modeled board power at the high-concurrency endpoint;
- `power_logistic_k`: curve steepness; and
- `power_logistic_x0`: midpoint on a `log2(concurrency)` axis.

When `power_logistic_k` is greater than zero, power at concurrency `b` is:

```text
P_range = P_nominal - P_idle
P(b) = P_idle + P_range / (1 + exp(-k * (log2(max(1, b)) - x0)))
```

When `k` is zero, the model linearly interpolates between `P_idle` and
`P_nominal` using `b / max_slots`.

Both variants use active sequences as the load variable. They do not model
clock state, temperature, kernel mix, host power, networking, or cooling.

### Computed profile

A `ComputedProfile` estimates power from the hardware TDP and the profile's KV
traffic and tensor-core activity. It interpolates between fixed fractions of
TDP as modeled activity rises.

This is a coarse transfer model, not a substitute for a batch-versus-power
measurement on the target GPU. Applying the same TDP fractions to another
architecture, model, precision, or parallel layout adds uncertainty even when
the hardware specification itself is correct.

## Built-in profile boundary

The CLI includes manual profiles named `h100`, `a100`, and `a10g`. They make
the examples runnable, but their constants should be treated as editable
planning assumptions:

- their prices are static values embedded in source, not a live cloud-price
  feed;
- their performance and power curves are not measurements of your deployment;
- the H100/A100 profiles are intended to represent a 70B-class, multi-GPU
  layout, while A10G represents a smaller, single-GPU model; and
- a profile name does not encode every serving parameter that affects power.

Consequently, a default `h100` versus `a10g` tokens-per-watt result compares
two modeled systems, including model size. It does not establish that one GPU
is more efficient for the same model.

## Tokens per watt

For one homogeneous pool, Fleet Sim estimates output throughput from arrival
rate and mean output length, then divides by modeled pool power:

```text
pool_output_tokens_per_second = lambda_pool * mean_output_tokens
pool_power_watts = N_pool * P(mean_active_sequences)
pool_tokens_per_watt = pool_output_tokens_per_second / pool_power_watts
```

For a multi-pool fleet, the correct aggregate is:

```text
fleet_tokens_per_watt =
  sum(lambda_i * mean_output_tokens_i) /
  sum(N_i * P_i(mean_active_sequences_i))
```

Fleet size can cancel algebraically for one homogeneous pool at a fixed
operating point. It does not cancel across pools with different sizes, traffic
fractions, or power curves.

The CDF workflow estimates mean output length using its default 20% share of
total tokens. If your output ratio differs, tokens-per-watt can move even when
the power curve does not.

## Run an energy study

Single-pool mode applies the full workload to each selected profile:

```bash
vllm-sr-sim tok-per-watt \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --slo 500 \
  --gpus h100 a100 \
  --rho-sweep \
  --out energy.json
```

Use this as a hardware comparison only after creating profiles for the same
model, precision, parallel layout, context, and vLLM configuration.

Two-pool mode compares a routed topology with a homogeneous long-pool
baseline:

```bash
vllm-sr-sim tok-per-watt \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --slo 500 \
  --b-short 6144 \
  --gpu-short a10g \
  --gpu-long h100 \
  --out routed-energy.json
```

In this form, a model switch can be intentional. Report it as a combined
routing, model, and hardware comparison, and validate answer quality
separately.

## Concurrency-cap analysis

`grid-flex` evaluates a fixed fleet while lowering its modeled concurrency
limit:

```bash
vllm-sr-sim grid-flex \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --n-gpus 32 \
  --gpu h100 \
  --slo 500 \
  --flex-pcts 0 10 20 30 \
  --verify-des 20000 \
  --out flex.json
```

For each requested reduction, Fleet Sim:

1. computes a target per-profile power relative to `power_nominal_w`;
2. inverts the power curve to select a concurrency cap;
3. recalibrates the analytical queue at that cap;
4. estimates P99 TTFT; and
5. optionally runs DES with `--verify-des`.

The requested percentage can be clipped by the idle-power floor, so inspect
the reported watts rather than assuming the target was reached exactly.

This command does not change `max_num_seqs`, control a live vLLM server, or
communicate with a power system. It produces a modeled trade-off curve for an
external controller design.

## Calibrate a profile

Calibrate performance before power; tokens per watt is not meaningful if the
throughput model is wrong.

1. Fix the model, precision, tensor parallelism, vLLM version, context mix,
   clocks, and power limit.
2. Measure TTFT, token throughput, active sequences, and KV capacity at several
   steady-state loads. Fit `W`, `H`, `calibration_ctx`, and `max_slots`.
3. Record board power from platform telemetry at the same loads, including low
   concurrency and the highest sustainable operating point.
4. Fit the linear or logistic curve to those points. Do not use TDP as a
   measured nominal value.
5. Validate on held-out loads and a production-shaped token distribution.
6. Store the measurement date and environment with the profile; remeasure
   after material runtime or model changes.

A source-defined manual profile can be constructed explicitly:

```python
from fleet_sim.gpu_profiles.manual import ManualProfile

profile = ManualProfile(
    name="measured-model-on-target-gpu",
    W=0.006,
    H=0.0004,
    calibration_ctx=8192,
    chunk=512,
    blk_size=16,
    total_kv_blks=50000,
    max_slots=96,
    cost_per_hr=0.0,
    power_idle_w=180.0,
    power_nominal_w=360.0,
    power_logistic_k=0.9,
    power_logistic_x0=3.5,
)
```

The numbers above illustrate the required fields; they are not recommended
values for any device.

## Reporting checklist

An energy result should state:

- whether watts mean GPU board power or whole-system/facility power;
- model, dtype, parallel layout, runtime, clocks, and power limit;
- workload and output-length distribution;
- measurement source and fitted curve error;
- fleet counts and utilization for every pool;
- whether latency was analytical, DES, or load-test measured; and
- uncertainty introduced by any unmeasured profile.

Do not turn modeled board-power savings into facility energy, emissions, or
demand-response commitments without accounting for hosts, networking, storage,
cooling, power conversion, workload displacement, and real controller
behavior.
