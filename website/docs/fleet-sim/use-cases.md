---
title: Capacity Planning
---

# Capacity-planning workflows

Fleet Sim is most useful as a sequence of questions. Start with the workload,
calibrate one baseline, and introduce new pools or policies only when the
baseline shows why they are needed.

The commands below demonstrate the workflow. They intentionally do not include
example savings or recommended GPU counts: those values depend on the workload
and profile assumptions you provide.

## 1. Describe the workload

Before sizing a fleet, collect:

- prompt and output token counts;
- request timestamps or arrival-rate ranges;
- the latency objective and how it is measured;
- the selected model or route, if traffic already uses semantic routing; and
- bursts, daily cycles, and failure periods that an average rate would hide.

The CLI accepts a cumulative distribution of **total tokens**:

```json
{
  "cdf": [
    [512, 0.25],
    [2048, 0.70],
    [8192, 0.95],
    [32768, 1.0]
  ]
}
```

Thresholds must increase and cumulative fractions should end at `1.0`. CLI
workloads sampled from a CDF assume 80% input tokens and 20% output tokens and
Poisson arrivals. If that split or arrival process does not resemble your
traffic, use the result only as a sensitivity study or use the Python library's
`TraceWorkload` with the `Fleet` API.

The standalone HTTP service accepts JSONL and CSV trace uploads and summarizes
prompt, output, arrival, and routing distributions. For simulation jobs, it
converts the uploaded lengths to a CDF and generates Poisson arrivals; it does
not replay original timestamps or route labels. Remove prompt text and user
identifiers before upload because only numeric planning fields are needed.

## 2. Establish a fixed-fleet baseline

Use `simulate` when you already know the current short- and long-pool counts:

```bash
vllm-sr-sim simulate \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --slo 500 \
  --b-short 6144 \
  --n-s 24 \
  --n-l 8 \
  --n-req 30000 \
  --out baseline.json
```

Look at each pool separately. A fleet-wide percentile can hide a small long
pool with a deep queue. Check at least:

- P99 TTFT and P99 queue wait by pool;
- completion and SLO-compliance fractions;
- mean utilization;
- preemptions or requests that did not complete; and
- sensitivity to the random seed and request count.

The first run is not a calibration. Adjust the profile so modeled TTFT,
throughput, concurrency, and KV capacity match a controlled load test of the
same model and serving configuration.

## 3. Find a short/long threshold

A two-pool design helps only when the pools have meaningfully different
service characteristics and the workload has enough traffic on both sides of
the split.

Use `pareto` to evaluate CDF breakpoints as candidate thresholds:

```bash
vllm-sr-sim pareto \
  --cdf data/lmsys_cdf.json \
  --lam 200 \
  --slo 500 \
  --gpu-short a100 \
  --gpu-long h100 \
  --out threshold-sweep.json
```

Choose a threshold for operational reasons, not just the lowest modeled cost:

- it should leave useful headroom in both pools;
- the short pool must be able to serve every request routed to it;
- route classification and token estimation must be stable near the boundary;
- answer quality must remain acceptable if the pools serve different models;
  and
- small workload shifts should not cause a large fleet-count jump.

After choosing a candidate, use `simulate` with the resulting counts instead
of relying only on the analytical row.

## 4. Search a two-pool fleet

`optimize` sizes pools analytically and can DES-check the lowest-cost
candidates:

```bash
vllm-sr-sim optimize \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --slo 500 \
  --b-short 6144 \
  --gpu-short a100 \
  --gpu-long h100 \
  --verify-top 3 \
  --n-sim-req 30000 \
  --out candidates.json
```

The search also sweeps a compress-and-route `gamma` band. That model assumes a
fraction of borderline traffic can be safely shortened. The CLI does not know
your real category mix or compression quality. Treat any benefit from
`gamma > 1` as conditional until you measure compression eligibility, latency,
and task quality on representative requests.

## 5. Plan for traffic growth

Use `whatif` to find rates where a pool needs another capacity unit or loses
its latency margin:

```bash
vllm-sr-sim whatif \
  --cdf data/azure_cdf.json \
  --lam-range 100 150 200 300 400 \
  --slo 500 \
  --b-short 6144 \
  --gpu-short a100 \
  --gpu-long h100 \
  --out arrival-sweep.json
```

Use several workload shapes, not only several rates. A change in long-context
share can overload the long pool even when total requests per second stays
constant. Add your own operational reserve for failures, deployments, and
burst absorption; CLI optimization does not infer that reserve from fleet
telemetry.

## 6. Compare routing policies

For a fixed two-pool fleet, `compare-routers` runs three CLI policies over the
same generated arrivals: length routing, compress-and-route with `gamma=1.5`,
and uniform random routing.

```bash
vllm-sr-sim compare-routers \
  --cdf data/agent_heavy_cdf.json \
  --lam 200 \
  --slo 500 \
  --b-short 6144 \
  --n-s 24 \
  --n-l 8 \
  --n-req 30000
```

This command is a controlled simulator comparison, not a benchmark of all
router implementations in the repository. In particular, it does not run a
live semantic classifier or include its latency and errors.

For routed production data, derive one CDF and arrival fraction per selected
model or pool, then use a `model` topology with per-pool `workloads`. This
preserves the observed aggregate routing mix without pretending to re-run the
classifier or reproduce request order.

## 7. Model more than two pools

Use `simulate-fleet` for model-specific pools or arbitrary heterogeneous
topologies. A minimal JSON file looks like this:

```json
{
  "pools": [
    {
      "id": "general",
      "gpu": "a100",
      "n_gpus": 12,
      "max_ctx": 8192
    },
    {
      "id": "long-context",
      "gpu": "h100",
      "n_gpus": 8,
      "max_ctx": 65536
    }
  ],
  "router": "length"
}
```

```bash
vllm-sr-sim simulate-fleet fleet.json \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --slo 500 \
  --n-req 30000 \
  --out fleet-result.json
```

Supported CLI JSON router values are `length`, `model`, `semantic`, `random`,
and `least_loaded`. For a model-routed CLI study, omit `--cdf` and provide a
`workloads` entry for each pool; passing `--cdf` overrides those per-pool
streams. Verify the fallback pool so missing model names do not silently
distort a programmatic study.

## 8. Evaluate disaggregated prefill and decode

Disaggregation is relevant when prefill and decode need different capacity or
hardware. It also introduces KV transfer, networking, and coordination costs.

```bash
vllm-sr-sim disagg \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --slo-ttft 500 \
  --slo-tpot 100 \
  --gpu-prefill h100 \
  --gpu-decode a100 \
  --mean-isl 2048 \
  --mean-osl 256 \
  --out disagg.json
```

The optimizer uses built-in degradation and transfer correction factors. Those
are assumptions, not measurements of your network or disaggregated runtime.
Replace the conclusion with a deployment test before selecting a prefill/decode
ratio.

## 9. Add power only after performance calibration

`tok-per-watt` and `grid-flex` build on the same performance profile plus a
power curve. They are useful for comparing scenarios after both have been
measured for the target model.

```bash
vllm-sr-sim tok-per-watt \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --slo 500 \
  --gpus h100 a100
```

```bash
vllm-sr-sim grid-flex \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --n-gpus 32 \
  --gpu h100 \
  --slo 500 \
  --verify-des 20000 \
  --out flex-curve.json
```

Do not use the built-in A10G versus A100/H100 `tok-per-watt` output as a
hardware-only comparison: the bundled profiles represent different model
sizes and parallel layouts. Even for profiles labeled with the same model,
calibrate wall power, throughput, and output length under the same test
conditions.

`grid-flex` estimates what happens when a concurrency cap reduces modeled
power. It does not apply a cap to vLLM or participate in a demand-response
system.

## Before using a result

Record the following with every decision:

- workload source and observation window;
- model, precision, tensor parallelism, GPU SKU, and vLLM settings;
- profile constants and how they were measured;
- simulator version, command, seed, and request count;
- analytical and DES results, including any disagreement; and
- the load-test result that accepted or rejected the design.

This turns a simulator run into a reviewable capacity assumption instead of an
unsupported performance claim.
