---
title: Profiles
description: Choose between the published answers and faster, approximate settings.
---

# Profiles

A profile says how a model runs: exactly as published, or faster with
slightly different numbers. You choose it per deployment.

| Profile | In plain words | When to use it |
| --- | --- | --- |
| `exact` (default) | Gives the same answers the model's publishers measured, bit for bit for Decision 2.0. Task models run in full 32-bit precision on every device. | Always, unless you measured that you need more speed. |
| `shared_context` | When you ask a decision model several questions about one request, it reads the request once and answers every question from that reading. | Many decision questions per request on a GPU. |
| `batching` | Requests that arrive within a few milliseconds of each other are run together. | High traffic on a GPU, where combining requests raises throughput. |
| `max_speed` | Uses faster kernels and 16-bit numbers where the hardware supports them, on top of the above. | GPU deployments where latency matters more than the last decimal place. |

Everything except `exact` is **approximate**: answers can differ slightly from
the published ones, and a borderline answer can flip. Each approximate profile
has a measured accuracy record in the runtime's
[records](https://github.com/vllm-project/semantic-router/tree/main/src/model-runtime/docs/records).

## Set a profile

For a router-managed model, set `profile` on the deployment:

```yaml
global:
  model_catalog:
    deployments:
      decision-lux:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Lux-9B
        device: rocm:0
        profile: shared_context
```

For a runtime you start yourself, pass `--runtime-profile` (`--profile` to
`vllm-srun serve`):

```bash
vllm-sr serve vllm-sr/Decision-2.0-Lux-9B --engine --platform rocm --device-ids 0 --runtime-profile shared_context
```

A runtime started with an approximate profile still answers requests that ask
for `exact` (`"options": {"profile": "exact"}`), so you can compare the two on
your own traffic before you switch.

## Performance without changing answers

The runtime is fast in the default profile too. It never trades correctness
for these:

- the signals of a request reach each runtime process in one bundled call;
- features that ask one model about the same text share one pass through the
  model;
- work for different models runs in parallel, on different processes and
  devices; the models that share one GPU take turns on it;
- a repeated input is not computed again: each model keeps a cache of recent
  results, keyed by the exact input and profile (`--result-cache-entries`);
- long inputs are split into windows only when you ask for windowing.
