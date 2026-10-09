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
the published ones, and a borderline answer can flip. On NVIDIA GPUs from
Ampere on, `max_speed` runs Decision 2.0 with fused Triton kernels, 11% to 48%
faster with no decision changed in its record; Vela 2.0 keeps the exact
kernels there. Each approximate profile
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

- compatible work for the same model and routing stage can share a bundled
  call; one bundle may still require several forwards;
- task families can fuse compatible work, while decision-model shared-context
  execution is an explicit profile choice;
- independent workers can execute in parallel when hardware capacity allows;
  placing multiple workers on one GPU does not add compute capacity;
- cacheable repeated work can reuse recent results while they remain in the
  worker's bounded result cache (`--result-cache-entries`); replicas have
  independent worker caches;
- long inputs are split into windows only when you ask for windowing.
