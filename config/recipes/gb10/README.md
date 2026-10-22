# GB10 Kai Recipe Model Card

## Overview

This recipe places Decision 2.0 Kai 0.6B on CUDA device 0 and routes each
request from that model's domain, reasoning, and difficulty answers. Code
that Kai treats as multi-step and at least moderately difficult goes to the
external generation backend. Every other answered request goes to the local
generation backend. The two generation backends are separate aliases with
one endpoint each.

Clients use `vllm-sr/auto`. The recipe does not install a generation model.

## Model details

| Item | Value |
| --- | --- |
| Decision deployment | `decision-kai` |
| Artifact | `vllm-sr/Decision-2.0-Kai-0.6B` |
| Revision | `cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764` |
| Device | `cuda:0` |
| Catalog memory floor | 4 GiB (`min_device_memory_gib`) |
| Local alias | `local/generation` at `local-generation:8000` |
| External alias | `external/generation` at `external-generation:8000` |
| Entrypoint | `vllm-sr/auto` |

The 4 GiB value is the catalog placement input for this model. It is not a
measured memory footprint. Replace the two generation endpoints with the
backends this deployment actually serves. The qualification profile starts
controlled stubs and rewrites only those endpoints for its run.

## Intended use

Use this recipe to exercise one explicit CUDA placement of Kai together with
one local generation backend and one external generation backend. The
operator profile under `tools/qualification/gb10` is the hardware check for
one documented GB10 host.

## Routing behavior

Kai answers three questions in one call. The questions use the same shapes
as Kai's readiness goldens: domain (`code`, `math`, `other`), multi-step
reasoning, and difficulty (`Trivial`, `Moderate`, `Hard`).

| Priority | Decision | When it matches | Backend |
| --- | --- | --- | --- |
| 200 | `external_code` | Domain is `code`, reasoning is at least 0.5, and difficulty is at least 1.0 | `external/generation` |
| 100 | `local_generation` | Domain is `math` or `other`, or the code request misses either threshold | `local/generation` |

Both decisions set `on_unknown: fail_request`. An unanswered Kai call returns
`decision_unresolved` and does not move the request to the other backend.
There is no routing fallback list.

## Requirements

- A Router CUDA image and an NVIDIA GPU visible as `cuda:0`.
- Reachable OpenAI-compatible endpoints for `local/generation` and `external/generation`.
- The pinned Kai revision. The deployment sets `device: cuda:0`, so a host without that GPU does not place Kai on CPU.

A passing GB10 receipt uses a digest-pinned image:

```bash
VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr-cuda@sha256:<digest> \
  python3 tools/qualification/gb10/qualification.py
```

A local `latest` tag can start the same `vllm-sr serve --platform cuda` flow
during development. That tag is not a qualification image.

## Data handling and safety

Kai reads each request inside the Router's model runtime. The selected
generation backend receives the chat request. Response caching and Router
Replay are off. Provider logging and retention still apply to whichever
generation endpoint is configured.

Preview returns the evaluated text and diagnostic values. Treat those
responses as application data.

## Quick start

Start the two generation backends, then validate and serve this file:

```bash
vllm-sr config validate --config config/recipes/gb10/config.yaml
vllm-sr serve --platform cuda --config config/recipes/gb10/config.yaml
```

Send requests to `vllm-sr/auto` on the listener in the config. The response
headers `x-vsr-selected-decision` and `x-vsr-selected-model` name the route.

## Evaluation

`probes.yaml` has one Preview probe for each decision and covers the
`vllm-sr/auto` entrypoint. Preview does not call a generation backend. The
CUDA device keeps this recipe out of the live CPU conformance run. A CPU
Preview pass is not a GB10 qualification. The operator profile records GPU
readiness, the routed request, and resource measurements in its own receipt.

## Limitations

The recipe configures one model, Kai 0.6B, on `cuda:0`. The generation
context windows in the config are example bounds for those aliases. Hardware
qualification is one GB10 host, one image digest, and one passing profile
receipt. Other models, other GPUs, Kubernetes GPU serving, and ARM64+CUDA as
a general platform stay outside this recipe. `CUDAAccelerator.validated`
remains false. The catalog floor of 4 GiB is not a measured capacity.

## References

- [Recipe metadata](metadata.yaml)
- [Runtime configuration](config.yaml)
- [Routing DSL](recipe.dsl)
- [Evaluation probes](probes.yaml)
- [Recipe authoring and conformance](../CONFORMANCE.md)
- [GB10 qualification profile](../../../tools/qualification/gb10/qualification.py)
