# AMD AI PC Recipe Model Card

## Overview

The AMD AI PC recipe runs ordinary requests on a small text model served on the
machine's CPU. It sends a request to an external model only when the request
itself says that is allowed. It also refuses two things this profile cannot do:
image input, and input larger than the local model's declared window.

The recipe covers the CPU path only. Radeon iGPU and Ryzen AI NPU acceleration
are not part of this recipe and are not qualified; see the
[AMD AI PC guide](../../../website/docs/installation/amd-ai-pc.md).

## Model details

| Lane | Reference alias | Purpose |
| --- | --- | --- |
| Local | `local/ai-pc` | Default and local-only traffic, served by a local Ollama server. |
| External | `cloud/external-chat` | Requests that explicitly allow an external model. |

Clients use `vllm-sr/auto`. The local alias is bound to the Ollama tag
`llama3.2:3b` and declares an 8192-token window. Replace both aliases with
backends that match the machine's memory and the deployment's data policy.

## Intended use

Use this recipe for:

- a developer or edge machine that should answer from a local model by default;
- text-only assistants where some requests may opt in to a larger external model;
- trying the standard `vllm-sr serve` flow on an AMD AI PC CPU.

Do not use it for image input or for workloads that need a vision model.

## Routing behavior

| Request pattern | Route | Behavior |
| --- | --- | --- |
| Contains an image | `unsupported_image_input` | Declined with a fixed message; no model is called. |
| Larger than the local window, and not opted in to an external model or marked local-only | `local_window_exceeded` | Declined with a fixed message; no model is called. |
| Marked local-only | `local_only` | Served by `local/ai-pc`. |
| Allows an external model and is not marked local-only | `external_opt_in` | Served by `cloud/external-chat`. |
| Everything else | `local_default` | Served by `local/ai-pc`. |

Local-only phrases outrank opt-in phrases. A request that contains both stays
local, and if it is also too large for the local window it is declined rather
than sent out. The phrases are English and Chinese keywords listed in
[`recipe.dsl`](recipe.dsl). They are an explicit-consent convention, not a
classifier.

## Requirements

- A local Ollama server reachable from the Router container as
  `host.docker.internal:11434`, serving the configured tag. Set the server's
  context length to at least the declared window.
- `OPENROUTER_API_KEY` in the environment only if the external route is used.
  Remove the external model and its route for local-only deployments.
- Enough free memory on the machine for the chosen model and its context.

## Data handling and safety

Routing happens before any backend call. A request that is not opted in never
reaches the external model, but it still passes through the Router and any
logs or stores enabled in the deployment. The recipe does not detect sensitive
content, so a request that is not marked local-only and does allow an external
model is sent out whatever it contains. Network isolation, backend ownership,
logging, and retention controls stay the operator's responsibility.

## Quick start

```bash
vllm-sr serve
```

In the Dashboard, connect the local Ollama model and, if allowed, an external
model, choose **AMD AI PC** in **Recipes**, assign both decisions, and publish
the Entrypoint. Follow the [AMD AI PC guide](../../../website/docs/installation/amd-ai-pc.md)
for the Ollama setup.

## Evaluation

The probes cover local-only and opt-in phrasing in English and Chinese, the
priority collisions between them, image refusal, oversized input, and the
default route. See [`probes.yaml`](probes.yaml) and the
[conformance guide](../CONFORMANCE.md). Probes check routing only. They do not
measure answer quality or run on AMD hardware.

## Limitations

- The recipe has not been run on an AMD AI PC. CPU status stays untested until a
  device owner supplies reproducible probes.
- The local model's quality, speed, and memory use depend on the machine and
  are not characterized here.
- Keyword consent can be missed or phrased differently by a user.
- Image input is refused, not routed to an accelerator.

## References

- [Recipe metadata](metadata.yaml)
- [Runtime configuration](config.yaml)
- [Routing DSL](recipe.dsl)
- [Evaluation probes](probes.yaml)
- [Recipe authoring and conformance](../CONFORMANCE.md)
