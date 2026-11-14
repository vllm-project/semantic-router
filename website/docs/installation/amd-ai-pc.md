---
title: AMD AI PC
description: Run the Router with a local Ollama model on an AMD AI PC CPU, and see which AI PC paths are qualified.
---

# Deploy on an AMD AI PC

This guide runs the standard `vllm-sr serve` flow on an AMD AI PC with a small
local model on the CPU. It uses the maintained
[`amd-ai-pc` recipe](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/amd-ai-pc/README.md),
which answers locally by default and uses an external model only when a request
allows it.

## What is qualified

An AI PC has three places a model could run. Each is a separate row, and none
of them is tested yet.

| Path | Status | Notes |
| --- | --- | --- |
| CPU | Experimental, untested | The recipe and this guide exist. No AMD AI PC owner has supplied reproducible probes. Tracked by [#4932](https://github.com/vllm-project/semantic-router/issues/4932). |
| Radeon iGPU | Unqualified | No exercised runtime. Tracked by [#4933](https://github.com/vllm-project/semantic-router/issues/4933). |
| Ryzen AI NPU | Unqualified | No exercised runtime. Tracked by [#4934](https://github.com/vllm-project/semantic-router/issues/4934). |

Do not assume the data-center ROCm images and settings in
[AMD ROCm](amd-rocm) work unchanged on an AI PC, and do not read the CPU row as
covering the other two.

## Prerequisites

- Semantic Router installed and runnable with [`vllm-sr serve`](/docs/installation).
- Docker, and [Ollama](https://ollama.com/) on the same machine.
- Enough free memory for the chosen model and its context window.
- `OPENROUTER_API_KEY` only if you want the external route.

## Start the local model

Follow [Configure models with Ollama](ollama) to install Ollama, make it
reachable from Docker, and pull the model. The recipe binds `local/ai-pc` to the
tag `llama3.2:3b` through `host.docker.internal:11434`:

```bash
ollama pull llama3.2:3b
```

The recipe declares an 8192-token window for this model and refuses estimated input of 6K tokens or more, which leaves room for the reply. Set the Ollama server's
context length to at least that value, or the declared limit is higher than what
the server accepts.

## Serve the recipe

```bash
vllm-sr serve
```

In the Dashboard, connect the Ollama model and, if allowed, an external model.
Choose **AMD AI PC** in **Recipes**, assign both decisions, and publish the
Entrypoint. Clients then use the model name `vllm-sr/auto`.

## How requests are routed

| Request | Result |
| --- | --- |
| Contains an image | Declined with a fixed message. No model is called. |
| Estimated at 6K tokens or more, and not marked as allowing an external model, or marked local-only | Declined with a fixed message. No model is called. |
| Contains a local-only phrase, such as `keep this local` | Served by the local model. |
| Contains an opt-in phrase, such as `you may use a cloud model`, and no local-only phrase | Served by the external model. |
| Anything else | Served by the local model. |

Local-only phrases outrank opt-in phrases. The phrases are English and Chinese
keywords in the recipe's `recipe.dsl`. They are an explicit-consent convention,
not sensitive-data detection, so a request that opts in is sent out whatever it
contains.

```bash
curl -s http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"vllm-sr/auto","messages":[{"role":"user","content":"Keep this local. Summarize why the sky is blue."}]}'
```

## Limits and failure behavior

- **Image input** is refused, because the profile has no qualified local vision
  path. It is not routed to an accelerator.
- **Oversized input** is refused instead of being sent to an external model
  unless the request allows one.
- **Upstream errors** are not retried on another model. Failover is tracked
  separately in [#2294](https://github.com/vllm-project/semantic-router/issues/2294).
- **Model quality and speed** depend on the machine and the model, and are not
  characterized here.

## Qualify your machine

Routing can be checked without hardware, but a tested status needs a real AI PC.
With the stack running, evaluate the recipe's probes:

```bash
make recipe-conformance-eval \
  RECIPE_CONFORMANCE_RECIPE=amd-ai-pc \
  RECIPE_CONFORMANCE_ROUTER_URL=http://127.0.0.1:8080
```

When you report a result on [#4932](https://github.com/vllm-project/semantic-router/issues/4932),
include the OS, container runtime, CPU model, memory, Ollama version, model tag
and digest, Router image tag, and the probe report. Leave out hostnames, paths,
and secrets. See [Recipe authoring and conformance](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/CONFORMANCE.md)
for how probes are evaluated.
