---
sidebar_position: 1
sidebar_label: Introduction
description: An open, programmable decision layer for models and compute.
---

import ThemedImage from '@theme/ThemedImage';

# Intelligence beyond any one model.

<div className="docs-intro-brand">
  <ThemedImage
    className="docs-intro-brand__logo"
    alt="vLLM Semantic Router"
    sources={{
      light: '/img/vllm-sr-logo.light.png',
      dark: '/img/vllm-sr-logo.white.png',
    }}
  />
  <p className="docs-intro-brand__tagline">An open, programmable <strong>decision layer</strong> for models and compute.</p>
</div>

Give your agent harness a stable model API. vLLM Semantic Router selects or
combines models through explicit policy, behind an OpenAI- or
Anthropic-compatible endpoint, so you can change models and policy without
rewriting the harness integration.

- **Route each call:** a fast local model for one call, a specialist, a longer
  context or a second opinion for the next.
- **Judge with one model:** a decision model answers the policy's questions
  about each request.
- **Serve judgments directly:** applications can also ask the decision model
  through the [System One](system-one/overview) API.

## Why route?

Capability, latency, cost and trust vary with the request, the user, the
session and the infrastructure at hand. A shared routing policy keeps these
choices out of each harness's code.

## What you can program

An entrypoint selects an isolated recipe. Its signals capture intent,
difficulty, context, modality, identity, risk and runtime observations. Use
them to:

- **Select or combine models:** choose a local model or specialist, escalate
  from a small model to a stronger one, or run a bounded multi-model workflow.
- **Add route behavior:** prompts, retrieval, memory, tool filtering, caching,
  safety checks and verification.
- **Choose an execution path:** cloud, data-center or edge backends across
  heterogeneous hardware.
- **Inspect and improve decisions:** routing metadata, feedback, replay and
  evaluation.

## Who does what

| Part | Owns |
| --- | --- |
| **Agent harness** | The agent loop, tool execution and task state |
| **Semantic Router** | Per-call policy and bounded model collaboration |
| **Model runtime** | The Router's own decision and embedding models |
| **Inference platforms** | Chat backends, with their placement, batching and capacity |

The Router serves its own listeners by default; an Envoy-based gateway is
optional. See the [component architecture](overview/component-architecture)
for the module boundaries.

## Start here

- [Run the Quickstart](installation/installation.md).
- [Connect an agent harness](installation/agent-harness).
- Explore [use cases](overview/use-cases).
- Read the [System Overview](overview/semantic-router-overview) and
  [Routing Pipeline](overview/signal-driven-decisions).
- Build virtual models with [entrypoints and recipes](tutorials/global/entrypoints-and-recipes).
- Compare [deployment options](installation/deployment-options).

## Project

vLLM Semantic Router is open source under the Apache 2.0 license. See the
[contributing guide](https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md)
to propose a change or join the community.
