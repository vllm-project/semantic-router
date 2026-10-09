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
combines models through explicit policy, behind an OpenAI- or Anthropic-compatible
endpoint. Change the models and policy without rewriting the harness integration.

Applications can also call [System One](model-runtime/quickstart) directly for
typed judgments. Router and Engine modes share a frontend and model runtime;
recipes add the optional routing layer. See the [component architecture](overview/component-architecture)
for the module boundaries and replica pools.

## Why route?

One call needs a fast local model; another needs a specialist, longer context,
or verification across models. Capability, latency, cost, and trust vary with the
request, user, session, and available infrastructure. Shared routing policy keeps
these choices out of each harness's code.

## What you can program

An entrypoint selects an isolated recipe. Its signals capture intent, difficulty,
context, modality, identity, risk, preference, and configured runtime observations.
Use those signals to:

- **Select or combine models:** choose a local model or specialist, escalate
  through a cascade, or run a bounded multi-model workflow.
- **Add route behavior:** prompts, retrieval, memory, tool filtering, caching,
  safety checks, and verification.
- **Choose an execution path:** configured cloud, data-center, or edge backends
  across heterogeneous hardware.
- **Inspect and improve decisions:** routing metadata, feedback, replay, and
  evaluation.

The harness owns the agent loop, tool execution, and task state. The Router owns
per-call policy and bounded model collaboration. The standalone frontend carries
requests by default; Envoy is an optional transport integration. The model runtime
executes judgment and feature-extraction tasks, while external inference platforms
run the Chat backends and manage their placement, batching, and capacity.

## Start here

- [Run the Quickstart](/docs/installation).
- [Connect an agent harness](/docs/installation/agent-harness).
- Explore [use cases](overview/use-cases).
- Read the [System Overview](overview/semantic-router-overview) and
  [Routing Pipeline](overview/signal-driven-decisions).
- Build virtual models with [entrypoints and recipes](tutorials/global/entrypoints-and-recipes).
- Compare [deployment options](installation/deployment-options).

## Project

vLLM Semantic Router is open source under the Apache 2.0 license. See the
[contributing guide](https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md)
to propose a change or join the community.
