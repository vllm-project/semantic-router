---
sidebar_position: 1
sidebar_label: Introduction
description: An open, programmable decision layer for models and compute.
---

import ThemedImage from '@theme/ThemedImage';

# Welcome to vLLM Semantic Router

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

vLLM Semantic Router lets agent harnesses use models and compute through an
open, programmable decision layer. A harness calls a stable OpenAI- or
Anthropic-compatible endpoint; explicit policy selects a model, coordinates a
bounded multi-model strategy, and applies route-specific behavior over
configured backends.

Our vision is intelligence that can evolve beyond any one model. Our mission is
to make the decisions connecting agent harnesses to models and compute open,
programmable, and observable.

## The problem: an AI request is more than traffic

An agent harness manages a task across model calls, tool execution, and evolving
context. One inference call may suit a fast local model; another needs a
specialist, more context capacity, or bounded verification across several
models. Those paths may span the cloud, a data center, or the edge.

Each path carries different tradeoffs in capability, latency, cost, and trust.
The right choice can also change with the request, user, session, and available
infrastructure.

When every harness hard-codes these choices, its integration code becomes
coupled to the current model fleet. The same routing logic is repeated across
clients and becomes difficult to change, explain, or evaluate as the system
grows.

## The idea: make intelligence programmable

Semantic Router moves that decision into a shared layer in the request path. It
can observe the work in front of it—intent, difficulty, context, modality,
identity, risk, preference, and system state—then resolve a stable entrypoint
to an isolated recipe.

A recipe can choose one model, escalate through a cascade, coordinate a bounded
multi-model workflow, or attach behavior such as retrieval, memory, tool
filtering, caching, safety checks, and verification. The harness keeps a stable
model API while the policy and configured model pool evolve behind it.

The result is more than a model name:

- **The right model path:** direct, specialist, local, cascade, or collaborative.
- **The right supporting capabilities:** retrieval, memory, tool filtering, prompts,
  caching, or verification where the request needs them.
- **The right execution boundary:** configured cloud, data center, or edge
  backends across heterogeneous hardware.
- **Evidence for what happened:** routing metadata plus configured feedback,
  replay, and evaluation workflows.

The harness owns the agent loop, tool execution, and task lifecycle. Semantic
Router owns per-call routing policy and configured model collaboration. Gateways
and Envoy carry traffic; inference platforms execute models and manage replica
placement, batching, and capacity. Here, decisions about **compute** concern the
configured inference path and bounded model calls, with execution managed by
those platforms.

## Start with what you want to do

- **Run it locally:** follow the [Quickstart](/docs/installation) and send a
  request through the Router.
- **Connect an agent harness:** follow the [agent harness
  guide](/docs/installation/agent-harness) for the integration and ownership
  boundaries.
- **Find the pattern for your workload:** explore [use cases](overview/use-cases)
  for agent model calls across cloud, data center, edge, and enterprise deployments.
- **Understand the system:** read the [System
  Overview](overview/semantic-router-overview) and [Routing
  Pipeline](overview/signal-driven-decisions).
- **Create a stable model experience:** learn how [entrypoints and
  recipes](tutorials/global/entrypoints-and-recipes) turn one shared model pool
  into purpose-built virtual models.
- **Choose an environment:** compare [Docker, Kubernetes, and hardware
  paths](installation/deployment-options).

## Project

vLLM Semantic Router is open source under the Apache 2.0 license. See the
[contributing guide](https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md)
to propose a change or join the community.
