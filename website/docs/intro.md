---
sidebar_position: 1
sidebar_label: Introduction
description: Build a programmable Mixture-of-Models system behind one stable model API.
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
  <p className="docs-intro-brand__tagline">Make your Mixture-of-Models programmable.</p>
</div>

vLLM Semantic Router is an open-source routing and control layer for building
Mixture-of-Models systems across heterogeneous AI infrastructure. Applications
call a stable OpenAI- or Anthropic-compatible endpoint while the serving layer
chooses—or composes—the capability path for each request.

## The problem: an AI request is more than traffic

Modern AI applications rarely rely on one interchangeable model. A request may
need a fast local model, a specialist or frontier model, retrieval, memory,
tools, a verifier, or several models working together. Those paths may span
the cloud, a data center, or the edge.

Each path carries different tradeoffs in capability, latency, cost, and trust.
The right choice can also change with the request, user, session, and available
infrastructure.

When every application hard-codes these choices, product code becomes coupled
to the current model fleet. The same routing logic is repeated across clients,
and it becomes difficult to change, explain, or evaluate as the system grows.

## The idea: make intelligence programmable

Semantic Router moves that decision into a shared layer in the request path. It
can observe the work in front of it—intent, difficulty, context, modality,
identity, risk, preference, and system state—then resolve a stable entrypoint
to an isolated recipe.

A recipe can choose one model, escalate through a cascade, coordinate a bounded
multi-model workflow, or attach behavior such as retrieval, memory, tool
filtering, caching, safety checks, and verification. The application keeps one
familiar API while the capability path can evolve behind it.

The result is more than a model name:

- **The right model path:** direct, specialist, local, cascade, or collaborative.
- **The right supporting capabilities:** retrieval, memory, tools, prompts,
  caching, or verification where the request needs them.
- **The right execution boundary:** configured cloud, data center, or edge
  backends across heterogeneous hardware.
- **Evidence for what happened:** routing metadata plus configured feedback,
  replay, and evaluation workflows.

vLLM Semantic Router does not replace the gateway or the model servers. Envoy
continues to carry traffic, and inference runtimes continue to generate
responses. The Router coordinates the semantic work between them.

## Start with what you want to do

- **Run it locally:** follow the [Quickstart](/docs/installation) and send a
  request through the Router.
- **Find the pattern for your workload:** explore [use cases](overview/use-cases)
  from cloud and data center to edge and enterprise deployments.
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
