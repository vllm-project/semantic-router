---
sidebar_position: 1
title: Why Semantic Routing
description: Why agent harnesses need a programmable decision layer over heterogeneous models and compute.
---

# Why Semantic Routing

Choose models through policy instead of hard-coding each harness integration.
Models differ in reasoning, latency, price, language, context length, modality,
tools, location, and safety. Their availability changes with upgrades, scaling,
and outages.
A shared routing policy lets harnesses adapt to these differences within a task
without duplicating model-selection logic.

## The problems it addresses

### Model choice becomes coupled to the harness

A harness needs a stable model contract while its task loop and tools evolve.
Public model entrypoints let routing policy and the physical pool change
independently of each harness integration.

### Constraints and preferences get mixed together

Some requirements are non-negotiable: authorization, privacy, data residency,
modality, context capacity, or tool compatibility. Others are objectives to
optimize, such as quality, latency, and cost. A useful router eliminates invalid
paths first and ranks only the remaining candidates.

### One routing rule is not enough

Keywords can express a hard policy but cannot capture every semantic intent.
A classifier can recognize intent but should not override an authorization
boundary. Runtime metrics can choose a healthy replica but do not understand
the task. Semantic Router keeps these responsibilities separate, then composes
them into one decision.

### The physical pool is dynamic

Routing is both a semantic and a systems problem. The request describes the
workload; the model pool contributes capacity, health, latency, and placement.
The Router must connect those two views without making either one the entire
policy.

### Some answers require collaboration

Selecting one model is often enough. Other tasks benefit from escalation,
verification, parallel opinions, or a bounded workflow. These are distinct
execution patterns and should be explicit in routing policy. The harness remains
responsible for the surrounding task loop and tool execution.

## Design goals

vLLM Semantic Router is designed around five goals:

1. **One stable API over many backends.** Harnesses use a public entrypoint while
   operators manage the pool behind it.
2. **Policy that can be read and tested.** Signals, projections, decisions,
   algorithms, and plugins are named configuration objects rather than
   scattered conditionals.
3. **Hard boundaries before optimization.** Ineligible routes are removed
   before quality, latency, cost, or load influences selection.
4. **Model selection and bounded collaboration.** A route can choose one model,
   cascade, compare, or coordinate several models within configured limits.
5. **Operational feedback.** Replay, evaluation, metrics, and user feedback
   support deliberate policy changes.

## What Semantic Router is not

- The agent harness owns task orchestration, tool execution, and durable task
  state. The Router supplies decisions for its model calls.
- It is not an LLM server. Backends such as vLLM, Ollama, or hosted providers
  still run the models.
- It is not only a load balancer. Replica health matters, but request meaning
  and policy determine which model pool is eligible.
- It is not a universal quality guarantee. Routing quality depends on the
  configured models, signals, policy, and evaluation data.
- It is not a replacement for network, identity, or data-governance controls.
  It enforces routing policy inside a broader security architecture.

## Next

Read the [System Overview](semantic-router-overview) for the components and
request lifecycle, then see [Use Cases](use-cases) for concrete routing
patterns and the [agent harness guide](/docs/installation/agent-harness) for
integration.
