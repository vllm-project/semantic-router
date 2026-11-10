---
sidebar_position: 1
title: Why Semantic Routing
description: Why heterogeneous model fleets need a decision layer between applications and inference backends.
---

# Why Semantic Routing

There is no single best model for every request. Models differ in reasoning,
latency, price, language, context length, modality, tool use, deployment
location, and safety profile. The pool also changes as models are upgraded,
scaled, or temporarily unavailable.

Without a routing layer, every application has to understand those differences.
Model choice becomes client-side conditionals, policy is copied across services,
and a backend change can require an application release.

Semantic routing moves that decision into the serving path.

## The problems it addresses

### Model choice leaks into applications

An application should express what it needs, not maintain a list of model
endpoints. Stable public model names let the routing policy and physical pool
change without changing every client.

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
execution patterns and should be explicit rather than hidden behind retries in
application code.

## Design goals

vLLM Semantic Router is designed around five goals:

1. **One stable API over many backends.** Clients use a public entrypoint while
   operators manage the pool behind it.
2. **Policy that can be read and tested.** Signals, projections, decisions,
   algorithms, and plugins are named configuration objects rather than
   scattered conditionals.
3. **Hard boundaries before optimization.** Ineligible routes are removed
   before quality, latency, cost, or load influences selection.
4. **Selection and orchestration in one system.** A route can choose one model,
   cascade, compare, or coordinate several models.
5. **Operational feedback.** Replay, evaluation, metrics, and user feedback
   support deliberate policy changes.

## What Semantic Router is not

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
patterns.
