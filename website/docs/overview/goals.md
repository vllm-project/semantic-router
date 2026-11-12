---
sidebar_position: 1
title: Why Semantic Routing
description: Why agent harnesses need a programmable decision layer over heterogeneous models and compute.
---

# Why Semantic Routing

Models differ in reasoning, speed, price, context length, modality, tools,
location and safety, and the pool changes with every upgrade or outage.
Semantic Router gives harnesses one stable model API and moves model choice
into a policy you can read, test and change.

- **One stable API.** Harnesses call a public model name; operators change the
  pool behind it.
- **Hard rules first.** Identity, data location, modality and context limits
  remove ineligible models before cost or latency is compared.
- **Plain-language judgments.** One decision model answers the policy's
  questions about each request, so rules read as questions, not code.
- **Measured change.** Replay, evaluation and metrics show whether a policy
  change helped.

## The problems it solves

| Problem | Without a routing layer | With Semantic Router |
| --- | --- | --- |
| **Model choice is coupled to the harness** | Every integration hard-codes its models. | Public entrypoints decouple the name a harness calls from the models behind it. |
| **Constraints and preferences are mixed** | A cheap model can win even when it is not allowed. | Decisions remove invalid paths first; algorithms rank only what remains. |
| **One rule is not enough** | Keywords miss intent, and a model's guess can override an authorization rule. | Signals keep each kind of evidence separate; decisions compose them. |
| **The pool keeps changing** | Upgrades, scaling and outages break routing code. | Health, latency and load inform selection without becoming the policy. |
| **Some answers need several models** | Escalation and review are ad hoc retries. | Escalation and orchestration are explicit, bounded steps of a route. |

Routing is both a semantic and a systems problem. The request says what the
work needs; the pool says what is available. The Router joins the two views
without letting either one become the whole policy.

## Design goals

1. **One stable API over many backends.** Harnesses use a public entrypoint
   while operators manage the pool behind it.
2. **Policy that can be read and tested.** Signals, projections, decisions,
   algorithms and plugins are named configuration objects, not scattered
   conditionals.
3. **Hard boundaries before optimization.** Ineligible routes are removed
   before quality, latency, cost or load influence selection.
4. **Model selection and bounded collaboration.** A route can choose one
   model, escalate, compare, or coordinate several within configured limits.
5. **Operational feedback.** Replay, evaluation, metrics and user feedback
   support deliberate policy changes.

## What Semantic Router is not

- **Not the agent harness.** The harness owns the task loop, tool execution
  and task state. The Router decides each model call.
- **Not the chat model server.** vLLM, Ollama or hosted providers generate
  the answers. The built-in model runtime serves only the Router's own
  decision and embedding models.
- **Not only a load balancer.** Replica health matters, but request meaning
  and policy decide which model pool is eligible.
- **Not a quality guarantee.** Routing is as good as the configured models,
  signals, policy and evaluation data.
- **Not a security perimeter.** It enforces routing policy inside a broader
  architecture of network, identity and data-governance controls.

:::tip Serve decisions without a chat backend
Applications can also ask the decision model directly, through the
[System One](../system-one/overview) API.
:::

## Next

- [System Overview](semantic-router-overview): the components and the request
  lifecycle.
- [Use Cases](use-cases): routing patterns for cloud, data-center, edge and
  enterprise pools.
- [Connect an agent harness](../installation/agent-harness): integrate your
  harness.
