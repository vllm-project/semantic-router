---
sidebar_position: 3
title: Use Cases
description: How semantic routing applies across cloud services, data centers, edge deployments, and enterprise hybrid environments.
---

# Use Cases

Semantic routing is useful anywhere an application should ask for an outcome
without knowing the physical model topology. The same request contract can sit
over hosted APIs, a shared data-center fleet, a small edge pool, or a hybrid
enterprise deployment. What changes is the policy boundary and the system that
owns model execution.

The environments below are not mutually exclusive. Many deployments use a
data-center pool for general traffic, an edge model for private or offline work,
and selected cloud providers for specialized capabilities.

This is the practical Mixture-of-Models pattern: clients use a stable virtual
model identity, a recipe describes the routing behavior, and deployment-specific
provider bindings connect the recipe's logical model roles to the endpoints
available in each environment.

## At a glance

| Environment | Typical problem | What Semantic Router decides | What remains outside the Router |
| --- | --- | --- | --- |
| **Cloud** | Several provider APIs differ in capability, cost, latency, and availability. | Which approved provider or virtual objective should handle the request. | Provider capacity, billing, regional service health, and account policy. |
| **Data center** | A shared accelerator fleet serves models with different strengths and resource profiles. | Which model pool fits the task, policy, and requested objective. | Replica placement, batching, autoscaling, and device scheduling. |
| **Edge** | Capacity is limited and data may need to remain local or work offline. | Whether a local path is capable and whether remote escalation is allowed. | Device runtime, model packaging, power limits, and network availability. |
| **Enterprise hybrid** | Workloads cross tenants, regions, trust zones, and internal or external providers. | Which paths are eligible under identity, residency, capability, and routing policy. | Identity issuance, network controls, key management, and data governance. |

## Cloud: one policy over several model services

Cloud applications often integrate more than one model provider. A provider may
offer the best vision model, another may be preferred for long context, and a
smaller hosted model may be the economical default for routine chat.

Semantic Router can expose one stable model API while decisions account for:

- task intent, modality, context length, and tool requirements;
- provider allowlists and regional or contractual constraints;
- cost, latency, or quality objectives within the eligible set; and
- explicit escalation or rejection when pre-routing evidence rules out a path.

This keeps provider selection out of application code. It does not make provider
accounts interchangeable: credentials, quotas, data-use terms, and regional
availability still need to be managed explicitly.

**Example:** a document assistant routes ordinary text to an economical model,
image-bearing requests to a vision-capable provider, and evidence-sensitive
questions to a verification route. Clients continue to request the same public
model name.

## Data center: match workloads to a shared model fleet

A data center may serve several open or privately fine-tuned models across one
accelerator fleet. The serving platform knows which replicas are healthy and
where capacity is available; it usually does not know which model family best
fits the meaning of a request.

Semantic Router adds that model-level decision:

- send code, mathematics, language, or domain work to specialist pools;
- reject candidates that lack the required modality, context, or tool support;
- reserve larger models or multi-model workflows for requests that justify
  their additional compute; and
- expose balanced, fast, economical, or accuracy-first virtual models over the
  same physical fleet.

The boundary is deliberate: Semantic Router chooses the model or model pool;
the inference platform chooses a replica and owns batching, cache locality,
autoscaling, and accelerator scheduling.

**Example:** an internal AI platform exposes `assistant-fast` and
`assistant-accurate`. Both entrypoints share the same provider pool, but their
isolated recipes use different eligibility rules and selection algorithms.

## Edge: local-first routing under constrained capacity

At the edge, privacy, intermittent connectivity, memory, and power can matter
more than access to the largest model. A useful policy starts with the local
capabilities and treats remote execution as an explicit choice rather than an
implicit fallback.

Semantic Router can:

- keep sensitive or offline-eligible traffic on a local model;
- select among small specialists available on the device or local network;
- block requests that the local pool cannot safely handle; or
- escalate selected requests to an approved remote model when policy and
  connectivity allow it.

“Local” is not an end-to-end privacy guarantee. Network routes, logs, replay,
caches, tool calls, and remote fallbacks must follow the same boundary.

**Example:** a field assistant answers common text requests locally, routes
image requests to an on-premises vision service when connected, and fails
closed instead of sending protected content to a public endpoint.

## Enterprise: enforce policy across hybrid model access

Enterprise deployments commonly combine internal models, regional data-center
pools, edge systems, and approved external providers. The central problem is
not simply finding the “best” model; it is finding the best path that is allowed
for this user, tenant, data class, and workload.

Semantic Router can make those choices explicit:

- use authorization, metadata, and content signals as eligibility constraints;
- isolate product, tenant, or objective-specific policy in separate recipes;
- keep regulated traffic within an approved region or local pool;
- preserve tool, modality, context, and provider compatibility; and
- record routing outcomes for evaluation or audit when replay is deliberately
  enabled and protected.

The Router complements identity, network segmentation, secrets management,
retention policy, and provider governance. It does not replace them, and signal
detection alone does not enforce a policy unless a decision acts on it.

**Example:** a company-wide assistant uses one API across business units. Public
questions can use approved hosted providers, confidential prompts stay on an
internal pool, and a restricted recipe removes remote and tool-enabled paths for
high-sensitivity workloads.

## Patterns that span environments

The same routing patterns appear in all four environments:

### Capability routing

Eliminate models that cannot satisfy the request's modality, context, tool, or
language requirements before comparing cost or latency.

### Specialist routing

Use semantic and deterministic signals to direct code, mathematics, research,
or domain-specific work to an appropriate model pool.

### Objective-based virtual models

Expose stable public names for balanced, fast, economical, accuracy-first, or
private behavior while operators evolve the underlying pool.

### Bounded recovery and orchestration

Escalate on low confidence, compare several answers, or run a bounded workflow
only where the additional calls have measurable value.

### Policy-first routing

Apply authorization, locality, data handling, and capability constraints before
an optimization algorithm ranks the remaining candidates.

## Choosing a starting point

1. Identify the physical model pools and the system that owns their execution.
2. Define hard eligibility boundaries before quality, latency, or cost goals.
3. Expose one public entrypoint for each behavior clients should intentionally
   choose.
4. Start with a simple decision and static selection.
5. Add signals or algorithms only when they change an observed routing outcome.
6. Validate representative requests with probes before changing production
   policy.

Continue with the [Routing Pipeline](signal-driven-decisions), or browse the
[recipe Model Cards](https://github.com/vllm-project/semantic-router/tree/main/config/recipes)
for complete examples.
