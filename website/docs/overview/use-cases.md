---
sidebar_position: 3
title: Use Cases
description: Programmable model decisions for agent harnesses across cloud, data center, edge, and enterprise environments.
---

import Tabs from '@theme/Tabs'
import TabItem from '@theme/TabItem'

# Use Cases

Route each model call by task, capability and policy. One stable entrypoint
lets your harness use hosted APIs, data-center fleets, edge models or a mix,
without changing its task loop or tools.

- **Inside an agent task:** pick the right model for each call, keep a tool
  loop on one model, and verify selected answers.
- **Across environments:** the same policy works for cloud, data-center, edge
  and hybrid pools.
- **Reusable patterns:** capability, specialist, objective, recovery and
  policy-first routing.

## Decisions inside an agent task

| Harness need | Router policy | Harness responsibility |
| --- | --- | --- |
| A coding or research call needs a different model capability. | Match request signals to a specialist lane, and filter candidates by declared capabilities and context capacity. | Choose the next task step and execute any returned tool calls. |
| A tool-using session continues across turns. | With session protection configured, keep an eligible model through active tool loops or provider state that can't move. | Supply the conversation and a stable session identity; manage tool results and task state. |
| A selected call benefits from verification. | Run a configured, bounded escalation, panel or multi-model workflow. | Decide how to use the returned answer in the larger task. |
| Sensitive work needs an approved execution path. | Apply explicit eligibility and data-handling policy over local or remote backends. | Enforce the same boundary in tools, task storage and other services. |

Start with the [agent harness guide](../installation/agent-harness) and the
[Agent Routing recipe](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/agent/README.md).

## Deployment environments

Most deployments mix environments: a data-center pool for general traffic, an
edge model for private or offline work, and cloud providers for specialized
skills. The virtual model name stays stable while each environment's
endpoints change behind it.

| Environment | Typical problem | What Semantic Router decides | What stays outside the Router |
| --- | --- | --- | --- |
| **Cloud** | Provider APIs differ in capability, cost, latency and availability. | Which approved provider or virtual objective handles the request. | Provider capacity, billing, regional health and account policy. |
| **Data center** | A shared accelerator fleet serves models with different strengths. | Which model pool fits the task, policy and objective. | Replica placement, batching, autoscaling and device scheduling. |
| **Edge** | Capacity is limited, and data may need to stay local or work offline. | Whether a local path is capable, and whether remote escalation is allowed. | Device runtime, model packaging, power and network availability. |
| **Enterprise hybrid** | Workloads cross tenants, regions, trust zones and providers. | Which paths are eligible under identity, residency, capability and policy. | Identity issuance, network controls, key management and data governance. |

<Tabs groupId="use-case-environment">
<TabItem value="cloud" label="Cloud" default>

**One policy over several model services.** One provider may have the best
vision model, another the longest context, and a small hosted model the
cheapest routine chat.

The Router exposes one model API while decisions account for:

- task intent, modality, context length and tool requirements;
- provider allowlists and regional or contractual constraints;
- cost, latency or quality objectives within the eligible set;
- explicit escalation, or rejection when a path is ruled out.

**Example:** a document assistant sends ordinary text to an economical model,
images to a vision-capable provider, and evidence-sensitive questions to a
verification route. Clients keep requesting the same model name.

:::note
Provider accounts are not interchangeable. Manage credentials, quotas,
data-use terms and regional availability explicitly.
:::

</TabItem>
<TabItem value="data-center" label="Data center">

**Match workloads to a shared fleet.** The serving platform knows which
replicas are healthy; it usually doesn't know which model fits a request.

The Router adds that model-level decision:

- send code, mathematics, language or domain work to specialist pools;
- reject candidates that lack the needed modality, context or tool support;
- keep larger models and multi-model workflows for requests that justify them;
- expose balanced, fast, economical or accuracy-first virtual models over the
  same fleet.

**Example:** an internal platform exposes `assistant-fast` and
`assistant-accurate`. Both share one provider pool, but their recipes use
different eligibility rules and selection algorithms.

:::note
The Router picks the model or pool. The inference platform still picks the
replica and owns batching, cache locality, autoscaling and scheduling.
:::

</TabItem>
<TabItem value="edge" label="Edge">

**Local-first routing under constrained capacity.** At the edge, privacy,
intermittent connectivity, memory and power can matter more than the largest
model. Remote execution is an explicit choice, not an implicit fallback.

The Router can:

- keep sensitive or offline-eligible traffic on a local model;
- select among small specialists on the device or local network;
- block requests the local pool cannot safely handle;
- escalate selected requests to an approved remote model when policy and
  connectivity allow.

**Example:** a field assistant answers common text locally, sends images to
an on-premises vision service when connected, and fails closed rather than
sending protected content to a public endpoint.

:::warning
"Local" is not an end-to-end privacy guarantee. Network routes, logs, replay,
caches, tool calls and remote fallbacks must follow the same boundary.
:::

</TabItem>
<TabItem value="enterprise" label="Enterprise hybrid">

**Enforce policy across hybrid model access.** The question is not only which
model is best, but which path is allowed for this user, tenant, data class
and workload.

The Router makes those choices explicit:

- authorization, metadata and content signals act as eligibility rules;
- separate recipes isolate product, tenant or objective policy;
- regulated traffic stays in an approved region or local pool;
- tool, modality, context and provider compatibility are preserved;
- route outcomes are recorded for evaluation or audit, when replay is enabled
  and protected.

**Example:** a company-wide assistant serves every business unit through one
API. Public questions use approved hosted providers, confidential prompts stay
on an internal pool, and a restricted recipe removes remote and tool-enabled
paths for high-sensitivity work.

:::note
A signal only reports; a decision must act on it. The Router complements
identity, network segmentation, secrets management, retention and provider
governance; it does not replace them.
:::

</TabItem>
</Tabs>

## Patterns that span environments

| Pattern | What it does | Built from |
| --- | --- | --- |
| **Capability routing** | Drops models that can't handle the request's modality, context, tools or language, before comparing cost or latency. | Candidate requirements, `input_modality`, `context` |
| **Specialist routing** | Sends code, mathematics, research or domain work to a fitting pool. | `domain`, `systemone`, `keyword` signals |
| **Objective-based virtual models** | Gives stable public names to balanced, fast, economical, accuracy-first or private behavior. | Entrypoints, [MoM V1](mom-model-family#mom-v1) |
| **Bounded recovery and orchestration** | Escalates on low confidence, compares answers or runs a workflow, only where the extra calls pay off. | `escalate`, `fusion`, `remom`, `workflows` |
| **Policy-first routing** | Applies authorization, locality, data handling and capability rules before any optimizer ranks candidates. | `authz` signals, decisions |

## Choosing a starting point

1. Identify the model pools, and the system that runs each one.
2. Define the hard eligibility boundaries before quality, latency or cost
   goals.
3. Expose one public entrypoint for each behavior clients should choose.
4. Start with a simple decision and static selection.
5. Add signals or algorithms only when they change an observed routing
   outcome.
6. Validate representative requests with probes before you change production
   policy.

## Next

- [Routing Pipeline](signal-driven-decisions): how the policy layers fit
  together.
- [Recipe Model Cards](https://github.com/vllm-project/semantic-router/tree/main/config/recipes):
  complete examples.
