---
title: Entrypoints
description: Expose stable virtual model names that select a routing recipe through the standard OpenAI-compatible model field.
---

# Entrypoints

## Overview

An Entrypoint is a public virtual Model whose name set selects one Recipe and a
complete Model assignment. Clients
select it through the normal OpenAI-compatible `model` field, so they do not
need a Router-specific API or header.

## What Problem Does It Solve?

Entrypoints solve a common coupling problem: an application can ask for a
stable objective such as `vllm-sr/mom-v1-flash` while operators change the models,
thresholds, or algorithms behind that objective.

## When to Use

Create an entrypoint when you want to:

- publish latency, quality, cost, safety, or team-specific routing objectives;
- move a client between policy versions without exposing backend model IDs; or
- run several isolated policies in one Router deployment.

Publish an explicit Entrypoint even when every routed request uses one policy.
Use a concrete Model name only when the caller deliberately wants to bypass
signals, decisions, algorithms, and route-local plugins.

## Configuration

An Entrypoint has one or more public `model_names`, one Recipe, and complete
Decision assignments:

```yaml
entrypoints:
  - model_names: [vllm-sr/mom-v1-flash, company/fast]
    recipe: flash
    assignments:
      fast: {models: [{model: local/fast}]}

recipes:
  - name: flash
    description: Low-latency routing for interactive requests.
    routing:
      strategy: priority
      decisions:
        - name: fast
          description: Handle the request with the assigned low-latency Model.
          priority: 100
          rules: {operator: AND, conditions: []}
```

Both names select the same Recipe. A client uses either name like any other
chat-completions model:

```bash
curl http://localhost:8899/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "vllm-sr/mom-v1-flash",
    "messages": [{"role": "user", "content": "Summarize this request."}]
  }'
```

The entrypoint name never reaches a provider. After the recipe chooses a
backend, the Router rewrites the request to that backend's model name.

## Assign Models without copying a Recipe

Keep routing policy reusable and bind its decisions to the models available in
each deployment:

```yaml
entrypoints:
  - model_names: [company/assistant]
    recipe: assistant
    assignments:
      quick: {models: [{model: local/fast}]}
      complex:
        models:
          - model: hosted/frontier
            priority: 0
            reasoning: {enabled: true, effort: high}
          - model: hosted/reviewer
            priority: 1
        fallback:
          strategy: priority
          on: [unavailable, timeout]
```

Each key under `assignments` is a Decision name from the selected Recipe.
Its value is that decision's assignment set. The required
`models` list identifies the active tier; an optional priority `fallback` policy
defines closed, Router-owned failover. Model names may change without changing
the Entrypoint's public name.

Lower priority numbers run first. The Router advances to the next tier only for
the failure classes named in `fallback.on`, before any response is visible.

The Router validates the complete effective Recipe before publication or
startup. Every Recipe and Model name must resolve, every decision must be
assigned, and a pool cannot be empty or repeat a Model. The assignment also must
satisfy the algorithm's candidate requirements.

## Request resolution

| Requested model | Router behavior |
| --- | --- |
| An Entrypoint `model_names` value | Evaluate only its compiled Recipe and assignments. |
| Any other name | No Entrypoint alias is created implicitly. |
| A concrete Model name, alias, or LoRA name | Send directly to that Model without Recipe routing. |

Entrypoints are listed by `/v1/models` with routing metadata. Successful routed
responses expose `x-vsr-selected-recipe`; Router Replay and Insights can also
filter records by recipe.

## Naming and validation rules

Configuration loading rejects an entrypoint when:

- `model_names` is empty or `recipe` references an unavailable Recipe revision;
- an action omits a decision or references an unavailable Model revision;
- the same virtual name is claimed by more than one entrypoint; or
- a virtual name collides with a configured Model name, Model alias, or LoRA.

Choose names that describe a durable client contract, not the current backend.
Do not put tenant data or secrets in a name: entrypoints appear in model
discovery, response metadata, metrics, and operational records.

An entrypoint is a policy selector, not a security boundary. Recipes share the
Router process and configured infrastructure; use network, compute, and storage
isolation when tenants require stronger separation.

Start with [Models, Entrypoints, and Serving](models-entrypoints-serving) for
the end-to-end CLI workflow, or continue to [Recipes](recipes) for the policy
owned by an entrypoint.
