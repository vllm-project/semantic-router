---
title: Entrypoints
description: Expose stable virtual model names that select a routing recipe through the model field of supported inference APIs.
---

# Entrypoints

## Overview

An entrypoint is a public virtual model name that maps to one recipe. Clients
select it through the `model` field in the supported Chat Completions, Responses,
or Messages API, so they do not need a Router-specific API or header to select
the recipe. See [Connect an agent harness](../../installation/agent-harness)
for the client connection and session setup.

## What Problem Does It Solve?

Keep a stable name such as `vllm-sr/mom-v1-flash` in the harness while changing
the models, thresholds, or algorithms behind it.

## When to Use

Create an entrypoint when you want to:

- publish latency, quality, cost, safety, or team-specific routing objectives;
- move a client between policy versions without exposing backend model IDs; or
- run several isolated policies in one Router deployment.

Use `vllm-sr/auto` or an explicitly declared default entrypoint when every routed
request should use the default policy. Use a concrete provider model name only
when the caller deliberately wants to bypass signals, decisions, algorithms,
and route-local plugins.

## Configuration

Each entrypoint lists one or more aliases and the named recipe they select:

```yaml
entrypoints:
  - model_names:
      - vllm-sr/mom-v1-flash
      - company/fast
    recipe: flash

recipes:
  - name: flash
    description: Low-latency routing for interactive requests.
    routing:
      strategy: priority
      decisions: []
```

Both aliases select the same recipe. A client uses either name like any other
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

## Request resolution

| Requested model | Router behavior |
| --- | --- |
| An `entrypoints[].model_names` value | Evaluate only the mapped recipe. |
| `vllm-sr/auto`, when no entrypoint explicitly targets `default` | Evaluate the `default` recipe from top-level `routing`. |
| An explicitly declared ReMoM, Fusion, or Flow entrypoint | Evaluate its recipe; the matched decision selects the looper algorithm. |
| A concrete provider model or LoRA name | Send directly to that backend without recipe routing. |

Entrypoints are listed by `/v1/models` with routing metadata. Successful routed
responses expose `x-vsr-selected-recipe`; Router Replay and Insights can also
filter records by recipe.

To rename the default entrypoint, declare `recipe: default` with your desired
`model_names`. This replaces the built-in `vllm-sr/auto` name, so include that
name explicitly if existing clients still need it. Bare `auto` and looper
names have no implicit behavior.

For example, to publish both the namespaced default and an older client's
`auto` name:

```yaml
entrypoints:
  - model_names: [vllm-sr/auto, auto]
    recipe: default
```

## Naming and validation rules

Configuration loading rejects an entrypoint when:

- `model_names` is empty or `recipe` names no configured recipe;
- the same virtual name is claimed by more than one entrypoint; or
- a virtual name collides with a provider model, LoRA, or another effective
  entrypoint, including the built-in default name.

Choose names that describe a durable client contract, not the current backend.
Do not put tenant data or secrets in a name: entrypoints appear in model
discovery, response metadata, metrics, and operational records.

An entrypoint is a policy selector, not a security boundary. Recipes share the
Router process and configured infrastructure; use network, compute, and storage
isolation when tenants require stronger separation.

Start with [Models, Entrypoints, and Serving](models-entrypoints-serving) for
the end-to-end CLI workflow, or continue to [Recipes](recipes) for the policy
owned by an entrypoint.
