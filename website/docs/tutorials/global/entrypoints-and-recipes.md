---
title: Virtual Models
description: Give clients stable virtual model names backed by isolated routing policies in one Semantic Router deployment.
---

# Virtual Models

## Overview

Entrypoints and recipes turn one Semantic Router deployment into a set of
purpose-built virtual models:

- an **entrypoint** is the model name a client requests;
- a **recipe** is the routing policy that handles requests for that name; and
- providers, model endpoints, and shared services remain available to every
  recipe.

## What Problem Does It Solve?

This separation lets an agent harness choose an objective such as low latency,
high quality, or a balanced trade-off without knowing which backend model will
serve the request.

In canonical YAML, `entrypoints` holds the public-name mappings and `recipes`
holds the named routing policies.

## How the pieces fit

```text
request model name -> entrypoint -> recipe -> decision -> algorithm -> backend
```

When the request `model` matches an `entrypoints[].model_names` value, the
Router evaluates only the mapped recipe. The virtual model name is then
replaced by the backend selected from that recipe.

- **One recipe** is published as `vllm-sr/auto` and needs no entrypoints.
- **Several recipes** need exactly one chat entrypoint marked
  `default: true`; its recipe is the default recipe.
- **Names are explicit.** An entrypoint for the default recipe replaces the
  built-in name, so include `vllm-sr/auto` in its `model_names` if clients
  should keep using it. Names such as `auto` or `vllm-sr/flow` work only when
  declared.
- **No match** sends the request to `providers.defaults.model`.

The selected recipe's decisions choose its algorithms.

Concrete backend model names are different: they select that model directly
and bypass recipe routing. Use a virtual entrypoint when clients should ask for
an objective, and a concrete model name only when they intentionally need that
exact backend.

## Configuration

The model catalog is shared. Each recipe owns its signals, projections,
decisions, strategy, algorithms, and route-local plugins.

```yaml
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
  - model_names: [vllm-sr/mom-v1-flash]
    recipe: flash
  - model_names: [vllm-sr/mom-v1-ultra]
    recipe: ultra
recipes:
  - name: default
  - name: flash
    description: Prefer the lowest-latency eligible backend.
    strategy: priority
    decisions:
      - name: fast-path
        description: Serve requests with the fast model.
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: fast-model
  - name: ultra
    description: Prefer the highest-quality eligible backend.
    strategy: priority
    decisions:
      - name: quality-path
        description: Serve requests with the accurate model.
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: accurate-model
providers:
  models:
    - name: fast-model
    - name: accurate-model
```

Shared strategy and fallback defaults live in `global.router`:

- **No parent recipe.** The default recipe is not a parent of the others.
  Each `recipes[]` entry fills missing strategy and fallback fields directly
  from the global defaults, and a decision can override its own fallback.
- **Sparse overrides.** `fallback: {enabled: false}` changes only that field,
  and keeps the shared timeout and retry settings.
- **Direct requests** use the global fallback policy. The runtime default
  strategy is `priority`.

Replay capture defaults are a `router_replay` plugin in `global.plugins`,
with storage in `global.services.router_replay`; a recipe's or decision's
`router_replay` plugin overrides them field by field.

Clients can discover entrypoint names through `/v1/models`. Routed responses
include `x-vsr-selected-recipe`, so operators can confirm which policy handled
a request without exposing the backend selection contract to the client.

For the connection, protocol, and session setup, see
[Connect an agent harness](../../installation/agent-harness).

## Limits for agent clients

`/v1/models` tells a client which virtual names exist and how each one
resolves. It does not report a context window, output limit, or capability for
them, and the model behind a name can change from one request to the next.
This is the entry for `vllm-sr/auto`:

```json
{
  "id": "vllm-sr/auto",
  "object": "model",
  "created": 1790323030,
  "owned_by": "vllm-semantic-router",
  "description": "Intelligent Router for Mixture-of-Models",
  "routing": {
    "resolution": "virtual",
    "selectable": true,
    "default_route": true,
    "recipe": "default"
  }
}
```

Coding agents and other clients that size a request before sending it need
these values in their own configuration. Any turn of a session can reach any
model the recipe can select, including `providers.defaults.model`, so
configure the client with the intersection of their model cards:

| Client setting | Value |
| --- | --- |
| Context window | The smallest `context_window_size` |
| Output limit | The smallest `max_output_tokens` |
| Tool calling | On only if every model declares `tools` |
| Image input | On only if every model declares `vision` or `image_input` |
| Reasoning settings | Sent only if every model declares `reasoning` |

For a recipe that selects among the three models below, configure a
32,768-token context window, an 8,192-token output limit, and tool calling.
Leave image input and reasoning settings off.

```yaml
recipes:
  - name: default
providers:
  models:
    - name: local-coder
      context_window_size: 32768
      max_output_tokens: 8192
      capabilities: [chat, tools]
    - name: reasoner
      context_window_size: 200000
      max_output_tokens: 64000
      capabilities: [chat, tools, reasoning]
    - name: vision-generalist
      context_window_size: 131072
      max_output_tokens: 16384
      capabilities: [chat, tools, vision]
```

By default, the Router skips a candidate whose declared context window is
smaller than the estimated input, or whose declared capabilities lack a
required input such as images:

- **Conservative estimate.** Structured data such as tool-call arguments
  counts one token per byte, and images, audio, video and files carry
  reserves of their own. A coding agent's long session can therefore be
  estimated above a window it fits.
- **Second count.** When the estimate exceeds every candidate's window, the
  Router counts the request without those reserves and keeps the candidates
  that fit. When none fits, the widest candidates stay: an estimate never
  rejects a request, and the backend checks the real length.
- **The Router's count.** A request whose bytes could exceed a window is
  counted with each model's
  [tokenizer](../../installation/model-configuration#check-prompts-against-the-context-window).
  A request whose count is beyond the band of every window it can reach (3%
  with a model's own tokenizer, 30% with the proxy) is rejected with `400`
  `context_length_exceeded` before any signal runs; one within the band goes
  on, and the backend decides.
- **Output limits are not checked** by default, so a request for 16,384 output
  tokens can still reach `local-coder`.

With [`candidate_requirements`](../../installation/configuration#recipe-candidate-requirements)
on the recipe, the Router also checks output limits and tool, reasoning and
structured-output declarations before scoring. A request that fits none of the
candidates is rejected before dispatch; see
[Request budget errors](../../api/router#request-budget-errors).

:::tip
A client configured with the intersection keeps every candidate available to
every request.
:::

## When to Use

Use entrypoints and several recipes when one deployment must expose more than
one routing objective, policy boundary, or rollout track. Keep a single recipe
when all clients should follow the same policy; its `vllm-sr/auto` entrypoint
needs no extra configuration.

Continue with:

- [Entrypoints](entrypoints) for naming, request resolution, discovery, and
  validation rules.
- [Recipes](recipes) for policy isolation, shared infrastructure, lifecycle
  APIs, and limitations.
- [Models, Entrypoints, and Serving](models-entrypoints-serving) for the
  end-to-end catalog, CLI, backend binding, serving, and operations workflow.
