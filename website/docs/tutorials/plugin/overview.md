# Plugins

## Overview

Plugins add route-local behavior after a decision matches. They can rewrite a
request, retrieve context, short-circuit generation, inspect a response, or
control what operational data is retained.

Shared services and stores belong under `global:`; the decision plugin only
enables and tunes that behavior for one route.

## Key Advantages

- Keeps behavior attached to the route that needs it.
- Reuses shared stores and services without duplicating their configuration.
- Makes request mutation, retrieval, and response inspection auditable.

## What Problem Does It Solve?

Routes often need different behavior even when they share the same Router.
Plugins keep those differences next to the decision instead of hiding them in
application middleware or enabling them globally.

## When to Use

Use a plugin when behavior should apply only after a specific route matches.
Use `global:` instead when every route shares the same service or backing
store. Plugin entries live under `routing.decisions[].plugins`.

## Configuration

```yaml
routing:
  decisions:
    - name: cached-support
      description: Reuse cached responses for support requests.
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: support-model
      plugins:
        - type: response_cache
          configuration:
            enabled: true
            ttl_seconds: 3600
```

The maintained examples live under
[`config/fragments/plugin/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin).

## Plugin Inventory

| Type | Goal | Shared dependency | Guide |
|---|---|---|---|
| `fast_response` | Return a configured response without calling a model | None | [Fast Response](./fast-response) |
| `system_prompt` | Insert, replace, or append route-specific instructions | None | [System Prompt](./system-prompt) |
| `header_mutation` | Add, update, or delete downstream headers | None | [Header Mutation](./header-mutation) |
| `request_params` | Enforce request parameter limits | None | [Request Parameters](./request-params) |
| `tools` | Allow, block, filter, or remove tools and tool history | Optional global tool catalog | [Tools](./tools) |
| `tool_selection` | Add tools from a catalog or filter caller tools semantically | Embedding runtime; tool database for `add` mode | [Tool Selection](./tool-selection) |
| `context_compression` | Reduce large provider-bound tool output or history | Optional embedding runtime and recovery store | [Context Compression](./context-compression) |
| `response_cache` | Reuse compatible prior responses | `global.stores.response_cache` | [Response Cache](./response-cache) |
| `memory` | Retrieve and optionally store conversational memory | `global.stores.memory` | [Memory](./memory) |
| `rag` | Retrieve documents before generation | Configured RAG/vector backend | [RAG](./rag) |
| `router_replay` | Override replay capture for one route | `global.services.router_replay` | [Router Replay](./router-replay) |
| `hallucination` | Inspect factual support in a response | Hallucination/NLI modules as configured | [Hallucination](./hallucination) |
| `response_jailbreak` | Screen a generated response for jailbreak content | Prompt-guard runtime | [Response Jailbreak](./response-jailbreak) |
| `image_gen` | Send a matched route to an image-generation backend | Configured image backend | [Image Generation](./image-gen) |

[Content Safety](./content-safety) is a maintained bundle of three supported
plugins, not an additional plugin type.

## Operational Boundaries

- Plugins can interact when more than one mutates the provider-bound request or
  response. The Router pipeline fixes their execution order; reordering entries
  in YAML does not change it.
- Retrieval, memory, cache, and replay may persist request-derived content.
  Configure retention, tenant/user scope, authentication, and encryption for
  the selected backend.
- Header and prompt mutation can cross trust boundaries. Do not copy untrusted
  caller metadata into privileged headers or system instructions.
- The supported type list is defined in
  [`routing_surface_catalog.go`](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/config/routing_surface_catalog.go).
