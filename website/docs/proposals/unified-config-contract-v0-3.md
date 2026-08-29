---
title: Unified Config Contract v0.3
description: Records the implemented configuration contract shared by the router, CLI, dashboard, Helm, operator, and DSL.
created: 2026-03-17
status: Implemented
---

> **Status:** Implemented · **Created:** 2026-03-17

## Problem

The router, CLI, dashboard, Helm chart, operator, and DSL previously interpreted
overlapping configuration shapes. A file accepted by one surface could require
translation or undocumented defaults in another. Model identity was also mixed with
deployment endpoints and credentials.

## Implemented contract

The public configuration has seven top-level sections:

```yaml
version:
listeners:
providers:
routing:
entrypoints:
recipes:
global:
```

| Section | Responsibility |
| --- | --- |
| `version` | Selects the configuration contract. |
| `listeners` | Defines request-facing and management listeners. |
| `providers` | Binds logical model names to provider identifiers and endpoints. |
| `routing` | Defines the default model cards, signals, projections, decisions, algorithms, and plugins. |
| `entrypoints` | Maps request-facing model names to the default profile or a named recipe. |
| `recipes` | Defines additional isolated routing profiles that share providers and global infrastructure. |
| `global` | Holds router-wide services, stores, integrations, model modules, and sparse runtime overrides. |

Unknown or retired shapes should fail with a clear validation error rather than be
silently translated at runtime.

## Provider and model boundary

`providers.defaults` owns the default provider behavior and default model.
`providers.models[].backend_refs[]` owns physical backend bindings.
`providers.models[].pricing` owns optional deployment cost metadata used by
cost-aware selection and accounting. Pricing does not belong to routing model cards.

`routing.modelCards` describes routing-facing model identity. Optional
`routing.modelCards[].loras` declare LoRA adapters that decisions may select with
`lora_name`. Signals and decisions reference logical model names, not endpoints or
credentials.

## Routing and DSL boundary

Routing owns:

- model cards;
- named signals and projections;
- decisions, candidate `modelRefs`, algorithms, and plugins;
- route-local output and adaptation policy.

Structured request controls remain facts at the signal boundary. For example,
conversation signals expose whether the protocol requires or forbids tool
execution, projections reconcile those facts with text-derived observations,
and decisions consume the resulting policy-facing output.

Top-level `entrypoints` select the default routing profile or a named item from
top-level `recipes`; they are not nested inside `routing`.

The DSL is an authoring view of routing semantics. It does not own provider
credentials, listeners, stores, or global runtime services. Import and export must
preserve the same canonical routing document rather than invent another steady-state
schema.

## Entrypoints and multi-recipe routing

`entrypoints[]` map request model names to either top-level routing or one named
recipe. `recipes[]` contain isolated routing profiles that reuse the same provider
inventory and global runtime.

This keeps the public API stable while allowing several routing policies to coexist in
one process. An entrypoint resolves the recipe before signals and decisions run.

## Defaults and configuration source

Built-in defaults live in the router. `global.router.config_source` selects file-backed
configuration or Kubernetes CRD reconciliation. External templates must not apply
hidden defaults after validation.

The dashboard, Helm chart, and operator may help users author or transport config, but
the resulting document still uses the same contract.

## Repository sources

`config/config.yaml` is the exhaustive canonical reference config. Reusable examples
live under:

- `config/fragments/signal/`;
- `config/fragments/decision/`;
- `config/fragments/algorithm/`; and
- `config/fragments/plugin/`.

Runtime deployment examples remain separate from routing fragments. Contract tests
and `make agent-lint` keep the reference config, schema, examples, and public docs
aligned.

## Migration

Use `vllm-sr config migrate --config old-config.yaml` to convert supported legacy
layouts. Review the result, resolve credentials through the deployment's secret
mechanism, and validate it before serving.

`vllm-sr init` was removed. Canonical YAML is the steady-state configuration source;
interactive or graphical authoring tools must export that same document.

## Scope and non-goals

The contract unifies configuration ownership. It does not require every authoring
surface to expose every advanced field in one form, nor does it make the DSL a
deployment-language replacement.

## References

- [Current configuration guide](../installation/configuration)
- [Configuration workflows](../installation/configuration-workflows)
- [Signals, decisions, and model selection](../overview/signal-driven-decisions)
- [Virtual Models](../tutorials/global/entrypoints-and-recipes)
- [Related issue #1505](https://github.com/vllm-project/semantic-router/issues/1505)
