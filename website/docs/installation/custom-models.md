---
title: Custom Models
description: Configure private, self-hosted, or newly released models with optional metadata and homogeneous backend replicas.
---

# Custom models

A custom Model is any model whose canonical identity is owned by your
deployment rather than the built-in catalog. This includes private checkpoints,
fine-tunes, newly released models, and self-hosted aliases. Omit `catalog` to
keep the Model custom.

## Bind a custom model

```yaml
version: v0.3

providers:
  defaults:
    model: private-chat
  models:
    - name: private-chat
      provider_model_id: private-chat-awq
      api_format: openai
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: model-gateway.example:8000
          protocol: http

routing:
  modelCards:
    - name: private-chat
      display_name: Private Chat
      context_window_size: 131072
      capabilities: [chat, tools]
      tags: [private, production]
```

For a custom Model, `routing.modelCards[].name` must equal the Model alias. The
card is optional: omit it when the Router only needs a backend binding. If
`provider_model_id` is omitted, the upstream ID defaults to the alias; set it
explicitly whenever the served checkpoint uses another name.

Model Card metadata can also include publisher, presentation, distribution,
LoRAs, and operator evaluations. It must not contain credentials. Put prices
under `providers.models[].pricing` and credentials under the backend binding,
preferably through `api_key_env`.

## Add evaluation evidence

Attach a result for a built-in benchmark directly to the Model Card:

```yaml
routing:
  modelCards:
    - name: private-chat
      evaluations:
        - benchmark: livecodebench/livecodebench@6.0.0
          benchmark_profile: independent-code-generation
          reasoning_effort: high
          metrics: {pass_at_1: 0.61}
          measured_at: 2026-09-09
          source: https://benchmarks.example/runs/private-chat-lcb6
```

The result enters every compatible built-in index at the exact reasoning effort.
For an organization-specific benchmark, first declare its metric and any index
under `evaluation_catalog`. See
[Custom evaluation catalogs](../benchmarking/custom-evaluation-catalog).

## Add custom reasoning

Reasoning is optional and has two supported forms:

```yaml
# Reuse one built-in family.
reasoning:
  family: qwen3
```

```yaml
# Define a model-local family when no built-in family matches.
reasoning:
  type: reasoning_effort
  parameter: reasoning_effort
  levels: [low, medium, high]
  default: medium
  modes: [enabled, disabled]
  default_mode: enabled
  disabled: none
```

Choose exactly one form. The previous global custom-family registry is not part
of canonical v0.3; inline definitions live on the custom Model that uses them.
See [Reasoning configuration](model-reasoning) for field semantics and
decision controls.

## Use replicas only for the same upstream contract

Multiple `backend_refs` under one Model form a load-balanced replica pool:

```yaml
providers:
  models:
    - name: private-chat
      provider_model_id: private-chat-awq
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: model-a.example:8000
          protocol: http
          weight: 80
        - name: secondary
          provider: vllm
          endpoint: model-b.example:8000
          protocol: http
          weight: 20
```

The replicas must keep one Provider, wire protocol, native model ID,
credentials, headers, effective request path, and TLS behavior. Hosts, ports,
and weights may vary within the supported transport rules. Create separate
Model aliases when Providers or request semantics differ, then let a routing
decision choose between those aliases.

## Use the Dashboard

Open **Build → Models → Add Model**:

- choose a Provider and enter the custom model ID in the connection flow; or
- choose **Manual setup** to edit the complete identity, reasoning, Model Card,
  pricing, reliability, and backend-reference fields.

Keep **Built-in Catalog Model** empty. **Reasoning Family** selects only a
built-in family; use the **Inline Reasoning** fields when your model needs a
new contract.
