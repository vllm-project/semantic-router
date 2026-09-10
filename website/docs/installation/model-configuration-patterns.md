---
title: Model Configuration Patterns
description: Compare supported catalog, custom, reasoning, provider, and replica combinations before writing YAML.
---

# Model configuration patterns

Use this matrix to choose the smallest valid Model declaration. All examples
are entries under `providers.models`.

## Supported combinations

| Goal | `catalog` | `reasoning` | `provider_model_id` | Backend |
| --- | --- | --- | --- | --- |
| Use a catalog model through a Provider that lists it | Canonical Model Card ID | Omit; inherited | Usually omit; Provider support supplies it | Set the Provider ID. |
| Use a catalog model through an operator-named deployment | Canonical Model Card ID | Omit; inherited | Set deployment name | Set the deployment Provider. |
| Use a private model without Router-generated reasoning controls | Omit | Omit | Set when different from `name` | Set the Provider and endpoint. |
| Use a private model with a known built-in family | Omit | `family: <built-in-id>` | Optional | Set a compatible Provider and endpoint. |
| Use a private model with unique reasoning controls | Omit | Complete inline definition | Optional | Set a compatible Provider and endpoint. |
| Add routing metadata to a private model | Omit | Optional | Optional | Add `routing.modelCards` under the alias. |
| Add local metadata to a catalog model | Canonical Model Card ID | Omit; inherited | Provider-dependent | Add `routing.modelCards` under the catalog ID. |
| Load balance identical replicas | Either | Determined by model identity | One effective ID | Put homogeneous targets in one `backend_refs` list. |
| Route across different Providers or protocols | Configure one Model per contract | Determined per Model | Determined per Provider | Use separate aliases in the same decision. |

## Catalog model with an automatic mapping

```yaml
- name: production
  catalog: openai/gpt-5.6-sol
  backend_refs:
    - provider: openai
      api_key_env: OPENAI_API_KEY
```

## Catalog model with an operator deployment name

```yaml
- name: team-reasoner
  catalog: microsoft/mai-thinking-1
  provider_model_id: my-production-deployment
  backend_refs:
    - provider: microsoft-foundry
      base_url: https://example.services.ai.azure.com
      api_key_env: FOUNDRY_API_KEY
```

## Custom pass-through model

```yaml
- name: local-chat
  provider_model_id: served-chat
  api_format: openai
  backend_refs:
    - provider: vllm
      endpoint: host.docker.internal:8000
      protocol: http
```

## Custom model with a built-in family

```yaml
- name: local-qwen
  provider_model_id: served-qwen
  reasoning:
    family: qwen3
  backend_refs:
    - provider: vllm
      endpoint: host.docker.internal:8000
      protocol: http
```

## Custom model with inline reasoning

```yaml
- name: private-reasoner
  reasoning:
    type: reasoning_effort
    parameter: reasoning_effort
    levels: [low, medium, high]
    default: medium
    modes: [enabled, disabled]
    default_mode: enabled
    disabled: none
  backend_refs:
    - provider: vllm
      endpoint: host.docker.internal:8000
      protocol: http
```

## Combinations that validation rejects

- `catalog` together with any `providers.models[].reasoning` block;
- `reasoning.family` together with inline reasoning fields;
- an inline definition without both `type` and `parameter`;
- a Model Card override named after a catalog-backed alias instead of its
  canonical `catalog` ID;
- a custom Model Card whose name differs from the custom Model alias;
- a decision effort or mode unsupported by the Model family or Provider
  mapping; and
- heterogeneous Providers, protocols, credentials, native IDs, paths, or TLS
  semantics inside one replica pool.

Run `vllm-sr validate --config config.yaml` after composing a pattern. See
[Configure models](model-configuration) for the identity model and
[Reasoning configuration](model-reasoning) for the complete reasoning fields.
