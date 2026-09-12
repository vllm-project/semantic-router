---
title: Configure Models
description: Choose a catalog-backed or custom model, bind it to providers, and use it from routing decisions.
---

# Configure models

A configured Model joins a stable Router name to one model identity and one or
more physical backends. Configure Models under `providers.models`, then refer to
their `name` from `providers.defaults.model` and
`routing.decisions[].modelRefs[].model`.

## Understand the model identifiers

The fields answer different questions and are not interchangeable:

| Field | Meaning |
| --- | --- |
| `name` | Router-local alias used by defaults, decisions, and direct model calls. |
| `catalog` | Optional canonical Model Card ID from the built-in catalog. Omit it for a private or newly released model. |
| `provider_model_id` | Optional model or deployment name sent to the upstream provider. |
| `backend_refs[].provider` | Stable Provider contract that supplies endpoint, authentication, path, protocol, and model-mapping defaults. |
| `api_format` | Optional wire-protocol override: `openai`, `responses`, or `anthropic`. It never selects a Provider. |

For example, `production-reasoner` can be your Router alias,
`openai/gpt-5.6-sol` its catalog identity, and a provider-specific deployment
name its `provider_model_id`.

## Choose a configuration path

| Starting point | Configure | Reasoning behavior |
| --- | --- | --- |
| Model exists in the catalog | Set `catalog` and a Provider on each backend. | Inherited from the catalog and selected Provider mapping. |
| Private or unlisted model with a known built-in family | Omit `catalog`; set `reasoning.family`. | Reuses the built-in family contract. |
| Private or unlisted model with unique controls | Omit `catalog`; define `reasoning` inline. | Uses the model-local contract you define. |
| Custom pass-through model | Omit both `catalog` and `reasoning`. | Router does not synthesize reasoning controls. |

See [Catalog-backed models](catalog-backed-models),
[Custom models](custom-models), and [Reasoning configuration](model-reasoning)
for each path. [Model configuration patterns](model-configuration-patterns)
shows the supported combinations and invalid mixtures.

## Start with one custom model

```yaml
version: v0.3

providers:
  defaults:
    model: local-chat
  models:
    - name: local-chat
      provider_model_id: served-model
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: host.docker.internal:8000
          protocol: http

routing: {}
```

The Router creates a sparse local Model Card for `local-chat`. Add a matching
`routing.modelCards` entry only when routing needs metadata such as context
window, capabilities, tags, or LoRAs. Add benchmark measurements independently
under top-level `evaluation.records[]`.

## Configure Models in the Dashboard

Open **Build → Models → Add Model**. You can then:

1. Choose a Provider, connect it, and select discovered or built-in model IDs.
2. Enter a model ID when the Provider cannot list models.
3. Use **Advanced settings** for a name prefix, a built-in reasoning family,
   routing metadata, pricing, and delivery settings.
4. Use **Manual setup** from the Provider chooser for complete control over
   `catalog`, inline reasoning fields, `provider_model_id`, `api_format`, and
   backend references.

The Dashboard edits the same v0.3 document as YAML. Use one authoring owner for
a deployment and review the exported YAML before serving it. See
[Configuration workflows](configuration-workflows).

## Validate the result

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

Validation resolves catalog entries, Provider mappings, reasoning controls,
Model Card identities, and backend-pool compatibility before startup.

## Configure models used by Router tasks

Learned Router tasks use `global.model_catalog.deployments` to declare local
resources or remote services. A recipe selects them through
`routing.model_bindings`; named recipes use the same field inside their own
`routing` block. Existing system and module defaults continue to work when
an explicit binding is absent.

```yaml
global:
  model_catalog:
    deployments:
      local-pii:
        artifact: models/mmbert32k-pii-detector-merged
        provider: candle
        device: cpu
        precision: native
        input:
          max_tokens: 512
          overflow: reject
    admission:
      local-pii:
        max_concurrency: 2
        max_queue: 8
        queue_timeout_ms: 1000
        on_overflow: shed
routing:
  model_bindings:
    pii_classifier:
      deployment: local-pii
      contract: token_spans.v1
      adapter: mmbert32k
      mapping_path: models/mmbert32k-pii-detector-merged/pii_type_mapping.json
```

The artifact must already be available to the Router. `revision` records its
revision; provider preparation validates the files, adapter, task and device.
Local deployments select `candle` or `ort`. An `http` deployment instead sets
`external_model` to a named entry in `global.model_catalog.external`; its
device and precision belong to the remote service.

The deployment describes execution and capacity. The binding describes the
result contract and its interpretation through `adapter`, optional `head`, and
`mapping_path`. Reusing an artifact path alone does not establish that two
bindings can share a physical resource. Shared resources enforce one total
admission budget, and a recipe cannot look up another recipe's binding.

`input.max_tokens` restricts the provider's actual task limit. The maintained
mmBERT classification and token tasks have a **512-token limit**, including
task preprocessing. The `mmbert32k` name or a longer embedding context does
not increase that limit. `overflow` requests `reject`, `truncate`, or `window`;
the adapter must support the requested policy and report any truncation.
Unsupported combinations fail preparation instead of silently changing the
provider or input policy.

Use the [generated configuration contract](configuration-contract) to inspect
the exact fields for the running Router. The Dashboard preserves the same
fields when editing individual recipes. In DSL, put `model_bindings` inside
the recipe's `ROUTING` block; deployment declarations remain in global YAML.
Helm `configOverride` accepts the complete canonical document. The Operator
maps `spec.config.model_deployments` and `spec.config.model_admission` to the
catalog and carries bindings in `spec.config.routing.model_bindings`.

See [Native backends](native-backends) for ownership and build boundaries, and
[Safety models and policy](../tutorials/global/safety-models-and-policy) for
named remote prompt guard backends and migration.
