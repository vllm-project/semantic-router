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
window, capabilities, tags, LoRAs, or operator evaluations.

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

## Routing capabilities

Author-declared model capabilities drive capability-aware dispatch: a request
whose required capability the originally selected backend cannot express is
rerouted to the first qualified model in the decision whose declared
capabilities cover it. Declare them on a model card:

```yaml
routing:
  modelCards:
    - name: image-backend
      capabilities: [image_generation]
```

The protocol capability vocabulary is defined by the `llmprotocol` package,
and only task/modality capabilities participate in declaration steering:
`image_input`, `image_output`, `image_generation`, `audio_input`,
`audio_output`, `video_input`, `video_output`, `file_input`, `file_output`.
A declared name outside this set (including transport or accounting fidelity
like `tools`, `reasoning`, `streaming`, `structured_json`) is treated as
unannotated for declaration filtering — those are verified against wire codec
expressibility, not the model declaration.

## Validate the result

```bash
vllm-sr validate --config config.yaml
vllm-sr serve --config config.yaml
```

Validation resolves catalog entries, Provider mappings, reasoning controls,
Model Card identities, and backend-pool compatibility before startup.
