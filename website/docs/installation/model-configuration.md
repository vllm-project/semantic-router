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

## Check task capabilities before dispatch

The selected model must satisfy both the provider protocol's capabilities and
its declared task capabilities. The same checks apply when the Router selects
a fallback from the matched decision. A fallback cannot reintroduce a model
that was excluded by the request's context-window check.

Model Card capability metadata accepts protocol names such as `image_input`
and `image_generation`. The catalog aliases `vision`, `audio`, and `video`
describe image, audio, and video **input**, respectively; they do not grant
media-generation support. `structured_output` maps to `structured_json`, and
`tool_use` maps to `tools`. Descriptive labels such as `long_context` or `coding`
do not erase recognized capability declarations alongside them.

A model without any recognized capability declaration retains protocol-only
compatibility. For an annotated model, a protocol that can encode a request
does not override missing task capabilities. When no eligible decision model
can serve the requested task, the Router returns `unsupported_capability`.

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
and a declaration reaches capability-aware dispatch in one of these states:

1. **No recognized declaration** — the model is unannotated and stays eligible
   on wire expressibility alone.
2. **Task/modality declaration** — names like `image_input`, `image_output`,
   `image_generation`, `audio_input`, `audio_output`, `video_input`,
   `video_output`, `file_input`, and `file_output` steer the declared-task
   filter: an annotated model must declare every task bit the request requires,
   otherwise the dispatch is rejected with `unsupported_capability`.
3. **Transport/accounting declaration** — names like `tools`, `reasoning`,
   `streaming`, and `structured_json` are recognized, yet they carry no task
   bit. Such a model is annotated, so it does not qualify for a media task it
   never declared; text requests require no task bit and are unaffected.
4. **Catalog aliases** — `vision`, `audio`, and `video` project onto
   `image_input`, `audio_input`, and `video_input`; `structured_output` maps to
   `structured_json` and `tool_use` maps to `tools`. Projection runs before the
   filter, so a card declaring `[image_input, vision]` is filtered on
   `image_input` alone: it stays eligible for image requests and is rejected
   for audio, which it never declared.
5. **Descriptive labels** — names outside the protocol vocabulary and its
   aliases (e.g. `long_context`, `coding`) are metadata: they contribute no
   task bit and do not void recognized names in the same declaration, so a
   declaration of descriptive labels alone is treated as unannotated.

The primary dispatch and the fallback candidates are judged by the same
qualification, so a fallback cannot reintroduce eligibility that the selected
model was denied.

## Validate the result

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

Validation resolves catalog entries, Provider mappings, reasoning controls,
Model Card identities, and backend-pool compatibility before startup.

## Configure models used by Router tasks

For classifiers, safety checks, and embeddings used inside the Router, start
with [Router Runtime](native-backends). It covers in-process and external
models, their configuration, and operations.
