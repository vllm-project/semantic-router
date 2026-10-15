---
title: Catalog-backed Models
description: Bind a built-in Model Card to a supported Provider without duplicating model metadata or reasoning rules.
---

# Catalog-backed models

Use a catalog-backed Model when the built-in catalog already describes the
model. The catalog owns the canonical Model Card and reasoning family. A
Provider mapping adds the upstream model ID, supported protocols, reasoning
transport, pricing, and any restrictions known for that Provider.

## Configure the smallest binding

```yaml
version: v0.3

providers:
  defaults:
    model: production
  models:
    - name: production
      catalog: openai/gpt-5.6-sol
      backend_refs:
        - name: openai-primary
          provider: openai
          api_key_env: OPENAI_API_KEY

routing: {}
```

`production` is the local alias. The Router resolves the canonical card and the
OpenAI-specific mapping from `catalog: openai/gpt-5.6-sol`; do not repeat its
reasoning definition in YAML.

Browse built-in identities and their Provider support in **Model Hub**, or
choose a Provider in **Build → Models → Add Model**. The Dashboard labels
catalog choices as **Built-in** and saves both the local alias and canonical
identity.

## Supply a deployment name when required

Some Providers map a catalog model to a fixed upstream ID. Others, such as a
managed deployment service, require the name that you assigned while
deploying the model. In that case set `provider_model_id`:

```yaml
providers:
  models:
    - name: team-reasoner
      catalog: microsoft/mai-thinking-1
      provider_model_id: my-production-deployment
      backend_refs:
        - provider: microsoft-foundry
          base_url: https://example.services.ai.azure.com
          api_key_env: FOUNDRY_API_KEY
```

The Dashboard shows **Deployment name** or **Provider model ID** only when the
selected mapping requires one. It is an upstream identifier, not a second
catalog identity.

## Override only operator-owned metadata

You can add local tags, descriptions, capabilities, or evaluations without
forking the built-in card. The override must use the canonical `catalog` ID,
not the local alias:

```yaml
routing:
  modelCards:
    - name: openai/gpt-5.6-sol
      tags: [production, approved]
      description: Approved for production reasoning traffic.
```

Do not set `providers.models[].reasoning` on a catalog-backed Model. That
combination is rejected so an alias cannot silently change a repository-owned
reasoning contract.

## Override the protocol only when necessary

The selected Provider mapping normally supplies the correct protocol and
request path. Set `api_format: openai`, `responses`, or `anthropic` only when
the backend intentionally exposes another compatible wire format. The field
does not choose a Provider and does not replace `backend_refs[].provider`.

Provider-specific reasoning modes and effort restrictions are validated
against the effective Model and Provider mapping. See
[Reasoning configuration](model-reasoning) before adding decision-level
controls.
