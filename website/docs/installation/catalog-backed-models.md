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
choose a Provider in **Build → Routing → Models → Add Model**. The Dashboard labels
catalog choices as **Built-in** and saves both the local alias and canonical
identity.

## MiniMax M3.1 Flash

`minimax/minimax-m3.1-flash` identifies the
[official open weights](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)
under the MiniMax Community License. The MiniMax API and an experimental vLLM
mapping share that card. Both mappings currently declare Chat Completions.

The card records the checkpoint's 1,048,576-token context window from its
[pinned configuration](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/resolve/76faee1b4ac0f412e547781172e94548a6526a88/config.json).
The MiniMax API mapping separately records its 1,000,000-token service limit.

```yaml
version: v0.3
providers:
  models:
    - name: flash
      catalog: minimax/minimax-m3.1-flash
      backend_refs:
        - name: primary
          provider: minimax
          api_key_env: MINIMAX_API_KEY
routing:
  decisions:
    - name: flash_default
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: flash
          use_reasoning: true
          reasoning_effort: high
```

The model uses adaptive thinking and supports `low`, `medium`, `high`, `xhigh`,
and `max`, with `max` as the catalog default. Disabling reasoning, selecting
`reasoning_mode: enabled`, or requesting `reasoning_effort: none` is rejected.
The Router sends the selected effort as a top-level field to MiniMax.
The [API](https://platform.minimax.io/docs/api-reference/text-openai-api)
supports a 1,000,000-token context and up to 524,288 output tokens, including
thinking. It ignores sampling changes to `temperature` and `top_p`, using 1
and 0.95; `service_tier` is unsupported. Preserve `reasoning_content` and
tool-call IDs when returning assistant history in a tool conversation.

For an existing self-hosted server, replace the backend above with:

```yaml
        - name: primary
          provider: vllm
          base_url: http://127.0.0.1:8000/v1
```

The mapping resolves `MiniMaxAI/MiniMax-M3.1-Flash` and sends effort inside
`chat_template_kwargs`. Follow the
[vLLM recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3.1-Flash): its reviewed
version requires vLLM 0.33.0 nightly with the dedicated `minimax-m31-flash`
image, `--block-size 128`, and `minimax_m3` reasoning/tool parsers. A stable pip
installation is not sufficient. The Hopper example limits context to 262,144
and disables video; match your served window and capabilities with canonical
card overrides as described below. The mapping is experimental and does not
certify a particular deployment.

Five published benchmark records are attributed to the pinned official model
card. Their reasoning effort is `unspecified`; they are vendor claims, not
vLLM-SR reproductions. Terminal-Bench 4.0 remains separate from 2.1, and the
selectable effort levels have no complete Intelligence 1.0 evidence.

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

## Match the window a self-hosted backend serves

A built-in card declares the model's full context window. A vLLM server
started with a smaller `--max-model-len` rejects longer requests, but the
Router still uses the card's window to decide which candidates fit a request
and how much input context compression may keep. To align them, start vLLM
with a matching `--max-model-len`, or set the served window on the canonical
card:

```yaml
routing:
  modelCards:
    - name: qwen/qwen3.6-27b
      context_window_size: 32768
```

For example, vLLM 0.11.1 derives a 32,768-token limit from the pinned
[Hunyuan 7B Instruct config](https://huggingface.co/tencent/Hunyuan-7B-Instruct/blob/6fd6ecb05e76589bc43b79f49e3619445c6b4593/config.json)
and rejects `max_model_len=262144`, although the model card and tokenizer
advertise 256K. For that deployment, override the canonical card window:

```yaml
version: v0.3
providers:
  defaults:
    model: local-hunyuan
  models:
    - name: local-hunyuan
      catalog: tencent/hunyuan-7b-instruct
      backend_refs:
        - provider: vllm
          base_url: http://127.0.0.1:8000/v1
routing:
  modelCards:
    - name: tencent/hunyuan-7b-instruct
      context_window_size: 32768
```

The override uses the canonical card ID, not `local-hunyuan`. Keep the published
card unchanged, and revalidate the deployment limit after changing the serving
runtime or model revision.

When a configuration loads or reloads, the Router reads `max_model_len` from
each `vllm` backend's `/v1/models` response. If a backend serves less than the
card declares, the Router logs a `served_context_window_below_model_card`
warning with both values. It keeps routing with the card's window until you
change the configuration or the deployment.

## Override the protocol only when necessary

The selected Provider mapping normally supplies the correct protocol and
request path. Set `api_format: openai`, `responses`, or `anthropic` only when
the backend intentionally exposes another compatible wire format. The field
does not choose a Provider and does not replace `backend_refs[].provider`.

Provider-specific reasoning modes and effort restrictions are validated
against the effective Model and Provider mapping. See
[Reasoning configuration](model-reasoning) before adding decision-level
controls.
