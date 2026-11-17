---
title: Migrate model resources
sidebar_label: Migrate model resources
description: Each earlier way to choose a model for a signal, next to the deployment-and-binding form that replaces it.
---

# Migrate model resources

Earlier releases chose models with `external[]` models, binding objects,
per-rule binding keys and a request backend's own connection fields. The
Router refuses each of these forms at load with a one-line error that names
`vllm-sr config migrate`, which rewrites it into the second form of its pair.
This page is the only place the earlier forms appear; the rest of the
documentation uses the second form of each pair.
[Choose a model for a signal](choose-a-model-for-a-signal.md) explains the
form they move to, and the
[release note](../release-notes/unified-model-resources.md) summarizes the
change.

See the refusals with `vllm-sr config validate --config config.yaml`, or with
the Router itself: `router -validate-config -config config.yaml` prints one
JSON line whose `error` holds the refusals, such as
`{"valid": false, "error": "<path>: removed_field: <what>; run vllm-sr config migrate (<becomes>)"}`.
Each line names the field and what replaces it.

`vllm-sr config migrate --config config.yaml` asks the Router to rewrite the
file: each form on this page, and the other forms this release removed, in
the form that replaces it; the rest of the file stays as written. It asks the
Router binary `VLLM_SR_ROUTER_BINARY` names, else the Router image
`vllm-sr serve` runs by default (the one `VLLM_SR_ROUTER_IMAGE` or
`VLLM_SR_IMAGE` names, else this CLI's own release), which must be present:
migrate never pulls it and never asks a running stack's Router. An
`external[]` model's inline key moves to an environment variable,
and the output names the variable to set. A binding that asked a classifier
head (a `label_distribution.v1`, `label_scores.v1`, `label_decision.v1`,
`score.v1` or `token_spans.v1` contract) is dropped, and so is a rule's or a
binding's reference to a `classify` or `http` deployment: no signal asks a
classifier any more, and the decision model answers the signal. A file in the
layout before config v0.3 migrates in the same run.

## What else changes

- A `systemone` decision model now answers the built-in signals too. Before,
  they ran on a local `vela3-fast`. Their thresholds are 0.5 unless the
  configuration writes them, and the load warns `uncalibrated_preset`.
- The signals and modules that only the removed learned classifiers used
  (classifier, embedding, knowledge-base and preference rules, the domain,
  fact-check, feedback and modality modules, safety hazard parts) no longer
  load. `vllm-sr config migrate` rewrites them into the application signals
  the decision model answers, or names what to write instead.
- The prompt guard, PII and hallucination detector modules no longer load
  either, nor do their system lines, admission keys, module backends and
  `external[]` guard fields. The decision model answers jailbreak, pii and
  hallucination rules; a module's threshold becomes the predicate of the
  rules that read it, and a binding of another model is keyed by the signal
  type.
- Embedding model keywords (`mmbert`, `qwen3`, `multimodal`),
  `embeddings.semantic` with its remote endpoint, embedding bindings and
  admission by model keyword no longer load. `vllm-sr config migrate` writes
  the deployment each one chose and names it where it is used; see
  [Embeddings](../model-runtime/guides/embeddings.md).
- Everything else resolves to the same models as before.

When an earlier and a new form name the model for the same signal, migrate
keeps the new form's value. A rule that writes both `deployment` and `model`
or `model_path`, or a binding written under both its signal type and its
earlier name, is refused.

## Binding objects

A binding is keyed by its signal type and names a deployment. The signal
decides the contract and the provider decides the adapter.

```yaml title="migrate-binding-object-before"
recipes:
  - name: default
    model_bindings:
      jailbreak: {deployment: kai, contract: decision.v1}
```

```yaml title="migrate-binding-object-after"
recipes:
  - name: default
    model_bindings:
      jailbreak: kai
```

## Rule bindings

A binding for one rule becomes the rule's `deployment`.

```yaml title="migrate-rule-bindings-before"
recipes:
  - name: default
    model_bindings:
      safety.harmful: kai
    signals:
      safety:
        - {name: harmful, predicate: {gte: 0.5}}
```

```yaml title="migrate-rule-bindings-after"
recipes:
  - name: default
    signals:
      safety:
        - {name: harmful, deployment: kai, predicate: {gte: 0.5}}
```

## The memory rewrite model

Memory search found the model that rewrites a query with its conversation by
`model_role: memory_rewrite`. It is an `openai_chat` deployment that
`global.stores.memory.query_rewrite` names.

```yaml title="migrate-memory-rewrite-before"
global:
  model_catalog:
    external:
      - name: rewriter
        llm_provider: vllm
        model_role: memory_rewrite
        llm_endpoint: {address: llm.example.com, port: 8000, protocol: http}
        llm_model_name: qwen3
        llm_timeout_seconds: 30
```

```yaml title="migrate-memory-rewrite-after"
global:
  model_catalog:
    deployments:
      rewriter:
        provider: openai_chat
        endpoint: http://llm.example.com:8000/v1
        model: qwen3
        timeout_ms: 30000
  stores:
    memory:
      query_rewrite: {deployment: rewriter}
```

## Deployment fields

Top-level `question_types` moves under `capabilities`, and `served_name`, on a
deployment or a replica, becomes `model`. A deployment of the removed
`classify` or `http` provider is dropped: no signal asks a classifier service.

```yaml title="migrate-deployment-fields-before"
global:
  model_catalog:
    deployments:
      d3-gateway:
        provider: systemone
        endpoint: https://gateway.example.com/Unified/v1/systemone
        model: vllm-sr/d3-ultra
        schema: vllm-sr/5.0
        question_types: [choice, noul, score]
        auth: {api_key_env: GATEWAY_KEY}
      decision-kai:
        provider: model_runtime
        endpoint: https://decision-worker.example.com:8100
        served_name: decision-kai
```

```yaml title="migrate-deployment-fields-after"
global:
  model_catalog:
    deployments:
      d3-gateway:
        provider: systemone
        endpoint: https://gateway.example.com/Unified/v1/systemone
        model: vllm-sr/d3-ultra
        schema: vllm-sr/5.0
        capabilities: {question_types: [choice, noul, score]}
        auth: {api_key_env: GATEWAY_KEY}
      decision-kai:
        provider: model_runtime
        endpoint: https://decision-worker.example.com:8100
        model: decision-kai
```

## Request backends

`base_url`, `api_key_env`, `auth_header`, `auth_prefix` and `extra_headers`
become `endpoint`, `auth` and `headers`, the shape deployments use.

```yaml title="migrate-backend-refs-before"
providers:
  defaults: {model: claude}
  models:
    - name: claude
      backend_refs:
        - provider: anthropic
          base_url: https://gateway.example.com/anthropic
          api_key_env: ANTHROPIC_KEY
          auth_header: x-api-key
          auth_prefix: ""
          extra_headers: {X-Org: acme}
```

```yaml title="migrate-backend-refs-after"
providers:
  defaults: {model: claude}
  models:
    - name: claude
      backend_refs:
        - provider: anthropic
          endpoint: https://gateway.example.com/anthropic
          auth: {header: x-api-key, prefix: "", api_key_env: ANTHROPIC_KEY}
          headers: {X-Org: acme}
```
