---
title: Migrate model resources
sidebar_label: Migrate model resources
description: Each earlier way to choose a model for a signal, next to the deployment-and-binding form that replaces it.
---

# Migrate model resources

Earlier releases chose models with system lines, module `model_id` and
`model_ref`, `external[]` models, module backends, classifier `type`/`model`
fields and binding contracts. These forms still load: the Router reads each
as named deployments and bindings, uses the same models as before, and reports
a `deprecated_model_field` warning for each field. This page is the only place
the earlier forms appear; the rest of the documentation uses the second form of
each pair. [Choose a model for a signal](choose-a-model-for-a-signal.md)
explains the form they move to, and the
[release note](../release-notes/unified-model-resources.md) summarizes the
change.

See the warnings with `vllm-sr config validate --config config.yaml`, or with
the Router itself: `router -validate-config -config config.yaml` prints one
JSON line, `{"valid": true, "warnings": [{"code": "deprecated_model_field",
"field": "...", "message": "..."}]}`. Each warning names the field and what the
Router reads it as, for example:

```json
{"code": "deprecated_model_field", "field": "global.model_catalog.system.prompt_guard", "message": "model paths are deprecated; read as bindings.jailbreak: Vela-1.0-Encoder-307M-Guard (model_runtime vllm-sr/Vela-1.0-Encoder-307M-Guard on cpu)"}
```

`vllm-sr config migrate --config config.yaml` rewrites an older file into the
v0.3 layout and asks the Router to write each field on this page in the second
form of its pair; it leaves the rest of the file as written. It needs a Router
to run (a Router binary in `VLLM_SR_ROUTER_BINARY`, or the Router image); with
`--offline` the model fields stay as they are. A key written inline moves to
an environment variable, and the output names the variable to set. When a
deprecated field holds a setting no deployment field expresses, the model
fields stay as written and the output says which setting.

## What else changes

- A `systemone` decision model now answers the built-in signals too. Before,
  they ran on a local `vela3-fast`. Their thresholds are 0.5 unless the
  configuration writes them, and the load warns `uncalibrated_preset`.
- A safety rule's hazard part without its own model uses the pre-declared
  `hazard` deployment (Vela 1.0 Hazard) under every decision model. Before, a
  configuration with such a part failed to load under a `systemone` decision
  model.
- Everything else resolves to the same models as before.

When an earlier and a new form name the model for the same signal, the new
form wins and the earlier field is ignored with a warning. A rule that writes
both `deployment` and `model` or `model_path`, or a binding written under both
its signal type and its earlier name, is refused.

## System lines

A system line that names the decision model's model goes away; one that names
another model becomes a deployment and a binding.

```yaml title="migrate-system-lines-before"
global:
  model_catalog:
    system:
      decision_model: {deployment: primary}
      pii_classifier: models/vela3-fast
      prompt_guard: models/Vela-1.0-Encoder-307M-Guard
    modules:
      prompt_guard: {enabled: true, model_ref: prompt_guard, use_cpu: true}
      classifier:
        pii: {model_ref: pii_classifier, use_cpu: true}
```

```yaml title="migrate-system-lines-after"
global:
  model_catalog:
    deployments:
      vela-guard:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Guard
        revision: 087f9e401012df839c83717b746967ac7aebfa3e
        device: cpu
    system:
      decision_model: {deployment: primary}
    bindings:
      jailbreak: vela-guard
    modules:
      prompt_guard: {enabled: true}
```

## Module `model_id`

```yaml title="migrate-model-id-before"
global:
  model_catalog:
    modules:
      classifier:
        domain: {model_id: models/Vela-1.0-Encoder-307M-Domain, use_cpu: true}
```

```yaml title="migrate-model-id-after"
global:
  model_catalog:
    deployments:
      vela-domain:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Domain
        revision: f6354f54adcf38770f635ad903be2b00577f6c11
        device: cpu
    bindings:
      domain: vela-domain
```

## `model_ref: ""`

An empty `model_ref` ran no model for the module. Turn the module off instead,
or remove the signal.

```yaml title="migrate-empty-model-ref-before"
global:
  model_catalog:
    modules:
      classifier:
        domain: {model_ref: ""}
```

```yaml title="migrate-empty-model-ref-after"
global:
  model_catalog:
    modules:
      classifier:
        domain: {enabled: false}
```

## Module backends and `external[]` models

A module `backend` and the `external[]` model it named become one deployment
whose `provider` is the API it speaks.

```yaml title="migrate-module-backend-before"
global:
  model_catalog:
    external:
      - name: guard-service
        model_role: guardrail
        llm_endpoint: {address: guard.example.com, port: 443, protocol: https}
        llm_timeout_seconds: 5
    modules:
      prompt_guard:
        enabled: true
        jailbreak_mapping_path: models/guard/jailbreak_type_mapping.json
        backend: {protocol: http_classify, model: guard-service, contract: label_distribution.v1}
```

```yaml title="migrate-module-backend-after"
global:
  model_catalog:
    deployments:
      guard-service:
        provider: classify
        endpoint: https://guard.example.com:443
        timeout_ms: 5000
    bindings:
      jailbreak: guard-service
    modules:
      prompt_guard:
        enabled: true
        jailbreak_mapping_path: models/guard/jailbreak_type_mapping.json
```

## `provider: http` deployments and binding contracts

```yaml title="migrate-http-deployment-before"
global:
  model_catalog:
    external:
      - name: guard-service
        model_role: guardrail
        llm_endpoint: {address: guard.example.com, port: 443, protocol: https}
        llm_timeout_seconds: 5
    deployments:
      guard-http: {provider: http, external_model: guard-service}
    modules:
      prompt_guard: {enabled: true, positive_labels: [INJECTION]}
routing:
  model_bindings:
    prompt_guard: {deployment: guard-http, contract: label_distribution.v1, adapter: http_classify}
```

```yaml title="migrate-http-deployment-after"
global:
  model_catalog:
    deployments:
      guard-http:
        provider: classify
        endpoint: https://guard.example.com:443
        timeout_ms: 5000
    modules:
      prompt_guard: {enabled: true, positive_labels: [INJECTION]}
routing:
  model_bindings:
    jailbreak: guard-http
```

## Binding keys and binding objects

A binding is keyed by its signal type and names a deployment. The signal
decides the contract and the provider decides the adapter.

```yaml title="migrate-binding-object-before"
routing:
  model_bindings:
    domain_classifier: {deployment: kai, contract: label_distribution.v1, adapter: modernbert}
```

```yaml title="migrate-binding-object-after"
routing:
  model_bindings:
    domain: kai
```

## Classifier rules

`type`, `model`, `model_path` and `use_cpu` become the rule's `deployment`.
The provider decides how the rule is asked.

```yaml title="migrate-classifier-rules-before"
global:
  model_catalog:
    external:
      - name: judge
        model_role: classification
        llm_provider: openai
        llm_endpoint: {address: llm.example.com, port: 443, protocol: https}
        llm_model_name: gpt-4.1-mini
      - name: risk-service
        model_role: classification
        llm_endpoint: {address: risk.example.com, port: 8080, protocol: http}
routing:
  signals:
    classifiers:
      - {name: safety-head, type: local, model_path: models/Vela-1.0-Encoder-307M-Safety, use_cpu: true, labels: [safe, unsafe]}
      - {name: tool-risk, type: llm, model: judge, labels: [SAFE, RISKY], instructions: Choose RISKY only for destructive operations.}
      - {name: remote-risk, type: sequence_classifier, model: risk-service, labels: [SAFE, RISKY]}
```

```yaml title="migrate-classifier-rules-after"
global:
  model_catalog:
    deployments:
      vela-safety:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Safety
        revision: 6e70e725a5f4d86da10f5be5e4dfd1da0358bb85
        device: cpu
      judge: {provider: openai_chat, endpoint: https://llm.example.com:443/v1, model: gpt-4.1-mini}
      risk-service: {provider: classify, endpoint: http://risk.example.com:8080}
routing:
  signals:
    classifiers:
      - {name: safety-head, deployment: vela-safety, labels: [safe, unsafe]}
      - {name: tool-risk, deployment: judge, labels: [SAFE, RISKY], instructions: Choose RISKY only for destructive operations.}
      - {name: remote-risk, deployment: risk-service, labels: [SAFE, RISKY]}
```

## Safety rules

```yaml title="migrate-safety-rules-before"
global:
  model_catalog:
    external:
      - name: binary-service
        model_role: classification
        llm_endpoint: {address: safety.example.com, port: 8080, protocol: http}
      - name: hazard-service
        model_role: classification
        llm_endpoint: {address: hazard.example.com, port: 8080, protocol: http}
routing:
  signals:
    safety:
      - name: unsafe
        model: binary-service
        threshold: 0.5
        hazard: {model: hazard-service, labels: [violence, privacy], categories: [privacy], threshold: 0.6}
```

```yaml title="migrate-safety-rules-after"
global:
  model_catalog:
    deployments:
      binary-service: {provider: classify, endpoint: http://safety.example.com:8080}
      hazard-service: {provider: classify, endpoint: http://hazard.example.com:8080}
routing:
  signals:
    safety:
      - name: unsafe
        deployment: binary-service
        threshold: 0.5
        hazard: {deployment: hazard-service, labels: [violence, privacy], categories: [privacy], threshold: 0.6}
```

## Rule bindings

A binding for one rule, or for every complexity rule, becomes the rules'
`deployment`.

```yaml title="migrate-rule-bindings-before"
routing:
  model_bindings:
    classifier.risk: {deployment: content-risk, contract: label_distribution.v1, adapter: http_classify}
    complexity: {deployment: kai, contract: decision.v1}
  signals:
    classifiers:
      - {name: risk, labels: [low, high]}
    complexity:
      - name: hard
        threshold: 0.1
        hard: {candidates: [prove this theorem]}
        easy: {candidates: [say hello]}
```

```yaml title="migrate-rule-bindings-after"
routing:
  signals:
    classifiers:
      - {name: risk, deployment: content-risk, labels: [low, high]}
    complexity:
      - name: hard
        deployment: kai
        threshold: 0.1
        hard: {candidates: [prove this theorem]}
        easy: {candidates: [say hello]}
```

## Remote embeddings

```yaml title="migrate-embedding-endpoint-before"
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config: {backend: openai_compatible, model_type: remote}
        endpoint:
          base_url: https://embedding.example.com/v1
          model: BAAI/bge-m3
          api_key_env: EMBEDDING_API_KEY
          timeout_seconds: 5
          max_retries: 2
          dimensions: 1024
```

```yaml title="migrate-embedding-endpoint-after"
global:
  model_catalog:
    deployments:
      remote-embed:
        provider: openai_embeddings
        endpoint: https://embedding.example.com/v1
        model: BAAI/bge-m3
        auth: {api_key_env: EMBEDDING_API_KEY}
        timeout_ms: 5000
        max_retries: 2
        capabilities: {dimensions: 1024}
    embeddings:
      default: remote-embed
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

## The RAG reranker binding

```yaml title="migrate-rag-reranker-before"
global:
  model_catalog:
    deployments:
      document-ranker:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Reranker
        revision: a388e41cbbd5dc5f16b6389fa76d0b8b8a38a8bf
        device: cpu
        input: {max_tokens: 4096, overflow: reject}
routing:
  model_bindings:
    rag.reranker:
      deployment: document-ranker
      contract: relevance_scores.v1
      adapter: vela_reranker
      pair_scorer: {layer: 22, dimension: 768}
  decisions:
    - name: rag
      plugins:
        - type: rag
          configuration: {enabled: true, backend: vectorstore, rerank: {top_k: 3}}
```

```yaml title="migrate-rag-reranker-after"
global:
  model_catalog:
    deployments:
      document-ranker:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Reranker
        revision: a388e41cbbd5dc5f16b6389fa76d0b8b8a38a8bf
        device: cpu
        input: {max_tokens: 4096, overflow: reject}
routing:
  decisions:
    - name: rag
      plugins:
        - type: rag
          configuration:
            enabled: true
            backend: vectorstore
            rerank: {deployment: document-ranker, top_k: 3, layer: 22, dimension: 768}
```

## Deployment fields

Top-level `question_types` moves under `capabilities`, `served_name` becomes
`model`, and a key written inline (`access_key`, a backend's `api_key`) moves
to `auth.api_key_env`, or on a deployment to `auth.api_key_file` as well.

```yaml title="migrate-deployment-fields-before"
global:
  model_catalog:
    deployments:
      d3-gateway:
        provider: systemone
        endpoint: https://gateway.example.com/Unified/v1/systemone
        model: vllm-sr/d3-ultra
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

## Admission keys

```yaml title="migrate-admission-before"
global:
  model_catalog:
    admission:
      prompt_guard: {max_concurrency: 4, max_queue: 64, on_overflow: shed}
```

```yaml title="migrate-admission-after"
global:
  model_catalog:
    admission:
      jailbreak: {max_concurrency: 4, max_queue: 64, on_overflow: shed}
```
