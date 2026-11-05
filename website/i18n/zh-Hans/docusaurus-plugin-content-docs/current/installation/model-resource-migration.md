---
is_mtpe: false
title: 迁移模型资源
sidebar_label: 迁移模型资源
description: 每种早期为信号选择模型的方式，以及取代它的“部署加绑定”形式。
translation:
  source_commit: "3e9e4262e5c4e286ede3fb75cdac83edb985b6aa"
  source_file: "docs/installation/model-resource-migration.md"
  outdated: false
---

# 迁移模型资源

早期版本通过 system 行、模块的 `model_id` 和 `model_ref`、`external[]` 模型、模块 backend、分类器的 `type`/`model` 字段以及绑定契约来选择模型。这些形式仍可加载：Router 把每一种都读作有名称的部署和绑定，使用与之前相同的模型，并为每个字段报告一条 `deprecated_model_field` 警告。本页是唯一出现早期形式的地方；文档其余部分都使用每组示例中的第二种形式。[为信号选择模型](choose-a-model-for-a-signal.md)说明了它们迁移到的形式，[发布说明](../release-notes/unified-model-resources.md)概述了这次变更。

用 `vllm-sr config validate --config config.yaml` 查看这些警告，或者直接用 Router：`router -validate-config -config config.yaml` 输出一行 JSON：`{"valid": true, "warnings": [{"code": "deprecated_model_field", "field": "...", "message": "..."}]}`。每条警告都指明字段以及 Router 把它读作什么，例如：

```json
{"code": "deprecated_model_field", "field": "global.model_catalog.system.prompt_guard", "message": "model paths are deprecated; read as bindings.jailbreak: Vela-1.0-Encoder-307M-Guard (model_runtime vllm-sr/Vela-1.0-Encoder-307M-Guard on cpu)"}
```

`vllm-sr config migrate --config config.yaml` 把较早的文件改写为 v0.3 布局，并请 Router 把本页列出的每个字段写成对应示例中的第二种形式；文件的其余部分保持原样。它需要能运行的 Router（`VLLM_SR_ROUTER_BINARY` 指向的 Router 二进制文件，或 Router 镜像）；使用 `--offline` 时模型字段保持不变。直接写在配置里的密钥会移到环境变量中，输出会给出需要设置的变量名。如果某个已弃用字段包含任何部署字段都无法表达的设置，模型字段保持原样，输出会说明是哪个设置。

## 其他变化 {#what-else-changes}

- `systemone` 决策模型现在也回答内置信号。以前这些信号在本地的 `vela3-fast` 上运行。除非配置写出阈值，它们的阈值为 0.5，加载时警告 `uncalibrated_preset`。
- 安全规则的 hazard 部分没有自己的模型时，在任何决策模型下都使用预先声明的 `hazard` 部署（Vela 1.0 Hazard）。以前，含有这种 hazard 部分的配置在 `systemone` 决策模型下无法加载。
- 其他一切都解析到与之前相同的模型。

早期形式和新形式为同一个信号指定模型时，新形式优先，早期字段被忽略并给出警告。同时写了 `deployment` 和 `model` 或 `model_path` 的规则，或者同时以信号类型和早期名称写出的绑定，会被拒绝。

## System 行 {#system-lines}

指向决策模型所用模型的 system 行直接删除；指向其他模型的 system 行变为一个部署和一个绑定。

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

## 模块的 `model_id` {#module-model_id}

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

## `model_ref: ""` {#model_ref-}

空的 `model_ref` 表示该模块不运行任何模型。改为关闭该模块，或者删除该信号。

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

## 模块 backend 和 `external[]` 模型 {#module-backends-and-external-models}

模块的 `backend` 和它指向的 `external[]` 模型合并为一个部署，其 `provider` 就是该服务使用的 API。

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

## `provider: http` 部署和绑定契约 {#provider-http-deployments-and-binding-contracts}

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

## 绑定键和绑定对象 {#binding-keys-and-binding-objects}

绑定以信号类型为键，值是一个部署名称。信号决定契约，provider 决定适配器。

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

## 分类器规则 {#classifier-rules}

`type`、`model`、`model_path` 和 `use_cpu` 变为规则的 `deployment`。provider 决定如何询问该规则。

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

## 安全规则 {#safety-rules}

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

## 规则绑定 {#rule-bindings}

为单条规则或为所有 complexity 规则写的绑定，变为这些规则的 `deployment`。

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

## 远程 embedding {#remote-embeddings}

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

## 记忆查询改写模型 {#the-memory-rewrite-model}

记忆检索以前按 `model_role: memory_rewrite` 找到结合对话改写查询的模型。现在它是 `global.stores.memory.query_rewrite` 指定的 `openai_chat` 部署。

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

## RAG 重排序绑定 {#the-rag-reranker-binding}

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

## 部署字段 {#deployment-fields}

顶层的 `question_types` 移到 `capabilities` 下，`served_name` 改为 `model`，直接写在配置中的密钥（`access_key`、后端的 `api_key`）改为 `auth.api_key_env`；在部署上也可以改为 `auth.api_key_file`。

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

## 请求后端 {#request-backends}

`base_url`、`api_key_env`、`auth_header`、`auth_prefix` 和 `extra_headers` 变为 `endpoint`、`auth` 和 `headers`，与部署使用的形式相同。

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

## 准入键 {#admission-keys}

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
