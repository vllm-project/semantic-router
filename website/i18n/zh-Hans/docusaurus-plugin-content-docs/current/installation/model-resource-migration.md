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

早期版本通过 `external[]` 模型、绑定对象和按规则的绑定键来选择模型。这些形式仍可加载：Router 把每一种都读作有名称的部署和绑定，使用与之前相同的模型，并为每个字段报告一条 `deprecated_model_field` 警告。本页是唯一出现早期形式的地方；文档其余部分都使用每组示例中的第二种形式。[为信号选择模型](choose-a-model-for-a-signal.md)说明了它们迁移到的形式，[发布说明](../release-notes/unified-model-resources.md)概述了这次变更。

用 `vllm-sr config validate --config config.yaml` 查看这些警告，或者直接用 Router：`router -validate-config -config config.yaml` 输出一行 JSON：`{"valid": true, "warnings": [{"code": "deprecated_model_field", "field": "...", "message": "..."}]}`。每条警告都指明字段以及 Router 把它读作什么。

`vllm-sr config migrate --config config.yaml` 把较早的文件改写为 v0.3 布局，并请 Router 把本页列出的每个字段写成对应示例中的第二种形式；文件的其余部分保持原样。它需要能运行的 Router（`VLLM_SR_ROUTER_BINARY` 指向的 Router 二进制文件，或 Router 镜像）；使用 `--offline` 时模型字段保持不变。直接写在配置里的密钥会移到环境变量中，输出会给出需要设置的变量名。如果某个已弃用字段包含任何部署字段都无法表达的设置，模型字段保持原样，输出会说明是哪个设置。

## 其他变化 {#what-else-changes}

- `systemone` 决策模型现在也回答内置信号。以前这些信号在本地的 `vela3-fast` 上运行。除非配置写出阈值，它们的阈值为 0.5，加载时警告 `uncalibrated_preset`。
- 只有被移除的学习型分类器使用的信号和模块（classifier、embedding、知识库和 preference 规则，domain、fact-check、feedback 和 modality 模块，安全规则的 hazard 部分）不再能加载。`vllm-sr config migrate` 会把它们改写为决策模型回答的应用信号，或说明应改写成什么。
- prompt guard、PII 和幻觉检测模块同样不再能加载，它们的 system 行、admission 键、模块 backend 以及 `external[]` 中的 guard 字段也一样。决策模型回答 jailbreak、pii 和 hallucination 规则；模块的阈值成为读取它的规则的 predicate，绑定其他模型时以信号类型为键。
- embedding 模型关键字（`mmbert`、`qwen3`、`multimodal`）、带远程 endpoint 的 `embeddings.semantic`、embedding 绑定以及按模型关键字的 admission 不再能加载。`vllm-sr config migrate` 会写出每一项所选的部署，并在使用它的地方写上它的名称；参见 [Embeddings](../model-runtime/guides/embeddings.md)。
- 其他一切都解析到与之前相同的模型。

早期形式和新形式为同一个信号指定模型时，新形式优先，早期字段被忽略并给出警告。同时写了 `deployment` 和 `model` 或 `model_path` 的规则，或者同时以信号类型和早期名称写出的绑定，会被拒绝。

## 绑定对象 {#binding-objects}

绑定以信号类型为键，值是一个部署名称。信号决定契约，provider 决定适配器。

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

## 规则绑定 {#rule-bindings}

为单条规则写的绑定，变为该规则的 `deployment`。

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
