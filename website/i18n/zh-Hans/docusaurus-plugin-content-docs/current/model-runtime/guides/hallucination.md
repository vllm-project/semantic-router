---
title: 幻觉检查
description: 把模型的答案和给它的上下文对一遍，标出上下文不撑腰的部分。
translation:
  source_commit: "439c22531380dfd51abf2ce33ce6dc4db87b1fb7"
  source_file: "docs/model-runtime/guides/hallucination.md"
  outdated: false
is_mtpe: true
---

# 幻觉检查 {#hallucination-checks}

Vela 1.0 Halu 把请求带的上下文（工具结果、检索来的文档）、用户的问题和模型的答案一起读，标出答案里没有上下文支撑的片段。[`hallucination` 插件](../../tutorials/plugin/hallucination.md) 随后加一个警告头、在响应体里加一条说明，或者只把结果记下来。

它查的是有没有人撑腰，不是查真相：答案可以被错误的上下文撑住，而没有上下文可查的答案，报的是未验证。

## 打开它 {#turn-it-on}

把这次观察声明成信号，再在要检查的路由上用插件执行。一个请求提的主张值不值得查，先由 Vela FactCheck 决定。

```yaml
routing:
  signals:
    fact_check:
      - name: needs_fact_check
        description: Requests that make factual claims.
    hallucination:
      - name: ungrounded_claims
        description: Claims the context does not support.
  decisions:
    - name: grounded-answers
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: fact_check
            name: needs_fact_check
      modelRefs:
        - model: answer-model
      plugins:
        - type: hallucination
          configuration:
            enabled: true
            hallucination_action: header
            unverified_factual_action: header
            include_hallucination_details: true
global:
  model_catalog:
    modules:
      hallucination_mitigation:
        enabled: true
```

FactCheck 和 Halu 都由 router 跑在 CPU 上。Halu 合起来最多读 8,192 个 token 的上下文、问题和答案；上下文更长就从尾部截，保证答案永远查得到。

## 选它跑在哪 {#choose-where-it-runs}

绑定名叫 `hallucination_detector`，读答案的文本片段（`token_spans.v1`）：

```yaml
global:
  model_catalog:
    deployments:
      vela-halu:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Halu
        device: cpu
        input:
          max_tokens: 8192
          overflow: reject
    bindings:
      hallucination_detector:
        deployment: vela-halu
        contract: token_spans.v1
```

## 验一下 {#check-it}

这些 worker 级示例在含有 `vllm-srun` 的环境里运行（例如 Router 镜像）。Classify、embeddings、rerank 和 bundle 是 worker API；实例前端发布的是 System One 和 decision 请求。

```bash
vllm-srun serve vllm-sr/Vela-1.0-Encoder-307M-Halu --device cpu --port 8100
curl -s localhost:8100/v1/classify -H 'content-type: application/json' -d '{
  "input": [{"context": "The Eiffel Tower is 330 metres tall and stands in Paris.",
             "question": "How tall is the Eiffel Tower?",
             "answer": "The Eiffel Tower is 450 metres tall."}]
}'
```

结果列出答案里没有支撑的 `spans`——这里是高度——带它们在答案里的字符偏移。经过 router 时，响应带着幻觉警告头和 `x-vsr-matched-hallucination`。

早先的版本能给每个片段附一段 NLI 解释。那个解释器已退役；见[迁移](model-runtime/migrate.md#features-that-were-retired)。
