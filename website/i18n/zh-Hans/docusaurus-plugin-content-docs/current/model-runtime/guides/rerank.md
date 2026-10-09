---
title: 重排文档
description: 在文档到达模型之前，按它们和请求的相关性重新排序。
translation:
  source_commit: "439c22531380dfd51abf2ce33ce6dc4db87b1fb7"
  source_file: "docs/model-runtime/guides/rerank.md"
  outdated: false
is_mtpe: true
---

# 重排文档 {#rerank-documents}

向量搜索找的是和请求相似的文档；重排器把请求和每篇文档放在一起读，给这篇文档答不答得上这个问题打分。Vela 1.0 Reranker 把 [RAG 插件](../../tutorials/plugin/rag.md#neural-reranking) 的候选重新排序，最相关的先进提示词。

## 打开它 {#turn-it-on}

写一个重排器部署、绑成 `rag.reranker`（它读相关性分，`relevance_scores.v1`），再让路由的 RAG 插件选择加入 `rerank`：

```yaml
global:
  model_catalog:
    deployments:
      document-ranker:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Reranker
        device: cpu
        input:
          max_tokens: 4096
          overflow: reject
routing:
  model_bindings:
    rag.reranker:
      deployment: document-ranker
      contract: relevance_scores.v1
      pair_scorer:
        layer: 22
        dimension: 768
  decisions:
    - name: answer-from-docs
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: answer-model
      plugins:
        - type: rag
          configuration:
            enabled: true
            backend: vectorstore
            backend_config:
              vector_store_id: vs-your-documents
            top_k: 10
            rerank:
              top_k: 3
            on_failure: block
```

`top_k` 取回十个候选，`rerank.top_k` 留最好的三个。`pair_scorer` 选模型大小：完整的 22 层、768 维最准；3、6、11 层和更小的维度更快。

输入上限管的是请求加一篇文档合起来的长度。超限的配对直接拒，绝不截短，请求何去何从看插件的 `on_failure`。

## 验一下 {#check-it}

```bash
vllm-sr serve vllm-sr/Vela-1.0-Encoder-307M-Reranker --device cpu --port 8100
curl -s localhost:8100/v1/rerank -H 'content-type: application/json' -d '{
  "query": "How do I reset my password?",
  "documents": ["Our offices are closed on Sunday.", "Open Settings, then Security, then Reset password."],
  "top_n": 2
}'
```

结果按相关度从高到低返回，每条带文档的 `index`、`relevance_score` 和原始 `logit`。分数只给一个请求内的文档排名；不是能跨请求比较的概率。
