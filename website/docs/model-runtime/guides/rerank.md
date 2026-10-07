---
title: Rerank documents
description: Reorder retrieved documents by their relevance to the request before they reach the model.
---

# Rerank documents

Vector search finds documents that are similar to a request; a reranker reads
the request and each document together and scores how well the document
answers it. Vela 1.0 Reranker reorders the candidates of the
[RAG plugin](tutorials/plugin/rag.md#neural-reranking) so the most
relevant ones reach the prompt.

## Turn it on

Describe the reranker deployment, bind it as `rag.reranker` (it reads
relevance scores, `relevance_scores.v1`) and opt a route's RAG plugin into
`rerank`:

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

`top_k` retrieves ten candidates and `rerank.top_k` keeps the best three.
`pair_scorer` selects the size of the model: the full 22 layers and 768
dimensions are the most accurate; layers 3, 6 and 11 and smaller dimensions
are faster.

The input limit covers the request and one document together. A pair over the
limit is rejected, never shortened, and the plugin's `on_failure` decides what
happens to the request.

## Check it

```bash
vllm-sr serve vllm-sr/Vela-1.0-Encoder-307M-Reranker --device cpu --port 8100
curl -s localhost:8100/v1/rerank -H 'content-type: application/json' -d '{
  "query": "How do I reset my password?",
  "documents": ["Our offices are closed on Sunday.", "Open Settings, then Security, then Reset password."],
  "top_n": 2
}'
```

The results come back most relevant first, each with the document's `index`,
its `relevance_score` and the raw `logit`. Scores rank documents for one
request; they are not probabilities you can compare across requests.
