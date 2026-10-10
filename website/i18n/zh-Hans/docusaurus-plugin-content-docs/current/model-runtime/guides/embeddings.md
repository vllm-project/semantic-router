---
title: 嵌入
description: 用于语义路由、语义缓存、记忆、RAG、工具选择和知识库的嵌入。
translation:
  source_commit: "439c22531380dfd51abf2ce33ce6dc4db87b1fb7"
  source_file: "docs/model-runtime/guides/embeddings.md"
  outdated: false
is_mtpe: true
---

# 嵌入 {#embeddings}

嵌入把文本变成向量，router 就能比语义。一个嵌入模型供所有需要它的功能用：

- [嵌入信号](../../tutorials/signal/learned/embedding.md)和[知识库](../../tutorials/signal/learned/kb.md)：按和示例文本的相似度路由；
- [语义缓存](../../tutorials/plugin/response-cache.md)：重复的问题从缓存里答；
- [记忆](../../tutorials/plugin/memory.md)和 [RAG](../../tutorials/plugin/rag.md)：找存着的事实和文档；
- [工具选择](../../tutorials/plugin/tool-selection.md)：给出贴合请求的那些工具。

默认模型是 Vela 1.0 Embedding，读得了 32,768 个 token；要快的话还能从更早的层取小向量：3、6、11、22 层，64、128、256、512 或 768 维。

## 打开它 {#turn-it-on}

启用一个需要嵌入的功能就够了，Vela Embedding 由 router 跑在 CPU 上。要定路由用的向量大小：

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          model_type: mmbert
          target_layer: 22
          target_dimension: 768
```

`mmbert` 指的是 Vela Embedding 这个槽位。Qwen3-Embedding-0.6B（1,024 维）用 `qwen3`，Vela Omni 用 `multimodal`（见[图像和音频](./multimodal.md)）。

## 选它跑在哪 {#choose-where-it-runs}

绑定名叫 `embedding`，读的是向量（`embedding.v1`）。所有需要嵌入的服务（缓存、记忆、向量库、工具）都用这个全局绑定，声明一次就够：

```yaml
global:
  model_catalog:
    deployments:
      vela-embedding:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Embedding
        device: cpu
        input:
          max_tokens: 8192
          overflow: truncate
    bindings:
      embedding:
        deployment: vela-embedding
        contract: embedding.v1
```

不同功能可以读同一个模型的不同视图，比如语义缓存用第 6 层的 256 维向量、路由用第 22 层的 768 维向量。它们共用一个部署；每个视图自成一个向量空间，分开存。

## 用外部嵌入服务 {#use-an-external-embedding-service}

任何 OpenAI 兼容的 `/embeddings` 端点（比如挂了嵌入模型的 `vllm serve`）都能换掉本地模型处理文本。它换掉的是上面那个 `embedding` 绑定，所以那个绑定就别写了：

```yaml alternative
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          backend: openai_compatible
          model_type: remote
          target_dimension: 1024
        endpoint:
          base_url: https://embedding.example.com/v1
          model: BAAI/bge-m3
          api_key_env: EMBEDDING_API_KEY
          dimensions: 1024
```

服务方会收到待嵌入的文本。要图像、音频、层视图或长文本开窗的功能，还是得用本地模型。

## 验一下 {#check-it}

这些 worker 级示例在含有 `vllm-srun` 的环境里运行（例如 Router 镜像）。Classify、embeddings、rerank 和 bundle 是 worker API；实例前端发布的是 System One 和 decision 请求。

```bash
vllm-srun serve vllm-sr/Vela-1.0-Encoder-307M-Embedding --device cpu --port 8100
curl -s localhost:8100/v1/embeddings -H 'content-type: application/json' \
  -d '{"input": ["How do I reset my password?", "I forgot my login password."], "dimensions": 256}'
```

响应是 OpenAI embeddings 格式。加 `"options": {"return_meta": true}` 还能拿到 `meta.representation`：标明这个向量空间的模型、层和维度。

## 换模型别混向量 {#change-the-model-without-mixing-vectors}

不同模型、不同层、不同大小的向量，从来不互相比。改其中任何一样之后，向量库要重建索引，需要的记忆要重新加，语义缓存也得等着重新攒。见[嵌入模型换了要重新嵌入](model-runtime/migrate.md#re-embed-when-the-embedding-model-changes)。
