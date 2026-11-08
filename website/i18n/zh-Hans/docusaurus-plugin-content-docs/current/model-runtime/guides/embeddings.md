---
title: 嵌入
description: 用于召回的嵌入：响应缓存、记忆、RAG、向量库和嵌入 API。
translation:
  source_commit: "439c22531380dfd51abf2ce33ce6dc4db87b1fb7"
  source_file: "docs/model-runtime/guides/embeddings.md"
  outdated: false
is_mtpe: true
---

# 嵌入 {#embeddings}

嵌入把文本变成向量，router 就能按语义召回。嵌入只用于召回，从不参与路由决策：

- [响应缓存](../../tutorials/plugin/response-cache.md)：重复的问题从缓存里答；
- [context 插件](../../tutorials/plugin/context.md)的记忆源和 RAG 源，以及向量库：找存着的事实和文档（`match.recall`）；
- `/v1/embeddings`。

默认模型是 Vela 1.0 Embedding，读得了 32,768 个 token；要快的话还能从更早的层取小向量：3、6、11、22 层，64、128、256、512 或 768 维。

## 打开它 {#turn-it-on}

启用一个做召回的功能就够了：Vela Embedding 由 router 跑在 CPU 上。`global.model_catalog.embeddings.default` 指定嵌入文本的部署，不改的话就是预先声明的 `vela-embedding`。`embeddings.multimodal` 指定嵌入图像和音频的部署；不设就不开（见[图像和音频](./multimodal.md)）。

## 选它跑在哪 {#choose-where-it-runs}

所有召回功能（响应缓存、记忆、向量库、RAG 源和 `/v1/embeddings`）都用 `embeddings.default` 指定的部署做嵌入。想改它的设备或输入预算，就用同一个名字声明这个部署：

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
    embeddings:
      default: vela-embedding
```

不同功能可以读同一个模型的不同视图，所以已有索引照样有效：用 Vela Embedding 时，记忆存 256 维向量，内存响应缓存存第 6 层的 256 维向量，向量库和 RAG 源用全部 768 维。它们共用一个部署；每个视图自成一个向量空间，分开存。

## 用别的模型建的库 {#stores-indexed-with-another-model}

向量由别的模型生成的库或集合，要在 `embedding` 块里写明那个模型的部署和向量的维度：`global.stores` 下的响应缓存、记忆和向量库，以及外部集合上 RAG 源的 `backend_config`：

```yaml
global:
  model_catalog:
    deployments:
      omni-nano:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Omni-Nano
        revision: 2ff2d66385dbdd661a560ec3e8bcb45a0527d92e
        device: cpu
        input: {overflow: truncate}
  stores:
    vector_store:
      enabled: true
      backend_type: memory
      embedding:
        deployment: omni-nano
        dimension: 384
```

没有 `embedding` 块的库用 `embeddings.default` 做嵌入。Router 加载配置时逐个检查：部署必须已声明、能嵌入文本，且给得出这个维度；远程部署看它的 `capabilities.dimensions`。

## 用外部嵌入服务 {#use-an-external-embedding-service}

任何 OpenAI 兼容的 `/embeddings` 端点（比如挂了嵌入模型的 `vllm serve`）都能换掉本地模型处理文本。把它声明为 `openai_embeddings` 部署，并在 `embeddings.default` 中指定它；Router 向 `<endpoint>/embeddings` 发送请求：

```yaml alternative
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
