---
title: Embeddings
description: Use Vela for semantic routing and retrieval, or connect a remote embedding service.
---

Embeddings turn text into vectors for semantic matching, retrieval, caches, and
memory. Vela Embedding is the default local model. Choose the local CPU setup
below, [AMD GPU inference](#amd-gpu), or a [remote service](#remote-embeddings).
Merge configuration fragments into an existing Router configuration.

## Local embeddings

Declare Vela once in the global catalog so recipes and services can reuse it.
This CPU example uses layer 22 and 768 dimensions for semantic routing:

```yaml
global:
  model_catalog:
    bindings:
      embedding:
        deployment: vela-embedding
        contract: embedding.v1
        adapter: mmbert
    deployments:
      vela-embedding:
        artifact: models/Vela-1.0-Encoder-307M-Embedding
        provider: candle
        device: cpu
        precision: fp32
        input:
          max_tokens: 512
          overflow: reject
    embeddings:
      semantic:
        embedding_config:
          model_type: mmbert
          preload_embeddings: true
          target_layer: 22
          target_dimension: 768
```

Use the CPU image for Candle inference. The name `mmbert` selects the compatible
inference architecture; the deployment selects Vela. The normal serve workflow
downloads the registered model when an enabled feature needs it.

### AMD GPU

To run Vela Embedding on AMD, add an explicit ROCm deployment and binding:

```yaml
global:
  model_catalog:
    bindings:
      embedding:
        deployment: local-embedding
        contract: embedding.v1
        adapter: mmbert
        head: onnx/model_fa.onnx
    deployments:
      local-embedding:
        artifact: models/Vela-1.0-Encoder-307M-Embedding
        revision: 1e57cebf5a7b7fec6e6973f05bbca97c5cca4436
        provider: ort
        device: rocm:0
        precision: native
        custom_ops_profile: ck_flash_attention
        input:
          max_tokens: 32768
          overflow: reject
```

Start with `vllm-sr serve --platform amd --config config.yaml`. The AMD image
must include the ROCm execution provider and CK operator library. The published
graph supports inputs through 32,768 tokens, including special tokens.
The Router rejects CPU fallback for this GPU deployment.

Keep the downloaded ONNX companion graphs and external weights together;
enabled features may use different embedding layers. For a complete
configuration including classifiers and reranking, use the
[Vela AMD recipe](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/vela-amd/README.md).
Its complete classifier pipeline has an 8K input limit.

## Share embeddings with services

The global embedding binding is also used by enabled caches, tools, memory,
and vector stores. Reuse that deployment rather than declaring another copy
for each service. A recipe can override its own binding when isolation is needed;
it does not change the shared services' model.

Services may need different representations. The in-memory semantic cache uses
`mmbert` layer 6 with 256 dimensions, while the routing example above uses layer
22 with 768 dimensions. Keep the model's companion graphs and weights available
for every required view, and keep the cache's `embedding_model` consistent with
the global model and adapter. Startup checks compatibility. Use only layer views
supported by the selected artifact; a layer-22 vector is not a substitute for a
layer-6 vector. Exact-match caching does not need an embedding.

## Test an embedding

After starting the Router, check readiness and generate two vectors:

```bash
curl -fsS http://localhost:8080/ready
curl -fsS http://localhost:8080/api/v1/diagnostics/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"texts":["How do I reset my password?","I forgot my login password."],"model":"mmbert","target_layer":22,"dimension":768}' \
  | jq '{total_count, total_processing_time_ms, embeddings: [.embeddings[] | {dimension, model_used, processing_time_ms}]}'
```

Expect two 768-dimensional results and measured processing times. To inspect
how your semantic signals affect routing, use
[Route Preview](lifecycle-diagnostics.md#inspect-the-executed-path).
The [API reference](../../api/apiserver.md) also covers similarity requests.

## Input policy

Semantic routing uses representative text samples by default. To embed the
complete routing text, set:

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          full_context: true
```

This setting applies to semantic embedding signals and local embedding-backed
Complexity. Independent remote Complexity scorers retain their own input policy.
Prompt compression still applies unless the signal is exempted.

The deployment's `input.max_tokens` still limits the processed input, including
special tokens. For ordinary embedding inference, `overflow: truncate` truncates
oversized input before constructing the embedding; `overflow: reject` rejects
it. This does not truncate the chat messages forwarded to a generation backend.
See [encoder input budgets](in-process.md#choose-an-input-budget).
Increasing that limit does not turn on `full_context`. Choose a budget that
fits the artifact and your latency target; long-input capacity alone does
not establish retrieval accuracy.

Vela offers layers 3, 6, 11, and 22 and dimensions 64, 128, 256, 512, and 768.
Smaller representations can reduce cost; evaluate retrieval quality before
changing `target_layer` or `target_dimension`.

## Remote embeddings

Set a provider key in the Router environment:

```bash
export EMBEDDING_API_KEY="<provider-key>"
```

Then configure the provider's model and vector size:

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          backend: openai_compatible
          model_type: remote
          preload_embeddings: false
          target_dimension: 1536
        endpoint:
          base_url: https://embedding.example.com/v1
          model: text-embedding-model
          api_key_env: EMBEDDING_API_KEY
          timeout_seconds: 10
          max_retries: 2
          max_response_bytes: 16777216
          dimensions: 1536
```

Replace the URL, model, and dimensions with your provider's values. Both
dimension settings must match. The Router calls `/embeddings` using bearer
authentication. The service receives the text being embedded.

Remote embeddings support text. Features requiring local tokenizer windows,
layer selection, image encoding, or audio encoding need a compatible local model.

## Change a model without mixing vector spaces

Reindex stored documents when changing embedding weights, layer, or dimension.
Equal vector dimensions do not make two embedding spaces compatible. Persistent
caches and memory isolate representations; existing vector-store documents need
re-ingestion when the space changes.

| Feature | Before changing representations |
| --- | --- |
| Semantic signals and model selectors | Recheck thresholds and selector compatibility |
| Vector stores, memory, and persistent caches | Match dimensions and reindex affected data |
| In-memory mmBERT cache | Retain the layer-6, 256-dimensional representation |
| Response cache and RAG windows | Retain local tokenizer-window support |

For ONNX deployments, include every layer used by enabled features. Missing
layers prevent startup. See [Troubleshooting](lifecycle-diagnostics.md).
