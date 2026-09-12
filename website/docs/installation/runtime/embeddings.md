---
title: Embeddings
description: Choose local or remote embeddings and match consumer dimensions, layers, tokenizer windows, and modalities.
---

# Embeddings

Embedding consumers share an execution provider while retaining their own
vector and input requirements. A provider must satisfy every reachable
consumer before the candidate configuration is published.

## Choose a local model

The semantic embedding catalog keeps the existing model selection:

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          model_type: mmbert
          preload_embeddings: true
          target_dimension: 768
          target_layer: 22
        mmbert_model_path: models/mmbert-embed-32k-2d-matryoshka
```

The catalog path must contain a compatible embedding checkpoint. A recipe can
instead bind its primary embedding provider explicitly. This complete example
uses the maintained ROCm image and a compatible mmBERT ONNX artifact. Replace
the downstream endpoint and provision the checkpoint before serving:

```yaml
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: answer-model
  models:
    - name: answer-model
      backend_refs:
        - name: answer
          endpoint: 127.0.0.1:8000
          protocol: http
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          model_type: mmbert
          preload_embeddings: true
          target_dimension: 768
          target_layer: 22
    deployments:
      local-embedding:
        artifact: models/mmbert-embed-32k-2d-matryoshka
        provider: ort
        device: migraphx:0
        precision: native
        input:
          max_tokens: 1024
          overflow: reject
routing:
  model_bindings:
    embedding:
      deployment: local-embedding
      contract: embedding.v1
      adapter: mmbert
  signals:
    embeddings:
      - name: technical-support
        threshold: 0.75
        aggregation_method: max
        candidates:
          - how to configure the system
          - troubleshooting an installation error
  decisions:
    - name: answer-support
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: embedding
            name: technical-support
      modelRefs:
        - model: answer-model
```

Save the example as `config.yaml` and run
`vllm-sr config validate --config config.yaml` before the
[local-image serve workflow](in-process.md#run-a-generic-sequence-classifier).
The 1024-token budget and similarity threshold are configuration examples,
not measured GPU quality or performance guarantees. Validate the selected
graph, budget, vectors, and provider profile for your deployment.

Candle supports BERT, Qwen3, Gemma, mmBERT, and the text branch of compatible
multimodal checkpoints. ORT supports mmBERT and multimodal graphs. Select
`provider: ort` only when the artifact contains the required graph files.
An explicit primary binding replaces that primary's obsolete module path;
other consumers selecting a distinct catalog model still need that model.

## Dimensions, layers, and windows

| Consumer | Requirement to preserve |
| --- | --- |
| Embedding/KB/reask/contrastive signals and embedding-based selectors | The trained or indexed embedding space and actual output dimension |
| In-memory semantic cache with mmBERT | Existing layer 6, dimension 256 view |
| Multimodal cache defaults | Existing dimension 384 view |
| Persistent cache and vector stores | Their configured or loaded index/schema dimension |
| Memory | Its configured family and dimension; mmBERT defaults to 256, multimodal to 384 |
| Response-cache and RAG query windowing | Actual tokenizer windows with byte boundaries |
| Image-based complexity | A provider with an actual image encoder |

An output dimension alone does not identify an embedding space. Rebuild or
migrate stored vectors and recalibrate thresholds when changing model,
revision, layer, or provider.

mmBERT layer/dimension requests use an owned view of the prepared model. The
layer must actually be loaded and the dimension supported. ORT layer exports
are discovered from their manifest/graphs; missing requested layers fail
preparation. Without an explicit graph, the loader selects a primary graph
from the artifact. An explicit `head` may select an early-exit graph declared
by the artifact's layer layout and manifest; that graph remains the only
execution target for its layer. A full-depth primary uses the architecture's
declared depth. External tensors come from real ONNX references rather than
an assumed `.data` filename.

Available-layer metadata comes from the native sessions that actually loaded.
The primary graph is loaded once even when it is also a selectable layer.
Before readiness, the Router executes a warmup for every advertised ORT layer
and checks its output values and dimension. A failed layer rejects the
candidate while the previous generation remains available. Cold compilation
for those layers is part of preparation, not evidence of steady-state latency.

Embedding token limits come from the loaded model and the deployment's
restriction, not the classification task's 512 cap. Multimodal text has its
own text-encoder limit. Tokenizer windows use actual token boundaries and
UTF-8 offsets; a zero window budget selects the prepared effective limit.
This window API does not enable automatic `input.overflow: window` for a
classifier that lacks such an adapter.

Owned MIGraphX mmBERT embeddings require an explicit positive
`input.max_tokens` in a deployment selected by the recipe's `embedding`
binding. Setting only the catalog's `use_cpu: false` does not supply this
budget. Omitting it or setting zero fails preparation. The tensor
length is fixed to that budget, within the model's actual capacity; the
maintained 32K checkpoint permits at most 32768 tokens. Shorter inputs use
padding with a zero attention mask. Token usage and tokenizer windows still
describe the real input, and special tokens count toward the budget. This is
separate from the classification task's 512-token cap. Owned CPU and legacy
embedding execution retain dynamic tensor lengths.

The AMD `serve` platform default preserves the semantic catalog's `use_cpu`
setting, using CPU when that setting is omitted. The explicit deployment in
the example selects MIGraphX independently of this catalog default.

## Use a remote text provider

Set a token in the Router environment and merge this fragment:

```bash
export EMBEDDING_API_KEY="<provider-key>"
```

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

The Router appends `/embeddings` unless that suffix is already present.
`endpoint.dimensions` and `target_dimension` must agree when both are set.
The response cap defaults to 16 MiB. Authentication is a bearer token, not
arbitrary custom headers or a provider-specific body. Startup checks returned
vectors for the expected dimension and valid values.

Embedding signals keep their existing candidates and thresholds. In this
remote mode, text consumers can use the shared provider, but the provider does
not offer exact tokenizer windows, layer selection, image, or audio encoding.
If a reachable consumer requires one of those capabilities, preparation fails
before publication. It does not invent windows or quietly load the overridden
primary local model. Use an explicitly supported local binding for a recipe
that needs those capabilities, or change that recipe's consumers.

Remote services see the text they embed. Their data-retention and logging
policies apply. Changing providers can change matching behavior even when
vector dimensions are identical.

## Multimodal models

A multimodal artifact must contain the encoders the consumer uses. The text,
image, and audio capabilities are checked individually. Candle accepts image
bytes through its owned encoder and audio feature tensors; ORT validates its
image tensor layout/resize and audio mel input contract. The current ORT audio
path uses the Whisper 3000-frame budget and reports explicit reject/truncate
behavior. Do not infer an encoder from the catalog name alone.

Remote OpenAI-compatible embedding execution currently supports text only.
A successful text warmup does not prove image/audio support. See
[Engines and hardware](engines-and-hardware.md#tasks-by-engine) for the
implementation and hardware-evidence boundaries.

## Inspect a running provider

```bash
curl -fsS http://localhost:8080/startup-status | jq '.embedding_provider'
```

Remote provider status exposes redacted metadata, including the key's environment
variable name and whether it is present, never the token value. Use
`POST /api/v1/diagnostics/embeddings` for actual text/image requests supported
by the current provider; the [API reference](../../api/apiserver.md) describes
the request shape. Diagnostics and stores borrow the current generation's
provider until their operation finishes.
