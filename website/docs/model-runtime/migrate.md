---
title: Migrate from the native bindings
sidebar_label: Migrate from the native bindings
description: Update a configuration that used the candle, ONNX Runtime or OpenVINO back ends, legacy model names or the NLI explainer.
---

# Migrate from the native bindings

Earlier releases ran models inside the router with three back ends: candle,
ONNX Runtime (`ort`) and OpenVINO. Those back ends are gone. Every model now
runs in the [model runtime](model-runtime/overview.md), on CPU by default and on GPUs when
you ask for them. The [release note](release-notes/built-in-model-runtime.md)
lists every breaking change of that release.

**Most configurations need no change.** If you only turn features on (a
`domain` signal, the semantic cache, PII detection) and never named a back
end, the router picks the same Vela models as before and runs them in the
runtime.

You need to migrate if your configuration contains any of these:

- `provider: candle`, `provider: ort` or `provider: openvino`;
- `precision`, `custom_ops_profile` or `compilation_cache_dir` on a deployment;
- `variant`, `model_type`, `use_modernbert` or `use_mmbert_32k` on a module;
- a binding `head` that names an ONNX or OpenVINO graph file;
- an older model name such as `models/mom-domain-classifier`,
  `models/mmbert-embed-32k-2d-matryoshka` or `lettucedect`;
- the NLI explainer (`hallucination_explainer`, `nli_model`,
  `enable_nli_filtering`, `use_nli`) or the response cache's `polarity_guard`
  setting;
- a hallucination detector with `backend: endpoint`.

The router refuses these settings at startup and tells you to run the
migration command.

## 1. Run the migration

```bash
vllm-sr config migrate --config legacy.yaml --output legacy.migrated.yaml
```

The command writes the migrated file and prints every change it made, for
example:

```text
Changes to review
  global.model_catalog.deployments.vela-domain
      provider candle -> model_runtime, device cuda:0
  global.model_catalog.deployments.vela-domain.precision
      fp16 -> profile max_speed (the default exact profile runs FP32)
```

It prints a **Warning** for anything you still have to do yourself: a
relative path to a model of your own that it could not rewrite safely (the
migrated file keeps that value, and the router rejects it until you fix it),
or stored vectors to re-embed because their embedding model changed (see
[Re-embed when the embedding model changes](#re-embed-when-the-embedding-model-changes)).

## 2. Review the changes

| Before | After |
| --- | --- |
| `provider: candle`, `ort` or `openvino` | `provider: model_runtime` |
| `device: cpu` | `device: cpu` |
| `device: cuda:N` | `device: cuda:N` |
| `device: rocm:N` or `migraphx:N` | `device: rocm:N` |
| `device: metal:0` | `device: mps` |
| an OpenVINO device (`CPU`, `GPU`, `NPU`, ...) | `device: cpu`; Intel GPUs can use `device: xpu:0` (not yet validated) |
| `precision: fp16` | `profile: max_speed` (approximate; see [Profiles](./profiles.md)) |
| `precision: native` or `fp32` | removed: the default `exact` profile already runs FP32 |
| `custom_ops_profile`, `compilation_cache_dir` | removed: the runtime picks its own kernels |
| a graph `head` such as `onnx/model_fa.onnx` | removed: the runtime picks the model's graphs |
| `artifact: models/Vela-1.0-Encoder-307M-...` | `artifact: vllm-sr/Vela-1.0-Encoder-307M-...`, the Hub repository |
| `embedding_config.backend: candle` or `openvino` | removed |
| `gemma_model_path` or `bert_model_path`, even empty | removed: the runtime has no EmbeddingGemma or MiniLM family, and Vela Embedding (`mmbert_model_path`) replaces them |
| `embedding_model: bert` or `gemma` on the response cache, memory or vector store, or no `embedding_model` where MiniLM was the default | `embedding_model: mmbert` (Vela Embedding); re-embed stored vectors |
| a vector size Vela Embedding does not serve (such as MiniLM's 384) on a store that embeds with `mmbert` | 256 for memory, 768 for the response cache and the vector store; re-create the collection or index at that size |
| `model_selection.ml.model_type: bert` or `gemma` | `model_type: mmbert`; retrain the selection models on Vela Embedding vectors |
| `variant`, `model_type`, `use_modernbert`, `use_mmbert_32k` on a module | removed: the runtime reads the architecture from the model |
| a label map inside an older model's directory, such as `category_mapping_path: models/mom-domain-classifier/category_mapping.json` | removed: the router reads the labels of the model it runs |
| `mlp.device` on the MLP selection algorithm | removed: the MLP selector runs in the router |
| `grounding.nli_contradiction_penalty` of the fusion algorithm | `grounding.contradiction_penalty`: grounding now reads the hallucination detector |
| a hallucination detector with `backend: endpoint`, `endpoint` and `model_id` | a `hallucination_detector` binding to that chat service; an endpoint whose path is not `/v1` has to be written by hand |

### Model names that changed

Older task models are replaced by the Vela 1.0 model for the same task. The
replacements use the same label names, so your signal rules and route
conditions keep matching what they matched before.

| Old model | New model | What changes |
| --- | --- | --- |
| `mom-domain-classifier`, `mmbert32k-intent-*` and their aliases | `Vela-1.0-Encoder-307M-Domain` | Same 14 domains |
| `mom-pii-classifier`, `mom-mmbert-pii-detector`, `mmbert32k-pii-*` | `Vela-1.0-Encoder-307M-PII` | Same 17 PII types |
| `mom-jailbreak-classifier`, `mmbert32k-jailbreak-*` | `Vela-1.0-Encoder-307M-Guard` | Same `benign` / `jailbreak` labels |
| `mom-halugate-sentinel`, `mmbert32k-factcheck-*` | `Vela-1.0-Encoder-307M-FactCheck` | Same labels |
| `mom-feedback-detector`, `mmbert32k-feedback-*` | `Vela-1.0-Encoder-307M-Feedback` | Adds `NO_FEEDBACK` for messages without feedback |
| `mmbert32k-modality-router-merged` | `Vela-1.0-Encoder-307M-Modality` | Same `AR` / `DIFFUSION` / `BOTH` labels |
| `mom-halugate-detector`, LettuceDetect v1 and v2 | `Vela-1.0-Encoder-307M-Halu` | Same inputs; answer spans as before |
| `mmbert-embed-32k-2d-matryoshka` | `Vela-1.0-Encoder-307M-Embedding` | **Re-embed** stored vectors |
| EmbeddingGemma (`mom-embedding-flash`), MiniLM (`mom-embedding-light`) | `Vela-1.0-Encoder-307M-Embedding`, or an OpenAI-compatible embedding endpoint | **Re-embed** stored vectors |
| `multi-modal-embed-small` / `-large` | `Vela-1.0-Omni-Nano` / `-Mini` | **Re-embed** stored vectors |
| Qwen3-Embedding-0.6B (`mom-embedding-pro`) | unchanged | |

### Features that were retired

- **The NLI explainer** (`hallucination_explainer`, `enable_nli_filtering`,
  `include_explanation` on a local detector, `use_nli`). Hallucination checks
  still mark the unsupported spans of an answer; they no longer add an NLI
  verdict per span.
- **The response cache's `polarity_guard` setting.** Its NLI tier is gone, so
  there is nothing left to choose: the lexical guard, which catches negations
  and antonyms, always runs. The migration removes the block.
- **OpenVINO.** Intel CPUs run models on `cpu`; Intel GPUs can use the
  `xpu` device.
- **The ONNX Runtime MIGraphX and CK flash-attention paths.** AMD GPUs run
  models through PyTorch for ROCm (`device: rocm:N`).

## 3. Validate and start

```bash
vllm-sr config validate --config legacy.migrated.yaml
vllm-sr serve --config legacy.migrated.yaml
```

On the first start the runtime downloads the models it does not have yet.
[Run it with the router](./deploy.md#check-what-is-running) shows how to see
when every deployment is ready.

## A complete example

This configuration uses ONNX Runtime on an AMD GPU for embeddings, candle on
an NVIDIA GPU for a domain classifier, an older jailbreak model and the NLI
explainer:

```yaml title="legacy.yaml"
version: v0.3
listeners: []
providers:
  defaults:
    model: answer-model
  models:
    - name: answer-model
      backend_refs:
        - name: answer
          endpoint: vllm:8000
          provider: vllm
routing:
  model_bindings:
    embedding:
      deployment: vela-embedding
      contract: embedding.v1
      adapter: mmbert
      head: onnx/model_fa.onnx
global:
  model_catalog:
    deployments:
      vela-embedding:
        artifact: models/Vela-1.0-Encoder-307M-Embedding
        provider: ort
        device: rocm:0
        precision: native
        custom_ops_profile: ck_flash_attention
      vela-domain:
        artifact: models/Vela-1.0-Encoder-307M-Domain
        provider: candle
        device: cuda:0
        precision: fp16
    modules:
      prompt_guard:
        enabled: true
        model_id: models/mom-jailbreak-classifier
        variant: candle
      hallucination_mitigation:
        enabled: true
        detector:
          backend: candle
          model_id: models/Vela-1.0-Encoder-307M-Halu
          enable_nli_filtering: true
        explainer:
          model_id: models/mom-halugate-explainer
  stores:
    response_cache:
      enabled: true
      backend_type: memory
      polarity_guard:
        mode: lexical+nli
```

`vllm-sr config migrate --config legacy.yaml` writes:

```yaml title="legacy.migrated.yaml"
version: v0.3
listeners: []
providers:
  defaults:
    model: answer-model
  models:
  - name: answer-model
    backend_refs:
    - name: answer
      endpoint: vllm:8000
      provider: vllm
routing:
  model_bindings:
    embedding:
      deployment: vela-embedding
      contract: embedding.v1
      adapter: mmbert
global:
  model_catalog:
    deployments:
      vela-embedding:
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Embedding
        provider: model_runtime
        device: rocm:0
      vela-domain:
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Domain
        provider: model_runtime
        device: cuda:0
        profile: max_speed
    modules:
      prompt_guard:
        enabled: true
        model_id: models/Vela-1.0-Encoder-307M-Guard
      hallucination_mitigation:
        enabled: true
        detector:
          model_id: models/Vela-1.0-Encoder-307M-Halu
  stores:
    response_cache:
      enabled: true
      backend_type: memory
      embedding_model: mmbert
```

The response cache had no `embedding_model`, so it used MiniLM, the former
default; it now names Vela Embedding. Running the command again on the
migrated file changes nothing.

## Re-embed when the embedding model changes

Changing the embedding model changes the vector space. The router keeps
vectors of different models apart, so nothing breaks, but stored data does not
carry over:

- **Semantic cache:** entries from the old model are no longer found; the
  cache fills again with new traffic.
- **Memory:** memories stored with the old model are not returned. Re-add the
  memories you need to keep.
- **Vector stores and RAG:** re-index your documents with the new model.
  Milvus, Qdrant and hybrid RAG backends embed the query with Vela Embedding
  (768 dimensions) where they used MiniLM (384), so re-embed their collections
  with Vela Embedding; the migration prints a warning for each one.
- **Embedding signals and knowledge bases:** their example texts are embedded
  again at startup. Check similarity thresholds on your own traffic; the
  scores of a different model are not comparable.

## Kubernetes and Docker

Router images no longer contain native libraries. They contain the CPU
runtime, so managed models work out of the box. Remove any environment
variables, init containers or volumes that existed only for candle, ONNX
Runtime or OpenVINO. For GPUs, run a GPU runtime and attach to it; see
[Run it with the router](./deploy.md#on-kubernetes).

If you deploy with the operator, remove `embedding_models.gemma_model_path`
from your `SemanticRouter` resources before you upgrade. The resource
definition no longer has the field: kubectl's default strict validation
refuses a manifest that still sets it (`unknown field`), and a resource stored
before the upgrade loses the field, as if it had never been set: its router
embeds with the model the rest of the resource configures, such as Vela
Embedding through `mmbert_model_path`. Re-embed the vectors that EmbeddingGemma
stored (see
[Re-embed when the embedding model changes](#re-embed-when-the-embedding-model-changes)).
