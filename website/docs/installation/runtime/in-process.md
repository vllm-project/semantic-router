---
title: In-process models
description: Run Vela locally, choose CPU or GPU inference, and inspect routing signals.
---

Run [Vela models](../../tutorials/global/vela-models.md) inside the Router to
classify requests, generate embeddings, and rerank documents. The CLI downloads
registered models when their features are enabled. You need a separate backend
to generate chat responses.

## Choose your hardware

Install the CLI and a matching Router image using the
[installation guide](../installation.md).

| Hardware | Runtime | Vela model format | Start here |
| --- | --- | --- | --- |
| CPU | Candle | Native weights | The example below |
| CPU | ONNX Runtime | ONNX | Use `provider: ort`, `device: cpu` |
| OpenVINO device | OpenVINO-enabled build | Exported IR | [OpenVINO models](openvino.md) |
| AMD GPU | ONNX Runtime with ROCm or MIGraphX | ONNX | [Vela AMD recipe](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/vela-amd/README.md) |
| NVIDIA GPU | Candle CUDA build | Native weights | Use `provider: candle`, `device: cuda:0`; validate on your GPU |
| Apple GPU | Candle Metal build | Compatible native weights | Use `provider: candle`, `device: metal:0`; check model compatibility |

Candle supports GPU index `0`; BERT and BERT LoRA models do not support Metal.
The CPU and AMD paths have runtime coverage. NVIDIA and Apple deployments need
validation on the intended hardware. For a separately hosted model, use
[External services](external.md).

## Run a Vela classifier on CPU

This configuration uses Vela Domain to recognize programming requests and sends
answers to your existing backend. Replace `vllm:8000` with an endpoint reachable
from the Router container, then save the file as `config.yaml`:

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
          endpoint: vllm:8000
          protocol: http
routing:
  signals:
    domains:
      - name: computer science
        description: Programming and computer science requests.
        mmlu_categories: [computer science]
  decisions:
    - name: programming
      priority: 100
      rules:
        operator: AND
        on_unknown: fail_request
        conditions:
          - type: domain
            name: computer science
      modelRefs:
        - model: answer-model
global:
  model_catalog:
    bindings:
      domain_classifier:
        deployment: vela-domain
        contract: label_distribution.v1
        adapter: modernbert
        mapping_path: models/Vela-1.0-Encoder-307M-Domain/category_mapping.json
    deployments:
      vela-domain:
        artifact: models/Vela-1.0-Encoder-307M-Domain
        provider: candle
        device: cpu
        precision: fp32
        input:
          max_tokens: 512
          overflow: reject
```

A **deployment** selects the model, engine, device, and input budget. A
**binding** connects that deployment to a routing feature. Global bindings can
be reused by every recipe that needs the feature. Here, `domain_classifier`
uses `vela-domain`; both matched and unmatched requests use `answer-model`.
Change the decision's backend or plugins to apply your routing policy.

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
curl -fsS http://localhost:8080/ready
curl -fsS 'http://localhost:8080/api/v1/routing/preview?trace=true' \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","text":"Help me debug this Python program."}' \
  | jq '{decision_result, signal_confidences, signal_errors, metrics}'
```

Preview runs the classifier and reports the selected decision, signal scores,
errors, and timings. It does not call the answer backend. To test a complete
request:

```bash
curl -fsS http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Explain Python tuples briefly."}],"max_tokens":128}'
```

## Share model serving across recipes and services

Use `global.model_catalog` as the common source for in-process models. Declare
execution settings once in `deployments`, then connect features through
`bindings`. For example, merge this embedding setup into your configuration:

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
        provider: ort
        device: cpu
        precision: native
        input:
          max_tokens: 512
          overflow: reject
```

Recipes and shared services use these bindings when they need a model. Declaring
a model does not load it by itself. For example, semantic response caching needs
an embedding, while exact-match caching does not. Enable caching on the routes
that should use it; see [response caching](../../tutorials/plugin/response-cache.md).

Use a recipe's `routing.model_bindings` only when that recipe intentionally
needs a different model or execution policy. Top-level `routing.model_bindings`
overrides the default recipe, not shared services. A recipe override does not
change the global catalog used by caches, tools, memory, or vector stores.
Matching deployments can share physical model resources while recipe data and
cache entries remain isolated.
The shared cache also uses the global `hallucination_explainer` binding for NLI
verification; a recipe override changes only that recipe's checks.

Inspect the Dashboard model inventory after startup. Shared consumers appear
under `@global`; `metadata.resource_id` identifies a shared physical resource.
Different embedding layers or dimensions can be views of that resource. Device,
graph, precision, or input-policy differences require separate resources; they
do not cause an automatic CPU fallback. See [embedding views](embeddings.md#share-embeddings-with-services).

## Run Vela on AMD

Use the [Vela AMD recipe](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/vela-amd/README.md)
for all ten task models, including embeddings and RAG reranking. It supplies
the model revisions, graph selections, device settings, and Preview examples.

```bash
curl -fL -o vela-amd.yaml \
  https://raw.githubusercontent.com/vllm-project/semantic-router/main/config/recipes/vela-amd/config.yaml
vllm-sr config validate --config vela-amd.yaml
vllm-sr serve --platform amd --config vela-amd.yaml
```

Connect the recipe's `vela-default` backend before sending chat requests.
`--platform amd` selects the image and GPU access; explicit deployments still
control model placement. First startup may take several minutes to compile
MIGraphX models. See [startup troubleshooting](lifecycle-diagnostics.md#check-startup).

| AMD recipe component | Input limit |
| --- | --- |
| Complete routing signal pipeline | 8,192 tokens |
| Standalone Embedding and Reranker | 32,768 tokens |
| Hazard | 2,048-token windows within a 32,768-token request |

All limits include special tokens. The complete AMD pipeline currently has an
8K limit even though individual retrieval models accept 32K.

## Choose an input budget

Choose the smallest input budget that covers your routing task. Local
classifiers default to 512 tokens. A larger budget also requires a compatible
model and graph; see the [AMD 32K options](../amd-rocm.md#optional-32k-domain-and-factcheck-on-rocm)
when using ROCm.

For Domain, FactCheck, Feedback, or ordinary embedding inference, edit the
existing deployment's `input` block:

```yaml
global:
  model_catalog:
    deployments:
      vela-domain:
        # Keep the existing artifact, provider, device, and precision.
        input:
          max_tokens: 32768
          overflow: truncate
```

This fragment processes up to 32,768 encoder tokens, including special tokens.
`truncate` keeps the beginning and preserves the tokenizer's special-token
envelope; the result reports the original and processed counts and whether it
was truncated. `reject`, the default, returns an error when the input is too long.
Inputs within the limit remain complete.

**Encoder limits and generation limits are separate.** This setting only changes
the text seen by the encoder. It does not edit chat messages, system instructions,
or tool calls sent to the answer model. A 32K generation backend must fit its
prompt, chat/tool formatting, and requested output within 32K. To shorten that
conversation, configure [context compression](../../tutorials/plugin/context-compression.md).

Safety scans and rerankers have separate completeness requirements. Keep their
required windowing or rejection policy; a truncated scan is not a complete
safety result. See [safety input policies](safety.md#native-classifier-context).
Embedding signals also have a separate [full-context option](embeddings.md#input-policy).

After validation and activation, test short, at-limit, and over-limit input with
the encoder's tokenizer. Use [Route Preview](lifecycle-diagnostics.md#inspect-the-executed-path)
to check the signals, then send a real chat request to check the answer model.
Measure latency and quality: long-input support alone does not establish either.

## Add another model or task

- [Embeddings](embeddings.md): semantic matching, memory, caches, and retrieval.
- [Safety models](safety.md): Guard, Safety, Hazard, and PII.
- [Neural reranking](../../tutorials/plugin/rag.md#neural-reranking): bind
  `rag.reranker` and enable the RAG plugin's `rerank` option.
- [Classifier signals](../../tutorials/signal/learned/classifier.md): custom labels
  and independent category scores.

Custom artifacts can use a local directory without a registry entry. Include
complete weights, tokenizer, configuration, task labels, and any ONNX external
tensor files. LoRA deployments need merged weights. An architecture name such
as `modernbert` identifies the adapter; a compatible task head is still required.

For a locally rewritten graph, use a separate artifact directory and keep its
compatible tokenizer, configuration, and label mappings. Record the source
revision, export settings, precision, and file hashes there. Omit the upstream
`revision` pin for changed bytes: it verifies the published snapshot, not your
local derivative. Keep the original snapshot unchanged.

For ONNX classifiers, use `provider: ort`, select the device, and set the
binding's `head` to the graph path, such as `onnx/model.onnx`. For GPU-specific
graphs, follow the model's runtime configuration. The Router rejects unavailable
GPU providers and CPU fallback.

Start with global bindings. Add a recipe override only when independent model
configuration is required. See the [configuration reference](../../api/configuration-schema.mdx)
and [model update guide](lifecycle-diagnostics.md#update-a-running-model).

## Advanced MIGraphX settings

Set `compilation_cache_dir` to reuse compiled models after a restart. This optional
setting requires `provider: ort`, a `migraphx:N` device, and a persistent, writable
absolute directory outside model directories. It is disabled by default.

A changed model, GPU, precision, or compiler can require new compilation. See
[AMD troubleshooting](lifecycle-diagnostics.md#amd-startup-problems) for settings
that conflict with deployment configuration.
