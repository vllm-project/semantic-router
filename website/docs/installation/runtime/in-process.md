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
  model_bindings:
    domain_classifier:
      deployment: vela-domain
      contract: label_distribution.v1
      adapter: modernbert
      mapping_path: models/Vela-1.0-Encoder-307M-Domain/category_mapping.json
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
**binding** connects it to a feature in the recipe. Here, `domain_classifier`
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

Declare shared task bindings in `global.model_catalog.bindings`. They use the
same binding fields as `routing.model_bindings`, with execution settings kept
in `global.model_catalog.deployments`:

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
  stores:
    response_cache:
      enabled: true
      embedding_model: mmbert
```

Recipes inherit these serving defaults. An explicit recipe binding overrides
only that recipe's consumer; the top-level `routing.model_bindings` belongs to
the default recipe. Shared response-cache, tool, memory, and vector-store
consumers resolve the global catalog independently of recipe overrides.
Rule-named classifier and safety bindings apply only where the matching rule
exists. A catalog declaration alone does not load a model.

Enable the `response_cache` plugin on the decisions whose responses should be
cached. With routing decisions present, an enabled store without a reachable
cache consumer does not load an embedding. Exact-only consumers use the store
without an embedding. The existing global semantic-cache behavior remains for
configurations without routing decisions.

The in-memory semantic cache uses the `mmbert` layer-6, 256-dimension view;
routing can use layer 22 and 768 dimensions of the same serving resource. The
configured cache model must match the global embedding model and adapter.
Startup validates the required tokenizer, layers, and dimensions before
publishing readiness. Consumer views do not change the deployment's device,
precision, graph, or input policy. Incompatible execution settings create
separate resources rather than falling back to CPU.

Resource sharing preserves recipe-scoped handles and cache partitions. Cache
entries remain isolated by recipe, decision, backend selection, protocol, user
scope, and representation identity. An ORT embedding resource can contain
multiple materialized layer sessions; sharing the resource does not imply a
single ONNX session. Model inventory exposes `metadata.resource_id` from the
actual resource pool and retains every consumer's metadata, including its
layer and dimension. The `@global` inventory scope is reserved for shared
model services. Tool and memory consumers share a service-owned handle when
they use the same model. Vector-store ingestion owns an independent handle
until its workers drain; that handle can share the same physical resource.

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

Local classifiers default to 512 tokens. Set a deployment's `input.max_tokens`
to increase the budget for a compatible checkpoint and graph; for example,
`32768` enables a 32K budget on the native Vela CPU path. On AMD, also select
and qualify a graph and execution provider that support that length; see the
[AMD 32K options](../amd-rocm.md#optional-32k-domain-and-factcheck-on-rocm).

For an ordinary Domain, FactCheck, Feedback, or embedding encoder, edit the
existing deployment's `input` block to process up to 32,768 tokens and truncate
longer input before inference:

```yaml
global:
  model_catalog:
    deployments:
      vela-domain:
        # Keep this deployment's artifact, provider, device, and precision.
        input:
          max_tokens: 32768
          overflow: truncate
```

This is a fragment, not a complete deployment. Keep the binding pointed at the
same deployment, or create a separate deployment when recipes need different
input policies. The budget includes tokenizer special tokens. `truncate` keeps
the beginning of the input within that budget and preserves the tokenizer's
special-token envelope. At or below the limit, the complete input is processed.
Above it, these encoders return their normal result with input-usage metadata
marking the truncation. `overflow: reject` explicitly rejects oversized input;
it remains the default when the policy is omitted.

Encoder budgets cover **input only**. Classification scores and embedding
vectors do not reserve generated output tokens. Truncating the encoder's input
does not modify the original chat messages, system instructions, or tool calls
sent to the generation backend. A backend configured with
`--max-model-len 32768` separately budgets its tokenized prompt **plus generated
output**, including chat-template and tool overhead. Encoder truncation alone
does not make an over-budget generation request eligible for that backend.

Do not apply this truncation policy to every binding. Safety scans, token-span
coverage, calibrated operating points, and rerankers have their own completeness
contracts. Keep their required windowing or rejection policies; a partial scan
must not be treated as a complete safety result.

Long-input CPU inference can be substantially slower. Choose the smallest
budget that covers your workload and measure both quality and latency. For
scanning local risks across a long request, see
[Safety input policies](safety.md#native-classifier-context). Embedding signals
also have a separate [full-context setting](embeddings.md#input-policy).

Validate the edited configuration, restart through `vllm-sr serve`, and test
short input, exactly-at-budget input, and over-budget input using the model's
actual tokenizer. Check original and processed token counts and the truncation
flag; character counts and a generation model's tokenizer are not substitutes.
Use [Route Preview](lifecycle-diagnostics.md#inspect-the-executed-path) to verify
signal execution separately from a real chat request's generation budget.

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

Keep locally rewritten graphs in a separate artifact directory, with their
compatible tokenizer, configuration, and label mappings. Record the source
revision, export settings, precision changes, and file hashes alongside the local
artifact. Omit the deployment's upstream `revision` pin for these locally
derived bytes; that pin verifies the published snapshot, including the selected
head. Keep the original pinned directory unchanged. Complete unregistered local
artifacts are validated when the provider prepares them, without a registry
download.

For ONNX classifiers, use `provider: ort`, select the device, and set the
binding's `head` to the graph path, such as `onnx/model.onnx`. For GPU-specific
graphs, follow the model's runtime configuration. The Router rejects unavailable
GPU providers and CPU fallback.

Bindings belong to a recipe. Put them in that recipe's `routing` block to change
its models independently. See the [configuration reference](../../api/configuration-schema.mdx)
and [model update guide](lifecycle-diagnostics.md#update-a-running-model).

## Advanced MIGraphX settings

Set `compilation_cache_dir` to reuse compiled models after a restart. This optional
setting requires `provider: ort`, a `migraphx:N` device, and a persistent, writable
absolute directory outside model directories. It is disabled by default.

A changed model, GPU, precision, or compiler can require new compilation. See
[AMD troubleshooting](lifecycle-diagnostics.md#amd-startup-problems) for settings
that conflict with deployment configuration.
