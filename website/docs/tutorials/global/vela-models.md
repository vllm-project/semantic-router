# Vela Router Models

## Overview

[Vela 1.0](https://huggingface.co/collections/llm-semantic-router/vela-10)
is a family of fourteen published model checkpoints for intelligent routing.
The Router registry currently includes the Vela 307M encoder base and ten task
models covering routing, prompt protection, content safety, retrieval and
reranking. Each registered release is pinned to an immutable revision.

| Model | Role |
| --- | --- |
| Encoder | Shared base for adapting new routing tasks |
| Domain | Request subject across 14 domains |
| Guard | Prompt injection and jailbreak detection |
| Safety | Unsafe content detection |
| Hazard | Twelve independent content risk categories |
| PII | Personal information spans across 17 entity types |
| FactCheck | Whether a request needs factual verification |
| Feedback | Four feedback types and a neutral `NO_FEEDBACK` class |
| Modality | Text, image, or combined output intent from a written request |
| Embedding | Multilingual retrieval and semantic matching |
| Reranker | Relevance scoring for query-document pairs |

The full model name is `Vela-1.0-Encoder-307M`, followed by the task suffix.
Modality classifies the requested output modality from text.
FactCheck requests verification and does not verify the truth of an answer.

The public collection also includes **Halu** for answer evidence-support
detection and **Omni Nano / Omni Mini** for text, image and audio embeddings.
These three checkpoints are available for direct use and integration work;
they are not yet the Router's default hallucination or multimodal components.
See the [Vela 1.0 release announcement](/blog/introduce-vela) for the full family.

## What Problem Does It Solve?

A shared model family supplies task-specific signals and retrieval components
with explicit model identities, input budgets and serving policies.

## When to Use

Use Vela for built-in routing tasks or adapt the shared Encoder for a new task.
Choose input length and representation size against your workload's quality
and latency requirements.

## Omni checkpoints

The September 18 releases provide separate text, image and audio encoders in a
shared embedding space. These checkpoints remain direct-use models rather than
the Router's default multimodal components.

| Checkpoint | Total parameters | Output dimensions | Text limit | Text backbone and readout |
| --- | ---: | ---: | ---: | --- |
| [Omni Nano](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Nano/blob/d8ac5b5ac2274a501fc61aeb5be70cec1855a806/README.md) | 135.4M (135,383,808) | 384 | 512 tokens | Frozen GIST-small; CLS readout |
| [Omni Mini](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/2aecd547915f5cffe68ba3c2e2c3a678de8b193e/README.md) | 1.33B (1,332,891,200) | 768 | 32,768 tokens | Qwen3-Embedding-0.6B; last-token Matryoshka readout |

Parameter counts include all three modality branches. Text limits include
special tokens; the public `encode_text` rejects inputs above the limit instead
of silently truncating them. Mini strips surrounding whitespace, uses the native
Qwen tokenizer without an added prompt, and rejects text that tokenizes to zero
tokens. Its text readout normalizes the full 1024-dimensional last nonpadding
token, takes the first 768 dimensions, then normalizes that prefix. Nano uses an
identity text projection after CLS readout. Both return L2-normalized vectors.
Audio input is mono, 16 kHz, up to 30 seconds per waveform.

The latest cards compare routing and cross-modal retrieval with the
[original small](https://huggingface.co/llm-semantic-router/multi-modal-embed-small/tree/fdf8e01b7b0f3a69ac1ac8e2a64dcb1ede177ba4)
and [original large](https://huggingface.co/llm-semantic-router/multi-modal-embed-large/tree/e21cde3ccc414c56f504b322662f42c603a939ee)
models. Scores are 0–100; each cell shows original → current:

| Metric | Original small → Nano | Original large → Mini |
| --- | ---: | ---: |
| Banking77, accuracy | 70.42 → 87.99 | 75.78 → 86.56 |
| MASSIVE English, accuracy | 65.95 → 81.97 | 72.31 → 80.96 |
| COCO, image → text, R@1 | 40.83 → 60.87 | 42.53 → 67.44 |
| COCO, text → image, R@1 | 30.18 → 55.82 | 35.04 → 61.60 |
| LibriSpeech, audio → text, R@1 | 4.21 → 17.16 | 56.99 → 86.37 |
| LibriSpeech, text → audio, R@1 | 9.58 → 25.44 | 78.58 → 95.36 |

Both sides use the same held-out pools, reused across releases, with a 128-token
text cap. Classification uses labeled TRAIN prototypes; retrieval uses complete
candidate pools and all matching positives. This differs from official MTEB
classification probes. Full metrics and uncertainty are in the linked evaluations.

Separately, current full-panel results use MTEB 2.21.0. The original small and
large models have not been evaluated under this complete-panel protocol:

| Evaluation | Nano | Mini |
| --- | ---: | ---: |
| English v2, 41 tasks, mean task score | 64.88 | 63.49 |
| MAEB audio-only, 19 tasks, mean task score | 39.77 | 42.53 |

Nano's scores apply to the smaller inference package through documented
[weight and computation equivalence](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Nano/blob/d8ac5b5ac2274a501fc61aeb5be70cec1855a806/benchmarks/inference-equivalence.json).
Mini's English panel was measured on its equivalent frozen Qwen text component,
as recorded in the [component report](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/2aecd547915f5cffe68ba3c2e2c3a678de8b193e/benchmarks/component-equivalence.json).
Neither English panel is a new full-checkpoint run; Mini's audio panel was
freshly evaluated on the released numerical model.

These panels do not cover every language or image task. In the updated
[task-level size–quality comparisons](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Nano/blob/d8ac5b5ac2274a501fc61aeb5be70cec1855a806/benchmarks/pareto-methodology.md),
Nano is on the observed NMSQA frontier at **63.28 max AP** and Mini is on the
SIBFLEURS frontier at **40.06 accuracy**. Nano scores **64.89 V-measure** on ArXiv
and **13.80 V-measure** on VehicleSoundClustering, but smaller independently
measured peers place it below those two frontiers. The horizontal axis counts
all model parameters, and peer protocols vary. These plots establish neither
latency nor overall benchmark leadership. See the pinned
[Nano evaluation](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Nano/blob/d8ac5b5ac2274a501fc61aeb5be70cec1855a806/benchmarks/EVALUATION.md) and
[Mini evaluation](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/2aecd547915f5cffe68ba3c2e2c3a678de8b193e/benchmarks/EVALUATION.md)
for all tasks, regressions and protocols.

## Defaults and input budgets

Built-in Domain, Guard, Safety, PII, FactCheck, Feedback and semantic Embedding
now use Vela. The reference configuration also selects Vela Modality, Hazard and
Reranker. Only models required by a recipe are loaded. The base encoder is a
training parent and is not loaded as an additional routing signal.

Default operating thresholds are **0.5** for Guard, **0.95** for FactCheck and
**0.7** for Feedback. `NO_FEEDBACK` emits no feedback match. Safety is independent
of Guard, so an unsafe content request need not be classified as a prompt attack.
Hazard uses per-label thresholds from its artifact-bound operating point; a
single threshold does not represent its published decision policy.

An input budget is a deployment choice. A module `max_sequence_length` of `0`
keeps the conservative 512-token policy. Embedding defaults to 22 layers,
768 dimensions and `full_context: false`. Explicit model bindings can accept
up to **32,768 tokens**, including special tokens, with `overflow: reject`.
A rejected input is not silently shortened.

## Configuration

The following excerpt binds Domain to a CPU deployment with a 32K budget. Apply
it to an existing configuration containing providers, signals and decisions.

```yaml
routing:
  model_bindings:
    domain_classifier:
      deployment: vela-domain
      contract: label_distribution.v1
      adapter: modernbert
      mapping_path: models/Vela-1.0-Encoder-307M-Domain/category_mapping.json

global:
  model_catalog:
    deployments:
      vela-domain:
        artifact: models/Vela-1.0-Encoder-307M-Domain
        provider: candle
        device: cpu
        precision: fp32
        input:
          max_tokens: 32768
          overflow: reject
```

Use the same deployment and consumer-binding structure for other classifiers.
PII returns `token_spans.v1`. Embedding uses the `mmbert` adapter and
`embedding.v1`; Reranker uses `vela_reranker` and `relevance_scores.v1`. These
adapter names describe inference architectures and are independent of release
names. See [in-process inference](/docs/installation/runtime/in-process) for the full contract.

For Hazard, bind the independent classifier contract `label_scores.v1` and pin
`operating_point.json` with its SHA-256. The policy binds the weights, tokenizer,
execution settings, overlapping windows and twelve thresholds. Decisions select
labels without replacing those thresholds. The reference configuration includes
this complete pattern.

PII's overlapping scan and Hazard's windowed policy differ from whole-document
classification. Select the policy appropriate to the task, and measure latency
with the input lengths your application will send.

## Inference engines and hardware

Native Vela artifacts run through Candle. CPU execution is validated, including
32K inputs. CUDA remains an available Candle backend; NVIDIA performance must be
measured on the target hardware.

ONNX is the portable inference format for the ORT provider. The
[Vela AMD recipe](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/vela-amd/README.md)
explicitly binds all ten task models to AMD GPU execution: CK FlashAttention
through ROCm for Embedding and Reranker, a fixed 8K ROCm graph for Guard, and
MIGraphX for the other classifiers. It pins
each artifact and preserves the published operating policies. `--platform amd`
selects the AMD image and device access; it does not make every authored model
use a GPU or override an explicit CPU deployment.

The complete signal pipeline is measured with an **8K** input budget.
Standalone Embedding and Reranker execution is qualified through **32K**;
Hazard uses its qualified 2,048-token windows within a 32K logical budget.
These limits do not establish 32K AMD execution for all classifiers. Initial GPU
compilation and warm request latency are separate measurements.

For longer Domain and FactCheck requests, use the optional
[32K ROCm deployments](../../installation/amd-rocm.md#optional-32k-domain-and-factcheck-on-rocm).
They require more GPU memory and add latency for short inputs.

The Embedding and Reranker repositories include FP32 ONNX graphs with shared
external weights. The full representation uses `onnx/model.onnx`; reduced
representations require their matching trained layer or layer/dimension graph.
The downloader resolves a full-size selection from the model's encoder
configuration, so an explicit full selection can use the primary graph.
The CK variants use `onnx/model_fa.onnx` for the full 22-layer, 768-dimensional
representation. Select that exact `head`, `device: rocm:0`,
`custom_ops_profile: ck_flash_attention` and `precision: native`. Reranker's
`pair_scorer` must match the graph: reduced exits use
`onnx/model_fa_layer_N_dim_D.onnx` with the same layer and dimension. Embedding
uses matching `onnx/model_fa_layer_N.onnx` companions. These graphs share the
published external weights and retain the portable exports.

Replacing native weights also requires regenerating the corresponding ONNX
artifacts before publishing the update.

All models expose their supported input length, usage and comparable evaluation
results in their model cards. Published comparisons use the previous mmBERT
family on matched evaluation data. Quality scores, maximum accepted input length
and inference performance measure different properties.

## Verify live routing and reranking

For the Vela AMD recipe, download and validate the complete configuration, then
start it with the AMD image. Connect an existing vLLM backend served as
`vela-default` at `vllm:8000`, as explained in the recipe's Model Card.

```bash
curl --fail --location --output vela-amd.yaml \
  https://raw.githubusercontent.com/vllm-project/semantic-router/main/config/recipes/vela-amd/config.yaml
vllm-sr config validate --config vela-amd.yaml
vllm-sr serve --platform amd --config vela-amd.yaml
```

Route Preview returns actual signal values, decisions and per-signal latency.
Keep its input within the recipe's 8K classifier budget:

```bash
curl --fail 'http://localhost:8080/api/v1/routing/preview?trace=true' \
  -H 'Content-Type: application/json' \
  -d '{"model":"vela-auto","text":"Help me debug this Python program."}' \
  | jq '{decision_result, signal_confidences, signal_values, signal_errors, metrics, eval_trace}'
```

Use the public model name declared by your entrypoint. Check `/ready` before
sending requests and inspect the returned signal values and evaluation trace.

Reranking executes inside the vectorstore RAG plugin after routing. Bind
`rag.reranker` and enable `rerank` as described in
[neural reranking](/docs/tutorials/plugin/rag#neural-reranking). Verify it through
a real chat request with indexed documents. Its request trace records candidate
count, reranker identity, relevance scores and reranking latency; routing Preview
does not execute the RAG plugin.

## Preserve an earlier deployment

Explicit older model paths and aliases retain their original repositories.
Select the matching mapping file, threshold and input policy when reproducing
an earlier classifier. Upgrading to Vela does not rewrite explicit model choices.

Changing embedding weights or representation changes the vector space. Response
caches and memory isolate data by representation identity. Existing vector stores
require compatible embeddings or reindexing; old data is not silently adopted
or deleted. See [stores and tools](./stores-and-tools.md).
