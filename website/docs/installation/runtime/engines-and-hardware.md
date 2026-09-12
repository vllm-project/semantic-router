---
title: Engines and Hardware
description: Actual Router execution engines, task support, device selection, and build requirements.
---

# Engines and hardware

Choose an engine that implements the task your recipe consumes. Engine
availability, model architecture, graph shape, precision, and device must all
agree at preparation time.

## Engines used by the Router

| Engine | Current Router uses | Configuration |
| --- | --- | --- |
| Candle | Sequence and token classifiers, embeddings, local grounding/NLI, and MLP model selection | `provider: candle` for model bindings; MLP has its existing model-selection configuration |
| ONNX Runtime (ORT) | mmBERT sequence/token classifiers and text or multimodal embedding graphs | `provider: ort` |
| HTTP connector | External classification, scoring, embeddings, and supported chat-based tasks | `provider: http` or the task's existing named `backend` |
| Linfa through `ml-binding` | KNN, K-means, and SVM selection over embedding features | `global.router.model_selection.ml` and the decision algorithm |
| `nlp-binding` | BM25 and N-gram keyword matching | Keyword signal configuration |
| Go implementations | Heuristics, policy evaluation, and NLP prompt-compression orchestration | The corresponding signal, algorithm, or module |

MCP classification is a separate [external tool protocol](external.md#mcp-classification),
not a new `provider` value. The downstream LLM can use vLLM or another supported
provider independently of these Router engines.

## Deployment device and precision matrix

| Deployment engine | Device values in the owned runtime | Precision in that path | Requirements and limits |
| --- | --- | --- | --- |
| Candle | `cpu` | `native` or `fp32` for encoder/classifier/embedding tasks | Native CGo library and compatible checkpoint |
| Candle | `cuda:0` | Encoder/classifier/embedding execution is float32 | CUDA-enabled Candle build and supported adapter; not every model family supports every device |
| Candle | `metal:0` | Encoder/classifier/embedding execution is float32 | Metal-enabled build; BERT and merged BERT/LoRA token paths reject unsupported Metal execution |
| ORT | `cpu` | `native`, preserving graph precision | Matching ONNX Runtime library and complete graph artifacts |
| ORT | `migraphx:N` | `native`, or explicit `fp16` graph conversion | MIGraphX-enabled ORT build, compatible ROCm/MIGraphX libraries, and supported graph |
| HTTP | Omit device and precision | Controlled by the external service | Reachable compatible API and credentials |

`N` in `migraphx:N` is an explicit nonnegative device index. The current
owned Candle adapter implements only `cuda:0` and `metal:0`; nonzero Candle
indices fail preparation even though the configuration schema accepts index
syntax. Explicit local deployments default to `cpu` and `native`. Existing module configurations without an explicit binding retain
the image's default engine and their `use_cpu` setting. The ROCm image uses
ORT for those implicit models; ordinary builds use Candle. An explicit
recipe deployment selects its engine independently. Model provisioning uses
the same default-engine choice to fetch the required artifact format. The schema recognizes precision names before adapter preparation;
that does not mean every engine implements every precision. For example, the
ORT owned adapter rejects `fp32` as an override: use a native FP32 graph with
`precision: native`. Candle encoder adapters do not implement a blanket FP16
conversion by accepting `precision: fp16`.

The owned ORT path disables CPU fallback for a requested MIGraphX execution
provider. Missing libraries, incompatible provider options, or unsupported
execution fail preparation. Provider registration alone does not prove that
inference ran on a GPU.

The maintained ROCm 7.0 images pin ORT 1.22.1 and MIGraphX 2.13 and set
`MIGRAPHX_MLIR_USE_SPECIFIC_OPS=~attention` before model preparation. This
excludes the vendor's MLIR attention fusion, which produced nonfinite outputs
for the maintained mmBERT classifier; other GPU MLIR operations remain enabled.
It is a fixed startup compiler policy, not an instance-level environment change
or an automatic retry. It does not change graph precision or permit CPU
fallback. Revalidate the policy when changing the vendor stack.

The validated classification path uses the standard `onnx/model.onnx` graph,
`precision: native`, and a fixed execution budget. The owned loader prefers
the standard graph when no explicit head is supplied. The maintained
`model_sdpa_fp16.onnx` graph still contains an `IsNaN` operation unsupported
by this ORT provider's capability check; do not treat it as validated by the
standard-graph result. Explicit graph selection must pass preparation on its
own. See [Preparation and resource ownership](lifecycle-diagnostics.md#preparation-and-resource-ownership)
for compilation and input-budget behavior.

For owned MIGraphX deployments, unset these process-level vendor overrides:

```text
ORT_MIGRAPHX_FP16_ENABLE
ORT_MIGRAPHX_BF16_ENABLE
ORT_MIGRAPHX_FP8_ENABLE
ORT_MIGRAPHX_INT8_ENABLE
ORT_MIGRAPHX_MODEL_CACHE_PATH
```

Any nonempty value causes preparation to fail. Precision overrides can replace
the instance's declared precision, and that vendor cache can load a compiled
model without precision in its cache identity. Use deployment `precision`
instead. The Router does not clear or rewrite the process environment, and CPU
execution is unaffected. Logging and ordinary tuning environment variables
are not covered by this restriction. The maintained images and serve flow do
not set these overrides.

Existing CUDA, ROCm, OpenVINO, and Intel build paths are separate platform
integrations; their presence in Cargo features or an image is not an additional
owned deployment device value. In particular, `provider: ort` does not currently
accept `cuda:N`, `rocm:N`, or `openvino:N`, and `provider: openvino` is not a
canonical deployment engine. Follow the platform's installation instructions
for its existing integration and validate its actual capabilities.

The existing OpenVINO primary embedding-rule path remains separate. It does
not provide a new owned OpenVINO adapter or imply support for cache/windows or
the embedding diagnostic API. Independent BERT consumers can still use their
own Candle providers. An explicit embedding binding takes precedence over the
legacy primary backend selector.

## Tasks by engine

“Available” means an adapter exists. Its model and artifact still have to pass
preparation.

| Router task | Candle | ORT | External HTTP |
| --- | --- | --- | --- |
| Domain and generic sequence classification | Available | mmBERT graph | `http_classify` |
| Prompt guard | Sequence classifier | mmBERT graph | `http_classify`, or categorical `http_chat` |
| PII | Token classifier | mmBERT token graph | `http_classify` token spans |
| Fact-check and feedback classification | Available | Compatible mmBERT graph | No adapter |
| Output-modality classifier | Three-label sequence classifier | Compatible mmBERT graph | No adapter |
| Grounded hallucination spans | Dedicated local detector | No adapter | Dedicated `http_chat` detector |
| NLI explanation | Dedicated sentence-pair classifier | No adapter | No adapter |
| Text embeddings | BERT, Qwen3, Gemma, mmBERT, multimodal | mmBERT and multimodal graphs | OpenAI-compatible embeddings |
| Image/audio embeddings | Actual modalities of a multimodal checkpoint | Multimodal graphs | No adapter |
| Complexity score/distribution override | No dedicated binding adapter | No dedicated binding adapter | `http_classify` |
| Generic LLM scored extraction | No local binding adapter | No adapter | `http_chat` with the generic classifier JSON contract |

Default embedding-based complexity is a separate use of the embedding engine;
it does not require an external scorer.

Candle's low-level binding also contains Qwen3Guard and generative classifier
APIs. Those APIs are not currently registered as local Router model-binding
tasks. Their existence does not make a Qwen checkpoint a sequence classifier.
Use the supported external task protocol when serving such a model separately.

## Checkpoint families

| Checkpoint / adapter family | Candle task support in the owned path | Important boundary |
| --- | --- | --- |
| ModernBERT and mmBERT, including `mmbert32k` / `mmbert-32k` | Sequence and token classification; mmBERT embedding | Classification remains capped at 512; compatible modern heads can share an encoder |
| BERT | Sequence/token classification and text embedding | The BERT paths do not support Metal |
| DeBERTa | Sequence classification | A plain sequence checkpoint is not automatically a grounded detector or NLI task |
| Merged BERT LoRA (`bert_lora`) | Sequence and token classification | Requires complete merged weights; no claim of applying separate adapter deltas |
| `lora_token` | Token classification | Uses its actual token model format and supported devices; no generic encoder-sharing claim |
| Qwen3 / Gemma embedding checkpoints | Text embedding | These embedding adapters do not enable a local chat/generative classification task |
| Multimodal embedding checkpoint | Its actual text/image/audio encoders | Each modality is checked independently |

The owned ORT path uses exported mmBERT sequence/token graphs, mmBERT
embedding graphs, or the supported multimodal graph layout. It does not infer
support for another architecture from an `.onnx` extension. Full model files,
compatible heads, tokenizer metadata, and declared labels remain required.

## MLP and NLP execution

The Router's MLP selector currently constructs and loads its Candle selector on
CPU. The low-level binding has GPU constructors, but the current
`global.router.model_selection.ml.mlp.device` field does not connect those
constructors to Router selection. Do not use it as evidence of GPU execution.
MLP consumes a trained selector artifact and embedding features; it is not
configured with `routing.model_bindings`.

KNN, K-means, and SVM use their owned `ml-binding` handles. BM25/N-gram use owned
`nlp-binding` handles and return matching scores in their own units. These CPU
algorithms are not transformer checkpoints and their scores are not softmax
probabilities. [In-process inference](in-process.md#ml-selectors-and-nlp)
shows where their existing configuration belongs.

## Build and verification

Use the existing local-image workflow:

```bash
make vllm-sr-dev
vllm-sr serve --config config.yaml --image-pull-policy never
```

Select the documented build/serve platform for the host. Both Candle and ORT
libraries remain separately linked; do not replace the Candle Go module with
the ORT module to choose one engine. CPU dynamic ORT builds use the `dynamic`
feature; AMD builds use `migraphx-dynamic`. `ORT_DYLIB_PATH` must identify the
actual ONNX Runtime shared library, with its provider and ROCm dependencies
available to the dynamic loader. The maintained AMD image pins these together.
A non-CGo or unavailable-backend stub cannot run local inference.

Real CPU tests cover owned handles and maintained mmBERT intent/embedding
paths; MIGraphX checks cover the maintained graphs with provider profiles.
This is not a certification of every architecture/device/precision combination.
Candle CUDA/Metal availability in code is distinct from measured support for
your checkpoint. Keep the source revision, model revision, runtime versions,
provider profile, and tolerances with your own deployment evidence.
