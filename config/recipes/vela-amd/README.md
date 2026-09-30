# Vela AMD Model Card

## Overview

Run all ten Vela task models on an AMD GPU and connect the Router to an existing
OpenAI-compatible vLLM backend. The public model name is `vela-auto`; the backend
serves `vela-default`. This recipe exposes real routing signals and enables
retrieval with neural reranking for document questions.

## Model details

The [config](config.yaml) pins every task artifact to an immutable Hugging Face
revision. The shared Encoder is a training parent and is not loaded separately.

| Models | Execution | Input policy |
| --- | --- | --- |
| Embedding, Reranker | ORT ROCm with CK FlashAttention, native graph precision | 32,768 tokens, including special tokens; reject overflow |
| Guard | ORT ROCm, native precision, `onnx/model_rocm_8k.onnx` | 8,192 tokens, including special tokens; reject overflow |
| Domain, Safety, PII, FactCheck, Feedback, Modality | ORT MIGraphX, native precision | 8,192 tokens, including special tokens; reject overflow |
| Hazard | ORT MIGraphX, native precision | Artifact-bound 2,048-token overlapping windows within a 32,768-token logical budget |

Embedding uses layer 22 and dimension 768, with `full_context: true`.
Both representation bindings select `head: onnx/model_fa.onnx` and deployments
select `custom_ops_profile: ck_flash_attention`, `device: rocm:0` and
`precision: native`. The Reranker's `pair_scorer` is exactly layer 22, dimension
768. The GPU provider must execute the model without CPU fallback.

The published CK graphs also contain Embedding exits at layers 3, 6 and 11 and
Reranker exits at layers 3, 6, 11 and 22 with dimensions 64, 128, 256, 512 and
768. For a reduced Reranker, choose `onnx/model_fa_layer_N_dim_D.onnx` and set
both `pair_scorer` coordinates to those same values. The full 22/768 selection
uses `onnx/model_fa.onnx`. Embedding resolves matching flat
`onnx/model_fa_layer_N.onnx` companions for enabled consumers. Keep the CK graph
family consistent; portable and FP16 graph variants are different selections.

## Intended use

Use this recipe to inspect Vela signals on AMD, add semantic routing to a local
backend, or rerank retrieved documents before generation. It is a starting
configuration for one backend, not a claim that routing improves that backend's
answer quality. Keep the all-signals request within the 8K classifier budget.
Standalone Embedding and Reranker execution is qualified through 32K; this does
not establish 32K AMD support for the complete classifier pipeline.

## Routing behavior

- `knowledge` matches the phrase `Search my documents`. It retrieves up to three
  chunks, reranks them and injects the best two before calling `vela-default`.
  Candidate retrieval uses `similarity_threshold: 0.0` and selects up to
  `top_k` candidates before neural reranking.
- `observe` combines the configured learned signals and calls the same backend.
  Requests without a matching decision also use the configured default backend.

Guard detects prompt attacks; Safety detects unsafe content. Their meanings are
independent. Guard retains threshold 0.5 and checks the current user input;
PII includes history. FactCheck uses 0.95 and Feedback uses 0.7. Hazard uses the
pinned operating point's twelve independent thresholds and window policy.

## Requirements

Use an AMD Router image containing ORT ROCm, MIGraphX and the CK custom operator
library, a supported AMD GPU, device access through `/dev/kfd` and `/dev/dri`,
and enough memory for ten loaded task models and the chosen concurrency.
The CLI downloads the pinned artifacts. Cold MIGraphX compilation can take
longer than startup on a cached deployment; size the startup budget from actual
hardware measurements.

Start your generation backend separately with `--served-model-name vela-default`.
It must be reachable from the Router as `http://vllm:8000`. For a Docker backend,
attach it to `vllm-sr-network` with network alias `vllm`; otherwise edit the
provider endpoint in the config. Verify its `/v1/models` and a direct chat
request first. The recipe does not provision a generation model.

## Data handling and safety

The ten task models run inside the Router. Chat text and retrieved context are
sent to the configured backend. Uploaded documents and vectors use local stores;
the example's in-memory vector index does not survive a restart. The CLI mounts
its workspace state at `/app/.vllm-sr`, including the compilation cache and files.

This recipe observes risk signals; it does not block requests merely because a
risk label matches. Configure your application's enforcement policy before
using those labels as a guardrail. Guard and PII retain `on_error: block`; decisions use
`on_unknown: fail_request` for unresolved conditions. A resolved branch of the
observation route's `OR` can still select that route, so inspect `signal_errors`
alongside scores. Retrieval failures block the knowledge route. Preview returns the supplied text and diagnostic values, so handle its
responses as application data.

## Quick start

Install the CLI using the [installation guide](https://vllm-sr.ai/docs/installation/installation),
connect the backend described above, and download the recipe:

```bash
curl --fail --location --output vela-amd.yaml \
  https://raw.githubusercontent.com/vllm-project/semantic-router/main/config/recipes/vela-amd/config.yaml
vllm-sr config validate --config vela-amd.yaml
vllm-sr serve --platform amd --config vela-amd.yaml
```

`--platform amd` selects the AMD image and device access; the recipe's explicit
deployments select which models use the GPU. It does not override an authored
CPU deployment. On a shared machine, select the Router's visible GPU with
`VLLM_SR_AMD_ROUTER_VISIBLE_DEVICES`; deployment index `0` refers to that visible
device. Keep generation capacity separate from the Router's measured needs.

The CLI's default startup timeout is 1,800 seconds. If measured cold compilation
needs longer, pass a positive `--startup-timeout SECONDS`. A timeout leaves the
owned containers running so their logs and readiness remain inspectable.

Once `/ready` succeeds, inspect actual signals and their timings:

```bash
curl --fail http://localhost:8080/ready
curl --fail 'http://localhost:8080/api/v1/routing/preview?trace=true' \
  -H 'Content-Type: application/json' \
  -d '{"model":"vela-auto","text":"Debug this Python program and fix its error."}' \
  | jq '{decision_result, signal_confidences, signal_values, signal_errors, metrics, eval_trace}'

curl --fail http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"vela-auto","messages":[{"role":"user","content":"Explain Python tuples briefly."}],"max_tokens":128}'
```

For RAG, create a vector store, upload and index documents, then replace
`vs_your_documents` in the plugin with the returned store ID and activate the
updated config. Follow [neural reranking](https://vllm-sr.ai/docs/tutorials/plugin/rag#neural-reranking)
and [vector stores](https://vllm-sr.ai/docs/tutorials/global/stores-and-tools).
Send a real chat request beginning with `Search my documents` after indexing.
Preview shows the selected RAG plugin but does not run retrieval or reranking.

## Evaluation

[Maintained probes](probes.yaml) check both routes, the public entrypoint, text
and message requests, semantic matching and plugin selection on a running AMD
Router. Runtime qualification covers full-signal 8K execution and standalone
Embedding/Reranker execution through 32K, including padded batches. These are
execution and numerical checks, not retrieval or classification quality scores.

Inspect actual per-signal timing in Preview's `metrics`. For retrieval, inspect
the real chat trace's candidate counts, reranker identity, scores and latency.
Measure cold startup, warm latency and memory on the intended hardware.

The [98x routing paper](https://arxiv.org/abs/2603.12646) reports latency and
memory for a different classifier and prompt-compression setup.
[AMD installation](https://vllm-sr.ai/docs/installation/amd-rocm#compare-with-the-98x-paper-setup)
compares its settings with this recipe.

## Limitations

All-signals Preview is bounded by the 8K classifier deployments even though the
representation models and Hazard's logical budget are larger. PII scanning and
Hazard window aggregation are different from a whole-document classifier.
Reranker capacity includes the query, document and pair special tokens.

Preview defaults to a 120-second request budget and 16 admitted workers. Choose
`global.services.api.routing_preview` limits from measured workload and hardware
capacity. Timeout returns HTTP 504 and saturation returns HTTP 429; a timed-out
native worker retains its admission slot until it finishes.

This recipe cannot run in a CPU-only image. Larger token limits, smaller exits,
different precision and another GPU require measurements. Changing embedding
identity requires reindexing stored documents. Signal predictions can be wrong,
and the single generation backend determines answer capabilities.

## References

- [Vela models](https://vllm-sr.ai/docs/tutorials/global/vela-models)
- [AMD installation](https://vllm-sr.ai/docs/installation/amd-rocm)
- [Embedding runtime](https://vllm-sr.ai/docs/installation/runtime/embeddings)
- [In-process model bindings](https://vllm-sr.ai/docs/installation/runtime/in-process)
- [Recipe authoring and conformance](../CONFORMANCE.md)
