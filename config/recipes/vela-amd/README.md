# Vela AMD Model Card

## Overview

Run the Vela decision model and the Vela 1.0 Embedding on an AMD
GPU and connect the Router to an existing OpenAI-compatible vLLM backend. The public model name is `vela-auto`; the backend
serves `vela-default`. This recipe exposes real routing signals and enables
retrieval for document questions, with the decision model judging which
documents are relevant.

## Model details

The [config](config.yaml) pins every model to an immutable Hugging Face
revision.

| Models | Execution | Input policy |
| --- | --- | --- |
| vela3-fast (the decision model) | Built-in model runtime on `rocm:0`, default engine arguments | 8,192-token windows, including special tokens |
| Embedding | Built-in model runtime on `rocm:0`, FP32, default engine arguments | 32,768 tokens, including special tokens; reject overflow |

Every deployment is a `model_runtime` deployment: the Router starts the runtime
and the runtime loads each pinned package on the GPU and chooses its kernels.
Embedding uses layer 22 and dimension 768, with `full_context: true`.

## Intended use

Use this recipe to inspect Vela signals on AMD, add semantic routing to a local
backend, or keep the relevant retrieved documents before generation. It is a starting
configuration for one backend, not a claim that routing improves that backend's
answer quality. The decision model answers every routing signal in one call and
scans longer inputs in 8K windows. Standalone Embedding execution is
qualified through 32K.

## Routing behavior

- `knowledge` matches the phrase `Search my documents`. It recalls up to three
  chunks, asks the decision model which are relevant and injects the best two
  before calling `vela-default`. Recall uses `min_similarity: 0` and keeps up
  to `top_k` candidates for the judge.
- `observe` combines the configured learned signals and calls the same backend.
  Requests without a matching decision also use the configured default backend.

The jailbreak rule detects prompt attacks; the safety rule detects unsafe
content. Their meanings are independent. The jailbreak rule matches at 0.5 and
reads the current user input; the PII rule includes history. Rules that write
no predicate use the thresholds calibrated for vela3-fast.

## Requirements

Use the ROCm Router image (it ships the model runtime with ROCm PyTorch), a
supported AMD GPU, device access through `/dev/kfd` and `/dev/dri`,
and enough memory for the three loaded models and the chosen concurrency.
The runtime downloads the pinned packages on first start; size the startup
budget from actual hardware measurements.

Start your generation backend separately with `--served-model-name vela-default`.
It must be reachable from the Router as `http://vllm:8000`. For a Docker backend,
attach it to `vllm-sr-network` with network alias `vllm`; otherwise edit the
provider endpoint in the config. Verify its `/v1/models` and a direct chat
request first. The recipe does not provision a generation model.

## Data handling and safety

The models run in the Router's model runtime on the same host. Chat text and retrieved context are
sent to the configured backend. Uploaded documents and vectors use local stores;
the example's in-memory vector index does not survive a restart. The CLI mounts
its workspace state at `/app/.vllm-sr`, including the files.

This recipe observes risk signals; it does not block requests merely because a
risk label matches. Configure your application's enforcement policy before
using those labels as a guardrail. Decisions use
`on_unknown: fail_request` for unresolved conditions. A resolved branch of the
observation route's `OR` can still select that route, so inspect `signal_errors`
alongside scores. Retrieval failures block the knowledge route. Preview returns the supplied text and diagnostic values, so handle its
responses as application data.

## Quick start

Install the CLI using the [installation guide](https://vllm-sr.ai/docs/installation/),
connect the backend described above, and download the recipe:

```bash
curl --fail --location --output vela-amd.yaml \
  https://raw.githubusercontent.com/vllm-project/semantic-router/main/config/recipes/vela-amd/config.yaml
vllm-sr config validate --config vela-amd.yaml
vllm-sr serve --platform rocm --config vela-amd.yaml
```

`--platform rocm` selects the AMD image and device access; the recipe's explicit
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
updated config. Follow [RAG sources](../../../website/docs/tutorials/plugin/context.md#rag-sources)
and [vector stores](https://vllm-sr.ai/docs/tutorials/global/stores-and-tools).
Send a real chat request beginning with `Search my documents` after indexing.
Preview shows the selected context plugin but does not run retrieval or the
relevance judge.

## Evaluation

[Maintained probes](probes.yaml) check both routes, the public entrypoint, text
and message requests, semantic matching and plugin selection on a running AMD
Router. Runtime qualification covers full-signal 8K execution and standalone
Embedding execution through 32K, including padded batches. These are
execution and numerical checks, not retrieval or classification quality scores.

Inspect actual per-signal timing in Preview's `metrics`. For retrieval, inspect
the real chat trace's retrieval latency and the relevance judge's
`vsr_match_judge_total{feature="rag"}` counter.
Measure cold startup, warm latency and memory on the intended hardware.

The [98x routing paper](https://arxiv.org/abs/2603.12646) reports latency and
memory for a different classifier and prompt-compression setup.
[AMD installation](https://vllm-sr.ai/docs/installation/amd-rocm#compare-with-the-98x-paper-setup)
compares its settings with this recipe.

## Limitations

The decision model reads a long input in 8K windows, so a signal that spans
two windows is judged per window.

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
