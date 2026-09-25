---
title: AMD ROCm
description: Connect an AMD vLLM backend and run Vela routing models on AMD GPUs.
---

# Deploy with AMD ROCm

Semantic Router can run on CPU while vLLM serves the selected model on AMD
Instinct GPUs. This guide starts one ROCm backend, verifies it directly, and
then connects it to the local Router stack. To also run all ten Vela routing
task models on AMD, use the [Vela AMD recipe](#run-vela-routing-models-on-amd)
below.

The example uses one checkpoint behind several served-model aliases so the
maintained `balance` recipe can exercise its routing lanes. That is useful for
functional evaluation, but it does not turn one checkpoint into several models.
In production, bind each logical provider to a backend with the capabilities,
capacity, and operating cost declared by the recipe.

## Prerequisites

- a host and GPU supported by the ROCm version in the selected vLLM image;
- Docker with access to `/dev/kfd` and `/dev/dri`;
- enough GPU memory for the model, context limit, and concurrency settings;
- a persistent Hugging Face cache directory; and
- network access to download the model, unless it is already cached.

Confirm the devices are visible before starting a large download:

```bash
rocminfo | head
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  rocm/dev-ubuntu-24.04:latest rocminfo | head
```

Pin image digests and model revisions in controlled environments. The tags
below are readable examples, not an immutability guarantee.

## Start the vLLM backend

Create the network used by the local Router stack and choose a cache directory:

```bash
docker network inspect vllm-sr-network >/dev/null 2>&1 || \
  docker network create vllm-sr-network

export VLLM_HF_CACHE=/mnt/data/huggingface-cache
mkdir -p "$VLLM_HF_CACHE"
```

Start the reference backend:

```bash
docker run -d \
  --name vllm \
  --network vllm-sr-network \
  --restart unless-stopped \
  -p 8090:8000 \
  -v "$VLLM_HF_CACHE:/root/.cache/huggingface" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --shm-size=32g \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e VLLM_ROCM_USE_AITER_MHA=0 \
  --entrypoint python3 \
  vllm/vllm-openai-rocm:v0.17.0 \
  -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-122B-A10B-FP8 \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name \
      qwen/qwen3.5-rocm \
      google/gemini-2.5-flash-lite \
      google/gemini-3.1-pro \
      openai/gpt5.4 \
      anthropic/claude-opus-4.6 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --max-model-len 262144 \
    --language-model-only \
    --max-num-seqs 128 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.85
```

This command mounts only the model cache. Do not mount an entire home directory
into a model-serving container. The example also omits `SYS_PTRACE`, an
unconfined seccomp profile, and `--trust-remote-code`; add broader privileges or
remote model code only when a reviewed, pinned workload demonstrably requires
them.

Tune `--max-model-len`, `--max-num-seqs`, tensor parallelism, and GPU memory
utilization for the available hardware. A model that starts with smaller limits
may fail or evict useful cache when copied with these reference values.

## Verify the backend first

Wait for model loading to finish, then verify the backend independently of the
Router:

```bash
curl --fail http://127.0.0.1:8090/health
curl --fail http://127.0.0.1:8090/v1/models

curl --fail http://127.0.0.1:8090/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen/qwen3.5-rocm",
    "messages": [{"role": "user", "content": "Reply with: ready"}],
    "max_tokens": 16
  }'
```

Do not continue until the direct generation request succeeds. Router validation
checks routing configuration; it does not prove that a provider can generate.

## Install and configure Semantic Router

Install the CLI:

```bash
curl -fsSL https://vllm-sr.ai/install.sh | \
  bash -s -- --channel stable --mode cli --runtime skip --no-launch
```

For a simple one-model deployment, open the Dashboard at
`http://localhost:8700`, add an OpenAI-compatible backend at `vllm:8000`, and
activate the generated config.

To evaluate the maintained balance recipe, download it into the current
workspace instead of relying on a repository-relative path:

```bash
curl --fail --location \
  --output balance.yaml \
  https://raw.githubusercontent.com/vllm-project/semantic-router/main/config/recipes/balance/config.yaml

vllm-sr config validate --config balance.yaml
vllm-sr serve --config balance.yaml
```

The balance recipe expects the five aliases exposed by the example backend.
Read its [Model Card](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/balance/README.md)
for intended use, routing behavior, data handling, and limitations. Fork the
configuration before replacing aliases, thresholds, prices, or provider roles.

## Verify the routed path

Send a request through Envoy using the automatic entrypoint:

```bash
curl --fail --include http://127.0.0.1:8899/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Explain prefix caching briefly."}],
    "max_tokens": 64
  }'
```

Check that the response is successful and inspect the routing headers for the
selected decision and provider model. Use the recipe's maintained probes for
broader routing evaluation; use representative application requests to measure
answer quality and operating behavior on the actual deployment.

## Run Vela routing models on AMD

The [Vela AMD Model Card](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/vela-amd/README.md)
and complete config select GPU execution explicitly for all ten task models.
Embedding and Reranker use their pinned CK FlashAttention graphs through ROCm,
with native precision and no CPU fallback. Guard uses its fixed 8K ROCm graph;
the other classifiers use MIGraphX.
The complete signal pipeline is measured at 8K; standalone Embedding/Reranker
execution is qualified through 32K. Hazard retains its qualified 2,048-token
windows and 32K logical budget. These results do not establish 32K AMD execution
for every classifier.

Connect an existing OpenAI-compatible backend served with
`--served-model-name vela-default`. The config expects `http://vllm:8000`; attach
that backend to `vllm-sr-network` with network alias `vllm`, or edit the endpoint.
The earlier multi-alias example does not expose `vela-default` unless you add
that served name. Verify a direct request using that name before routing.
Reserve enough memory and compute for the Router alongside the generation
backend; use `VLLM_SR_AMD_ROUTER_VISIBLE_DEVICES` when selecting a Router GPU.
The deployment's device index `0` refers to its visible GPU.

```bash
curl --fail --location --output vela-amd.yaml \
  https://raw.githubusercontent.com/vllm-project/semantic-router/main/config/recipes/vela-amd/config.yaml
vllm-sr config validate --config vela-amd.yaml
vllm-sr serve --platform amd --config vela-amd.yaml
```

The platform flag selects the image and device access. Named deployments select
the actual providers and graphs; explicit CPU choices remain CPU choices.
Cold MIGraphX compilation can exceed a cached startup. The CLI waits up to 1,800
seconds by default; use `--startup-timeout SECONDS` if measurements justify a
longer bounded wait. A timeout leaves the owned containers available for logs
and readiness inspection.

For AMD Docker serves, the Router keeps its COMGR compiler cache below
`<state-root>/.vllm-sr/compiler-cache/`. The cache is private to the stack and
the immutable Router image, so a restart can reuse compiled kernels while a
new image or ROCm userspace version starts with a fresh cache. The state root
defaults to the directory containing the source config; stop the stack before
removing an old image's cache to reclaim disk space.

Once `/ready` succeeds, inspect the real signals and their timings:

```bash
curl --fail http://localhost:8080/ready
curl --fail 'http://localhost:8080/api/v1/routing/preview?trace=true' \
  -H 'Content-Type: application/json' \
  -d '{"model":"vela-auto","text":"Debug this Python program and fix its error."}' \
  | jq '{decision_result, signal_confidences, signal_values, signal_errors, metrics, eval_trace}'
```

With the default recipe, keep all-signals Preview within the 8K classifier
budget. Preview does not execute retrieval or generation. Follow the recipe and
[neural reranking](../tutorials/plugin/rag.md#neural-reranking) guide to index
documents and test RAG through a real chat request using `vela-auto`.

### Optional 32K Domain and FactCheck on ROCm

[Vela Domain](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Domain)
and [Vela FactCheck](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-FactCheck)
provide a fixed 32K FP32 graph, `onnx/model_rocm_32k.onnx`. The configuration
below pins model releases containing this graph.

To opt in, replace the two binding entries and the two complete deployment
entries in `vela-amd.yaml` with this fragment. The ROCm entries below replace
the MIGraphX entries, including their `compilation_cache_dir` setting; keep the
rest of your recipe.

```yaml
routing:
  model_bindings:
    domain_classifier:
      deployment: domain-amd
      contract: label_distribution.v1
      adapter: modernbert
      head: onnx/model_rocm_32k.onnx
      mapping_path: models/Vela-1.0-Encoder-307M-Domain/category_mapping.json
    fact_check_classifier:
      deployment: factcheck-amd
      contract: label_distribution.v1
      adapter: modernbert
      head: onnx/model_rocm_32k.onnx
global:
  model_catalog:
    deployments:
      domain-amd:
        artifact: models/Vela-1.0-Encoder-307M-Domain
        revision: f6354f54adcf38770f635ad903be2b00577f6c11
        provider: ort
        device: rocm:0
        precision: native
        input:
          max_tokens: 32768
          overflow: truncate
      factcheck-amd:
        artifact: models/Vela-1.0-Encoder-307M-FactCheck
        revision: 99ede1aba1563e59e416f744d25b3f6b7e9d8274
        provider: ort
        device: rocm:0
        precision: native
        input:
          max_tokens: 32768
          overflow: truncate
```

Validate and serve the edited file with the same commands above. Each binding
always uses its selected graph; it does not switch graphs by request length.
Here, `truncate` processes the first 32,768 tokens, including special tokens,
when the encoder input exceeds its budget. Use `reject` if complete input is
required. This policy affects the encoder view only; it does not shorten the
chat request or change the generation model's context window. See
[encoder input budgets](runtime/in-process.md#choose-an-input-budget).
The fixed graph pads even short inputs to 32,768 tokens, increasing their
latency and memory cost. Keep the default 8K MIGraphX deployments when that
budget fits your workload.

For capacity planning, one measured 27,001-token Preview request using these
two classifiers completed in 15.84 seconds. Separate single-classifier
validation reached 38.70 GiB of GPU memory; budget additionally for other
resident models and concurrent requests. These measurements cover this
Domain/FactCheck option. Other enabled classifiers retain their input limits,
so changing these two deployments does not establish a 32K all-ten-model
pipeline.

### Choose a Guard input budget

The default Guard deployment uses `onnx/model_rocm_8k.onnx` on ROCm. For requests
up to 512 tokens, select `onnx/model_rocm_512.onnx` and set that deployment's
`input.max_tokens: 512`. For long documents, select
`onnx/model_rocm_32k.onnx` and set `input.max_tokens: 32768`. Keep
`overflow: reject`, `precision: native`, and `device: rocm:0`.

Set the graph through `routing.model_bindings.prompt_guard.head`. These are
explicit deployment choices: the Router pads to the selected graph's input
shape and does not switch graphs automatically. Use the smallest graph that
covers your workload; a 32K graph can add substantial latency to short requests.
Changing Guard's budget does not change other enabled classifiers' limits.

## Production checklist

- Pin the Router, vLLM image, and model revision.
- Give the container only the devices, files, and network access it needs.
- Use distinct provider endpoints when the policy depends on real capability or
  cost differences.
- Protect the backend port from untrusted networks.
- Size context, concurrency, and parallelism from measured memory use.
- Monitor backend health, queueing, GPU memory, and routed generation—not only
  Router configuration validation.
