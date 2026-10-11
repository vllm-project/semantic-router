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
  -p 8000:8000 \
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

The host port is only for checking the backend yourself; the Router reaches it
as `vllm:8000` on `vllm-sr-network`. Keep it off 8090, which the local stack's
sr-bench service uses, or `vllm-sr serve` stops with "sr-bench port 8090 is
already in use". With `VLLM_ROCM_USE_AITER=1`, the first start compiles AITER
kernels, which can take tens of minutes on a host with few free CPU cores.

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
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models

curl --fail http://127.0.0.1:8000/v1/chat/completions \
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

Install the CLI as the [Quickstart](installation/installation.md#install) describes. To
install only the CLI, without starting the stack:

```bash
curl -fsSL https://vllm-sr.ai/install.sh | \
  bash -s -- --mode cli --runtime skip --no-launch
```

For a simple one-model deployment, start the stack. Use `--platform cpu` to
keep Router-side inference on the CPU. Plain `vllm-sr serve` uses automatic
platform detection; `--platform rocm` selects the GPU-capable Router image
([Run Vela routing models on AMD](#run-vela-routing-models-on-amd)):

```bash
vllm-sr serve --platform cpu
```

Then open the Dashboard at `http://localhost:8700`, connect a model with
provider vLLM, the served model name and the address `vllm:8000`, and activate
the generated config.

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

Send a request through the Router's listener using the automatic entrypoint:

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

The Router's own models (the Vela classifiers, embeddings, reranker and
decision models) run in the [model runtime](model-runtime/overview.md). On
AMD Instinct MI300X and MI325X GPUs it runs them through PyTorch for ROCm,
which is validated. `--platform rocm` selects the AMD image, which ships the
runtime with the validated stack (PyTorch 2.12 for ROCm 7.2, FLA 0.5.2, and
`causal-conv1d` 1.7.0 built for ROCm), and passes the GPUs to the Router.
Every model checks its answers against references verified on that stack
when it loads; see
[Choose a model](model-runtime/choose-a-model.md#hardware).

The [Vela AMD Model Card](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/vela-amd/README.md)
and complete config place all ten task models on `rocm:0`. Connect an existing
OpenAI-compatible backend served with `--served-model-name vela-default`. The
config expects `http://vllm:8000`; attach that backend to `vllm-sr-network`
with network alias `vllm`, or edit the endpoint. Verify a direct request using
that name before routing. Reserve enough memory and compute for the Router
alongside the generation backend; use `VLLM_SR_AMD_ROUTER_VISIBLE_DEVICES` when
selecting a Router GPU. The deployment's device index `0` refers to its visible
GPU.

```bash
curl --fail --location --output vela-amd.yaml \
  https://raw.githubusercontent.com/vllm-project/semantic-router/main/config/recipes/vela-amd/config.yaml
vllm-sr config validate --config vela-amd.yaml
vllm-sr serve --platform rocm --config vela-amd.yaml
```

The platform flag selects the image and device access. Each deployment's
`device` decides where its model runs; explicit CPU choices remain CPU
choices. The first start downloads the models; the CLI waits up to 1,800
seconds by default, and `--startup-timeout SECONDS` sets a longer bounded wait.
A timeout leaves the owned containers available for logs and readiness
inspection.

Once `/ready` succeeds, inspect the real signals and their timings:

```bash
curl --fail http://localhost:8080/ready
curl --fail 'http://localhost:8080/api/v1/routing/preview?trace=true' \
  -H 'Content-Type: application/json' \
  -d '{"model":"vela-auto","text":"Debug this Python program and fix its error."}' \
  | jq '{decision_result, signal_confidences, signal_values, signal_errors, metrics, eval_trace}'
```

Preview does not execute retrieval or generation. Follow the recipe and
[neural reranking](../tutorials/plugin/rag.md#neural-reranking) guide to index
documents and test RAG through a real chat request using `vela-auto`.

### Longer inputs

Every Vela task model reads up to 32,768 tokens on ROCm. Set the longest input
a deployment accepts with `input.max_tokens`; short requests stay fast because
nothing is padded to a fixed size:

```yaml
global:
  model_catalog:
    deployments:
      domain-amd:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Domain
        device: rocm:0
        input:
          max_tokens: 32768
          overflow: truncate
```

`truncate` classifies the first 32,768 tokens, including special tokens, and
reports that it did; `reject` leaves the signal unknown for longer input
instead. This limit applies to the classifier only; it does not shorten the
chat request or change the generation model's context window. Guard and PII
scan long inputs in overlapping windows (`overflow: window`); see
[Prompt attacks and unsafe content](../model-runtime/guides/safety.md) and
[Detect PII](../model-runtime/guides/pii.md).

Earlier releases selected fixed ONNX Runtime graphs (`head:
onnx/model_rocm_32k.onnx`) and MIGraphX compilation caches here.
`vllm-sr config migrate` removes those settings; see
[Migrate from the native bindings](model-runtime/migrate.md).

## Compare with the 98x paper setup

The [98x routing paper](https://arxiv.org/abs/2603.12646) benchmarks one
classifier and prompt-compression setup on an AMD Instinct MI300X. The
maintained [Vela AMD recipe](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/vela-amd/README.md)
and the reference [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml)
use different settings, so the paper's latency and memory figures describe that
benchmark rather than these configurations.

| Setting | Paper benchmark | Current setting |
| --- | --- | --- |
| Router models | Three mmBERT-32K classifier sessions (270M parameters, FP16) for domain, jailbreak and PII | Vela AMD recipe: ten Vela 1.0 307M task models in the model runtime's exact FP32 profile; Domain, Guard and PII cover those three tasks |
| Execution | ONNX Runtime ROCm with CK Flash Attention for all three classifiers | PyTorch for ROCm in the model runtime; the request's model work arrives in one bundled call |
| Prompt compression | On, with a 512-token budget | Off by default and in the Vela AMD recipe. The reference config enables it with `max_tokens: 4096` |
| Weights for TextRank, position, TF-IDF and novelty | 0.20, 0.40, 0.35, 0.05 | Reference config: 0.4, 0.2, 0.3, 0.1 |
| Position depth | 0.5 | Reference config: 0.1 |
| Sentences always kept | First 3 and last 2 | Reference config: first 3 and last 2 |
| Classifiers that read compressed text | Domain, jailbreak and PII in the latency tables | Domain reads it. Jailbreak and PII read the full prompt through `skip_signals`, which matches the production setup the paper describes |

The paper's router GPU footprint of under 800 MB covers its three classifier
sessions with compression on. The Vela AMD recipe loads ten task models, so
size Router memory from measurements on your hardware.

To use the paper's 512-token compression profile, add this block to your
config:

```yaml
global:
  model_catalog:
    modules:
      prompt_compression:
        enabled: true
        profile: default
        max_tokens: 512
        skip_signals: [jailbreak, pii]
```

The `default` profile supplies the paper's weights, position depth and kept
sentences. Explicit weight fields and `position_depth` override the profile, so
remove them if you start from the reference config. Compression shortens only
the text used for signal evaluation. Compare routing results on representative
requests before you change the budget.

## Production checklist

- Pin the Router, vLLM image, and model revision.
- Give the container only the devices, files, and network access it needs.
- Use distinct provider endpoints when the policy depends on real capability or
  cost differences.
- Protect the backend port from untrusted networks.
- Size context, concurrency, and parallelism from measured memory use.
- Monitor backend health, queueing, GPU memory, and routed generation—not only
  Router configuration validation.
