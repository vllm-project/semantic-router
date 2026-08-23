---
title: AMD ROCm
description: Run an OpenAI-compatible vLLM backend on AMD Instinct GPUs and connect it to vLLM Semantic Router.
---

# Deploy with AMD ROCm

Semantic Router can run on CPU while vLLM serves the selected model on AMD
Instinct GPUs. This guide starts one ROCm backend, verifies it directly, and
then connects it to the local Router stack.

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

Start the Router and Dashboard:

```bash
vllm-sr serve --platform amd
```

In **Models**, connect the AMD-hosted endpoints. Then choose the built-in
Balanced Recipe, assign those Models to its decisions, and publish an
Entrypoint. The same lifecycle is available through the Router Management API;
no Recipe is selected on the `serve` command line.

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

## Production checklist

- Pin the Router, vLLM image, and model revision.
- Give the container only the devices, files, and network access it needs.
- Use distinct provider endpoints when the policy depends on real capability or
  cost differences.
- Protect the backend port from untrusted networks.
- Size context, concurrency, and parallelism from measured memory use.
- Monitor backend health, queueing, GPU memory, and routed generation—not only
  Router configuration validation.
