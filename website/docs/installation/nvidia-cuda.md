---
title: NVIDIA CUDA
description: Run vLLM backends on NVIDIA GPUs and optionally accelerate Semantic Router's local signal models with CUDA.
---

# Deploy with NVIDIA CUDA

The model server and Semantic Router are separate services. A common deployment
keeps the Router on CPU and gives the NVIDIA GPU to vLLM. Use the Router's CUDA
image when its local embeddings or classifiers also need GPU acceleration.

`--platform nvidia` affects the local Router stack only. It selects the CUDA
Router image, passes NVIDIA GPUs into the Router container, and changes its
generated runtime configuration so supported local signal models prefer CUDA.
It does **not** download a language model or start a vLLM server.

## Prerequisites

- Linux and an NVIDIA GPU supported by the vLLM release you plan to run;
- an x86-64 host when using the current Semantic Router CUDA image;
- an NVIDIA driver compatible with the selected container images;
- Docker and NVIDIA Container Toolkit;
- enough GPU memory for the vLLM model, KV cache, and any Router-side models;
  and
- a complete Semantic Router configuration with a reachable model endpoint.

Use the current
[vLLM NVIDIA requirements](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
for supported hardware. Install and configure the runtime with the
[NVIDIA Container Toolkit guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
then verify both the host driver and container access:

```bash
nvidia-smi
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

Do not continue until the container command can see the expected GPUs. The
second command is NVIDIA's
[sample workload](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html).

## Start and verify a vLLM backend

The following example follows the official
[vLLM Docker deployment](https://docs.vllm.ai/en/latest/deployment/docker/).
It publishes an OpenAI-compatible endpoint on port `8000` and keeps downloaded
model files in a named volume:

```bash
docker volume create vllm-huggingface-cache

docker run -d \
  --name vllm-nvidia \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v vllm-huggingface-cache:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B
```

Choose a model and vLLM arguments that fit the available GPUs. Pass
`HF_TOKEN` as an environment variable when a model requires authentication;
do not put the token in an image, command history, or Router config. Pin the
vLLM image and model revision for a controlled deployment instead of relying
on `latest`.

Wait for model loading to finish, then test vLLM before adding the Router:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models

curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Reply with: ready"}],
    "max_tokens": 16
  }'
```

Router validation cannot prove that a backend can load a model or generate a
response, so fix any direct vLLM error before continuing.

## Connect the backend

Bind the served model in your canonical config. For the local Docker stack,
the Router can reach a host-published port through `host.docker.internal`:

```yaml
providers:
  defaults:
    default_model: local/qwen
  models:
    - name: local/qwen
      provider_model_id: Qwen/Qwen3-0.6B
      api_format: openai
      backend_refs:
        - name: nvidia-vllm
          endpoint: host.docker.internal:8000
          protocol: http
          type: vllm
          weight: 1
```

This is a provider fragment, not a complete Router config. Add the matching
model card and route to your existing recipe, or configure the endpoint in the
Dashboard. The `provider_model_id` must match a model returned by vLLM's
`/v1/models` endpoint. See [Configuration](configuration) for a complete
minimal document and [Models and Recipes](models-and-recipes) for backend
binding and routing policy.

The example publishes port `8000` on the host for direct testing. Restrict that
port with host networking controls, or use private service discovery in a
production deployment. Do not expose an unauthenticated vLLM endpoint to an
untrusted network.

## Run the Router on NVIDIA

If vLLM should own all GPU memory, keep the Router on CPU:

```bash
vllm-sr validate --config config.yaml
vllm-sr serve --config config.yaml
```

To run supported Router-side ONNX embeddings and classifiers on CUDA, use
`--platform nvidia`. The CLI selects and pulls the published
`ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest` image by default:

```bash
vllm-sr validate --config config.yaml
vllm-sr serve --platform nvidia --config config.yaml
```

For a source checkout, build the maintained CUDA image first. The
`ifnotpresent` policy preserves that local build while still allowing the CLI
to obtain missing companion images:

```bash
VLLM_SR_PLATFORM=nvidia make vllm-sr-build
vllm-sr serve \
  --platform nvidia \
  --config config.yaml \
  --image-pull-policy ifnotpresent
```

Pin a release tag or digest in production. If the Router shares a GPU with
vLLM, measure memory and latency under representative concurrency; moving
small, batch-one signal models to CUDA does not always improve end-to-end
latency.

## Verify the routed path

Check the local stack and Router logs:

```bash
vllm-sr status
vllm-sr logs router | grep 'Using CUDA execution provider'
nvidia-smi
```

The CUDA log appears only when the active recipe loads a supported local ONNX
model. Then send a request through an entrypoint exposed by that recipe. Replace
`vllm-sr/auto` if your config uses another public model name:

```bash
curl --fail --include http://127.0.0.1:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Explain prefix caching briefly."}],
    "max_tokens": 64
  }'
```

A successful direct vLLM request proves the model server works. A successful
routed request proves the Router, recipe, and backend binding work together.

## Troubleshooting

### Docker rejects `--gpus all`

Configure Docker with `nvidia-ctk`, restart Docker, and repeat NVIDIA's sample
container command. Debug the container runtime before debugging either vLLM or
Semantic Router.

### The Router uses the CPU

Confirm that `--platform nvidia` selected the `vllm-sr-cuda` image and that
`VLLM_SR_NVIDIA_PRESERVE_CPU` is not enabled. Check the generated runtime
configuration and startup logs, not only the source recipe. A recipe without a
local ONNX signal model has nothing to move to CUDA.

### vLLM or the Router runs out of GPU memory

The vLLM model, KV cache, and Router-side models compete for the same device
memory. Leave the Router on CPU, reduce vLLM memory or concurrency settings, or
place the services on separate GPUs. Do not assume that silent CPU fallback
meets the same latency target.

### The backend works directly but routed requests fail

Check that the backend endpoint is reachable from the Router container and
that `provider_model_id` exactly matches `/v1/models`. Inside the Router
container, `localhost:8000` refers to the Router itself; use
`host.docker.internal:8000`, container DNS on a shared network, or a reachable
service address.

### Kubernetes does not schedule a GPU

`--platform nvidia` is a local-container shortcut. For Kubernetes, choose the
CUDA image and configure GPU resources, the NVIDIA device plugin, and node
placement through Helm values or the Operator. See
[Configuration Workflows](configuration-workflows#helm) for the deployment
boundary.
