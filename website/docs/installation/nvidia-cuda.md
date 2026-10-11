---
title: NVIDIA CUDA
description: Run vLLM backends on NVIDIA GPUs and optionally accelerate Semantic Router's local signal models with CUDA.
---

# Deploy with NVIDIA CUDA

The model server and Semantic Router are separate services. A common deployment
keeps the Router on CPU and gives the NVIDIA GPU to vLLM. Use the Router's CUDA
image when its local embeddings or classifiers also need GPU acceleration.

`--platform cuda` selects the execution backend for the Router instance. On Docker it selects the CUDA
Router image, passes NVIDIA GPUs into the Router container, and changes its
generated runtime configuration so supported local signal models prefer CUDA.
It does **not** download a language model or start a vLLM server.

## Prerequisites

- Linux and an NVIDIA GPU supported by the vLLM release you plan to run;
- an x86-64 host when using the current Semantic Router CUDA image;
- a GPU of compute capability 7.0 or newer (Volta and later) for the Router
  image, the oldest architecture its PyTorch build (CUDA 12.8) includes;
- an NVIDIA driver, and GPU passthrough into the Router container for the
  Router-side models that run on CUDA;
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
    model: local/qwen
  models:
    - name: local/qwen
      provider_model_id: Qwen/Qwen3-0.6B
      api_format: openai
      backend_refs:
        - name: nvidia-vllm
          endpoint: host.docker.internal:8000
          protocol: http
          provider: vllm
          weight: 1
```

This is a provider fragment, not a complete Router config. Add the matching
model card and route to your existing recipe, or configure the endpoint in the
Dashboard. The `provider_model_id` must match a model returned by vLLM's
`/v1/models` endpoint. See [Configuration](configuration) for a complete
minimal document and
[Models, Entrypoints, and Serving](../tutorials/global/models-entrypoints-serving)
for backend binding and routing policy.

The example publishes port `8000` on the host for direct testing. Restrict that
port with host networking controls, or use private service discovery in a
production deployment. Do not expose an unauthenticated vLLM endpoint to an
untrusted network.

## Run the Router on NVIDIA

If vLLM should own all GPU memory, explicitly keep Router-side inference on CPU:

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --platform cpu --config config.yaml
```

Without `--platform cpu`, automatic platform detection can select CUDA on a
GPU host.

The Router's own models (classifiers, embeddings, decision models) run in the
[model runtime](model-runtime/overview.md). To run them on CUDA, use
`--platform cuda`: the CUDA image ships the runtime with the CUDA build of
PyTorch, and deployments with `device: auto` or `device: cuda:0` use the GPU.
CUDA support works but is not yet validated; measure it on your hardware.
A stable CLI selects the matching published release image
(for example, CLI `0.4.0` uses `vllm-sr-cuda:v0.4.0`). Development CLI builds
use `:latest` unless an image is specified explicitly:

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --platform cuda --config config.yaml
```

For a source checkout, build the maintained CUDA image first and explicitly
select its `latest` tag. An editable CLI installation with a stable package
version otherwise selects the release tag. With the image override,
`ifnotpresent` reuses the local build while allowing the CLI to obtain missing
companion images:

```bash
VLLM_SR_PLATFORM=cuda make vllm-sr-build
VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest \
  vllm-sr serve \
  --platform cuda \
  --config config.yaml \
  --image-pull-policy ifnotpresent
```

If you override the build tag or registry, set `VLLM_SR_IMAGE` to the actual
built image and use `VLLM_SR_DASHBOARD_IMAGE` for a custom companion image.

Pin a digest when deployments require an immutable image identity. If the
Router shares a GPU with vLLM, measure memory and latency under representative
concurrency; moving small, batch-one signal models to CUDA does not always improve end-to-end
latency.

## Verify the routed path

Check the local stack and Router logs:

```bash
vllm-sr status
vllm-sr logs router | grep model_binding_ready
nvidia-smi
```

Every prepared Router-side model reports the device it runs on, so a GPU
deployment shows `"device":"cuda:0"` for the configured classifiers and
`nvidia-smi` lists the Router process. A model reporting `"device":"cpu"` runs
on the CPU regardless of GPU passthrough. Then send a request through an
entrypoint exposed by that recipe. Replace
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
Semantic Router: without working passthrough, the Router-side models configured
for CUDA cannot load.

### The Router uses the CPU

Confirm that `--platform cuda` selected the `vllm-sr-cuda` image and that
`VLLM_SR_CUDA_PRESERVE_CPU` is not enabled. Check the generated runtime
configuration and startup logs, not only the source recipe. A recipe without a
local signal model has nothing to move to CUDA. `GET /v1/models` on a runtime,
or the Dashboard's model inventory, shows the device each model runs on.

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

On Kubernetes, `--platform cuda` selects the CUDA image and GPU resources from
canonical replica placement. The NVIDIA device plugin and matching schedulable
nodes must already exist. Configure node placement through Helm values or the
Operator; `--platform auto` inspects the selected cluster, not the CLI host. See
[Configuration Workflows](configuration-workflows#helm) for the deployment
boundary.
