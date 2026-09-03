# NVIDIA Local Runtime

Use this runbook to place the router's ONNX-based embedding and classifier
models on an NVIDIA GPU. It does not serve the backend LLM; backend endpoints
remain normal provider configuration.

GPU residency is most useful at high classifier concurrency or with larger
local signal models. Benchmark the CPU path first for small batches.

## Prerequisites

- Linux with an NVIDIA driver compatible with the CUDA runtime in
  [`Dockerfile.cuda`](../../../src/vllm-sr/Dockerfile.cuda).
- Docker with NVIDIA Container Toolkit, or Podman with NVIDIA CDI.
- A valid router recipe and reachable backend model.

Verify container GPU access before starting the router:

```bash
nvidia-smi
docker run --rm --gpus all \
  nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

For Podman, verify the equivalent `--device nvidia.com/gpu=all` path.

## Build or Select the Image

The supported local development build is:

```bash
VLLM_SR_PLATFORM=nvidia make vllm-sr-build
```

For a published build, `vllm-sr serve --platform nvidia` selects
`ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest` by default. Pin an
immutable image with `--image IMAGE_REF` or set the NVIDIA-specific default with
`VLLM_SR_IMAGE_NVIDIA`.

Verify that the selected image exposes the CUDA Execution Provider:

```bash
docker run --rm --gpus all \
  --entrypoint /opt/vllm-sr-venv/bin/python \
  IMAGE_REF \
  -c 'import onnxruntime as ort; print(ort.get_available_providers())'
```

Do not continue if `CUDAExecutionProvider` is absent.

## Start

Use the normal local stack path:

```bash
vllm-sr serve --platform nvidia --config config/recipes/RECIPE/config.yaml
```

The platform option:

- selects the CUDA router image;
- adds NVIDIA GPU passthrough for Docker or Podman;
- sets router-internal `use_cpu` fields to `false` in the generated runtime
  config.

Set `VLLM_SR_NVIDIA_PRESERVE_CPU=1` when the recipe's CPU/GPU choices must be
kept, or `VLLM_SR_NVIDIA_GPU_PASSTHROUGH=0` when another orchestrator owns GPU
attachment.

Do not replace the router container by hand as part of the normal workflow. The
CLI owns its mounts, network, ports, environment, and generated runtime config.

## Verify

Check the router container and startup logs:

```bash
docker ps --filter name=vllm-sr-router-container
docker logs vllm-sr-router-container 2>&1 | \
  grep 'Using CUDA execution provider'
docker logs vllm-sr-router-container 2>&1 | \
  grep embedding_models_init_started
nvidia-smi
```

Expect CUDA-provider messages for the ONNX models enabled by the recipe and
`use_cpu: false` in the embedding initialization event. A process in
`nvidia-smi` proves GPU memory allocation; observe GPU utilization under a
representative request load before making performance claims.

Then send a normal request through the router endpoint configured by the local
stack:

```bash
curl -sS http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Explain TCP and UDP."}]
  }'
```

## Troubleshooting

### Container runtime cannot attach the GPU

If Docker rejects `--gpus all`, configure NVIDIA Container Toolkit and repeat
the prerequisite container test. For Podman, verify NVIDIA CDI before debugging
the router.

### Models stay on CPU

Check, in order:

1. the selected image contains `CUDAExecutionProvider`;
2. `--platform nvidia` reached the serve command;
3. no preserve-CPU environment option is set;
4. the generated runtime config, not only the source recipe, contains the
   expected `use_cpu: false` fields;
5. startup logs do not report CUDA allocation or session-creation failures.

### Only some models use CUDA

Mixed CUDA/CPU startup usually means that later sessions could not allocate GPU
memory. Use `nvidia-smi` to identify competing processes and available memory.
Reduce co-tenant usage, run fewer local signal models, or deliberately preserve
CPU placement for modules that do not benefit from the GPU. Do not assume that
silent CPU fallback preserves the latency target.

### CUDA is active but latency does not improve

Small, batch-one BERT inference can be competitive on CPU. Compare CPU and CUDA
with the same recipe, model cache, request lengths, concurrency, and warm-up.
Report throughput and tail latency, not one request's wall time.

## Implementation Sources

- [`src/vllm-sr/Dockerfile.cuda`](../../../src/vllm-sr/Dockerfile.cuda)
- [`tools/make/docker.mk`](../../../tools/make/docker.mk)
- [`src/vllm-sr/cli/container_images.py`](../../../src/vllm-sr/cli/container_images.py)
- [`src/vllm-sr/cli/container_run_command.py`](../../../src/vllm-sr/cli/container_run_command.py)
- [`src/vllm-sr/cli/commands/runtime_config_mutation.py`](../../../src/vllm-sr/cli/commands/runtime_config_mutation.py)
- [`onnx-binding`](../../../onnx-binding/README.md)

Use [`amd-local.md`](amd-local.md) for the corresponding ROCm local-runtime
contract.
