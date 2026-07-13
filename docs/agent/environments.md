# Environments

## `cpu-local`

- Build with `make vllm-sr-dev`
- Local runtime defaults to the split router/envoy/dashboard topology
- Split local runtime uses the local `vllm-sr` router image directly by default
- Split Intelligent Routing for Mixture-of-Models uses the local `vllm-sr` router image directly by default
- Only if that local router image is already up to date, you can reuse it with `make vllm-sr-dev SKIP_ROUTER_IMAGE=1`
- Start with `vllm-sr serve --image-pull-policy never`
- Use this for the default local Docker workflow
- Default smoke config: [config.agent-smoke.cpu.yaml](../../e2e/config/config.agent-smoke.cpu.yaml)
- If you need a non-default config, run `make agent-serve-local ENV=cpu AGENT_SERVE_CONFIG=<config>`
- For isolated parallel local stacks, add `AGENT_STACK_NAME=<name>` and `AGENT_PORT_OFFSET=<n>`, for example:
  `make agent-serve-local ENV=cpu AGENT_STACK_NAME=lane-a AGENT_PORT_OFFSET=0`
  and `make agent-serve-local ENV=cpu AGENT_STACK_NAME=lane-b AGENT_PORT_OFFSET=200`
- Use the same `AGENT_STACK_NAME` and `AGENT_PORT_OFFSET` values with `make agent-smoke-local` and `make agent-stop-local`

## `amd-local`

- Build with `make vllm-sr-dev VLLM_SR_PLATFORM=amd`
- Local runtime defaults to the split router/envoy/dashboard topology
- Split local runtime uses the local `vllm-sr-rocm` router image directly by default
- Only if that local router image is already up to date, you can reuse it with `make vllm-sr-dev VLLM_SR_PLATFORM=amd SKIP_ROUTER_IMAGE=1`
- Start with `vllm-sr serve --image-pull-policy never --platform amd`
- Use this for ROCm/AMD validation and platform-default image checks
- Default smoke config: [config.agent-smoke.amd.yaml](../../e2e/config/config.agent-smoke.amd.yaml)
- If you need a non-default config, run `make agent-serve-local ENV=amd AGENT_SERVE_CONFIG=<config>`
- The same `AGENT_STACK_NAME=<name>` and `AGENT_PORT_OFFSET=<n>` overrides work for isolated AMD-local stacks
- For real AMD model deployment and backend container setup, read [deploy/amd/README.md](../../deploy/amd/README.md)
- Use [deploy/recipes/balance.yaml](../../deploy/recipes/balance.yaml) as the reference YAML-first AMD routing profile
- See [amd-local.md](amd-local.md)

## `nvidia-local`

- Build with `VLLM_SR_PLATFORM=nvidia make vllm-sr-build`
- Local runtime uses the `ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest` image (published via `docker-publish.yml` / `docker-release.yml`); a local build tagged `vllm-sr-cuda:local` is also supported
- Start with `vllm-sr serve --platform nvidia --config <recipe>` — `--platform nvidia` selects the published CUDA image, wires `--gpus all` / CDI passthrough, and flips `use_cpu` to false for router internal models under `global.model_catalog`, at parity with `--platform amd`. Pass `--image` / `VLLM_SR_IMAGE` only to pin a specific image, and `VLLM_SR_NVIDIA_PRESERVE_CPU=1` to keep the recipe's CPU settings
- Use this for CUDA/NVIDIA validation of BERT classifiers, mmBERT embeddings, jailbreak guard, PII detection, and other router-side ML modules on GPU
- Verified hardware in the playbook: RTX 4090 (compute_cap 8.9), CUDA driver 580
- Set `use_cpu: false` under `global.model_catalog` in your routing recipe for the modules you want on GPU
- For the full router-on-CUDA build/serve/validation playbook, read [nvidia-local.md](nvidia-local.md)
- Note: `deploy/nvidia/` is intentionally empty until an NVIDIA-specific routing profile (a real backend recipe parallel to [deploy/amd/README.md](../../deploy/amd/README.md)) is authored — that is separate work from this router-on-GPU environment

## `ci-k8s`

- Run local profile checks with `make e2e-test E2E_PROFILE=<profile>`
- CI expands to the standard kind/Kubernetes matrix in [integration-test-k8s.yml](../../.github/workflows/integration-test-k8s.yml)

## Selection Rule

- Default to `cpu-local`
- Use `amd-local` when platform behavior, ROCm image selection, or AMD defaults are affected
- Use `nvidia-local` when CUDA image selection, NVIDIA GPU passthrough, or router-side ML modules on GPU are affected
- Use `ci-k8s` for merge-gate coverage and all profile-sensitive routing/deploy behavior
