# Environments

## `cpu-local`

- Build with `make vllm-sr-dev`
- Local runtime defaults to the split router/envoy/dashboard topology
- Split local runtime uses the local `vllm-sr` router image directly by default
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

## `ci-k8s`

- Build with `make vllm-sr-dev`
- Start with `vllm-sr serve --target k8s --image-pull-policy never`
- Use this for the managed Kind-backed shared-suite workflow
- The default shared suites run through [integration-test-k8s.yml](../../.github/workflows/integration-test-k8s.yml)
- The explicit local reproduction path is `make vllm-sr-test-integration VLLM_SR_TEST_TARGET=k8s`

## Shared CLI Surface

- Treat Docker and Kubernetes as deployment backends behind the same `vllm-sr` lifecycle commands rather than separate environment products.
- Docker stays the default target for local work; switch the same commands to Kubernetes with `--target k8s`.
- When `--target k8s` is used without an explicit `--context`, the CLI owns the default Kind cluster lifecycle for dev regression.
- The shared control surface is `vllm-sr serve`, `vllm-sr status`, `vllm-sr logs`, `vllm-sr dashboard`, and `vllm-sr stop`.
- Add `--namespace`, `--context`, and `--profile` only when the Kubernetes backend needs them.

## Selection Rule

- Default to `cpu-local`
- Use `amd-local` when platform behavior, ROCm image selection, or AMD defaults are affected
- Use `ci-k8s` for merge-gate coverage and all profile-sensitive routing/deploy behavior
