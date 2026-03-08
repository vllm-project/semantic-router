# Environments

## `cpu-local`

- Build with `make vllm-sr-dev`
- Start with `vllm-sr serve --image-pull-policy never`
- Use this for the default local Docker workflow
- Default smoke config: [config.agent-smoke.cpu.yaml](../../config/testing/config.agent-smoke.cpu.yaml)
- If you need a non-default config, run `make agent-serve-local ENV=cpu AGENT_SERVE_CONFIG=<config>`

## `amd-local`

- Build with `make vllm-sr-dev VLLM_SR_PLATFORM=amd`
- Start with `vllm-sr serve --image-pull-policy never --platform amd`
- Use this for ROCm/AMD validation and platform-default image checks
- Default smoke config: [config.agent-smoke.amd.yaml](../../config/testing/config.agent-smoke.amd.yaml)
- If you need a non-default config, run `make agent-serve-local ENV=amd AGENT_SERVE_CONFIG=<config>`
- For real AMD model deployment and backend container setup, read [deploy/amd/README.md](../../deploy/amd/README.md)
- Use [deploy/amd/config.yaml](../../deploy/amd/config.yaml) as the reference YAML-first AMD routing profile
- See [amd-local.md](amd-local.md)

## `ci-k8s`

- Run local profile checks with `make e2e-test E2E_PROFILE=<profile>`
- CI expands to the standard kind/Kubernetes matrix in [integration-test-k8s.yml](../../.github/workflows/integration-test-k8s.yml)

## Selection Rule

- Default to `cpu-local`
- Use `amd-local` when platform behavior, ROCm image selection, or AMD defaults are affected
- Use `ci-k8s` for merge-gate coverage and all profile-sensitive routing/deploy behavior
