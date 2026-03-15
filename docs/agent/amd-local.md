# AMD Local Notes

- Build command: `make vllm-sr-dev VLLM_SR_PLATFORM=amd`
- Serve command: `vllm-sr serve --image-pull-policy never --platform amd`
- Default smoke config: [config.agent-smoke.amd.yaml](../../e2e/config/config.agent-smoke.amd.yaml)
- Real AMD deployment playbook: [deploy/amd/README.md](../../deploy/amd/README.md)
- Real AMD routing profile: [deploy/amd/config.yaml](../../deploy/amd/config.yaml)
- Expected behavior:
  - ROCm image defaults are selected
  - `VLLM_SR_PLATFORM=amd` is passed through to the container
  - AMD GPU defaults in the CLI config merge path are preserved

## When To Use Which Config

- Use [config.agent-smoke.amd.yaml](../../e2e/config/config.agent-smoke.amd.yaml) for fast local smoke and feature-gate validation.
- Use [deploy/amd/config.yaml](../../deploy/amd/config.yaml) when you need a single real AMD backend with a multi-alias routing profile.
- Use [deploy/amd/README.md](../../deploy/amd/README.md) when you need the full backend deployment flow, Docker network setup, model container examples, and dashboard-first vs YAML-first setup guidance.

## Validation Checklist

- Local image exists before `serve`
- `vllm-sr status all` reports the container and dashboard as healthy
- No unexpected fallback to the default non-AMD image
- Relevant local E2E profiles still pass
