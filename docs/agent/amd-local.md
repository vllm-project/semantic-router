# AMD Local Notes

- Build command: `make vllm-sr-dev VLLM_SR_PLATFORM=amd`
- Serve command: `vllm-sr serve --image-pull-policy never --platform amd`
- Default smoke config: [config.agent-smoke.amd.yaml](../../e2e/config/config.agent-smoke.amd.yaml)
- Real AMD deployment playbook: [deploy/amd/README.md](../../deploy/amd/README.md)
- Real AMD routing profile: [deploy/recipes/balance.yaml](../../deploy/recipes/balance.yaml)
- Expected behavior:
  - ROCm image defaults are selected
  - `VLLM_SR_PLATFORM=amd` is passed through to the container
  - router internal signal model `use_cpu` settings default to `false`
  - `VLLM_SR_AMD_PRESERVE_CPU=1` preserves CPU settings when the router has no dedicated GPU headroom

## When To Use Which Config

- Use [config.agent-smoke.amd.yaml](../../e2e/config/config.agent-smoke.amd.yaml) for fast local smoke and feature-gate validation.
- Use [deploy/recipes/balance.yaml](../../deploy/recipes/balance.yaml) when you need a single real AMD backend with a multi-alias routing profile.
- Use [deploy/amd/README.md](../../deploy/amd/README.md) when you need the full backend deployment flow, Docker network setup, model container examples, and dashboard-first vs YAML-first setup guidance.

## Validation Checklist

- Local image exists before `serve`
- `vllm-sr status all` reports the container and dashboard as healthy
- No unexpected fallback to the default non-AMD image
- Relevant local E2E profiles still pass

## Router Learning AMD Validation

For the agentic AMD recipe, validate both Router Learning scopes before using
the endpoint as PR evidence:

- `conversation` scope: keep the same `x-session-id`, change
  `x-conversation-id`, and verify a new user run can re-route while turns inside
  one conversation stay stable during tool loops or cache-heavy follow-ups.
- `session` scope: keep the same `x-session-id` across multiple
  `x-conversation-id` values and verify the established session model is
  protected until `idle_timeout_seconds` releases it or a decision uses
  `adaptations.session_aware.mode: bypass`.
- Privacy, security, and local-only decisions should bypass Router Learning and
  route to the local model even when the previous protected model was remote.
- Responses should include compact learning headers such as
  `x-vsr-learning-methods`, `x-vsr-learning-actions`,
  `x-vsr-learning-scopes`, `x-vsr-learning-reasons`, and
  `x-vsr-learning-modes`; full details should be joined through
  `x-vsr-replay-id` and Router Replay.
- Dashboard and API readiness require the Envoy/OpenAI endpoint, router status,
  replay diagnostics, and dashboard health to be available from the validation
  host before reporting the AMD router address.
