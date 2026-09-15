# Environments

The shared `.venv-agent` contains pinned harness tools. Linked Git worktrees
reuse the primary worktree's environment through an ignored symlink; run
`make harness-bootstrap` to create it. Set `AGENT_VENV=<path>` only for an
intentionally isolated tool environment.

## Local runtime

| Environment | Build | Serve |
| --- | --- | --- |
| CPU | `make vllm-sr-dev` | `vllm-sr serve --image-pull-policy never` |
| AMD | `make vllm-sr-dev VLLM_SR_PLATFORM=amd` | `vllm-sr serve --image-pull-policy never --platform amd` |
| NVIDIA | `VLLM_SR_PLATFORM=nvidia make vllm-sr-build` | `vllm-sr serve --platform nvidia --config <recipe>` |

Use CPU by default. Select AMD or NVIDIA only when platform defaults, GPU
passthrough, router-side ML execution, or platform images are part of the
change. Do not invent a second serve path. `SKIP_ROUTER_IMAGE=1` is valid only
when the required local router image is already current.

For real AMD backend deployment, use
`website/docs/installation/amd-rocm.md` and
`config/recipes/balance/config.yaml`. Discover private validation hosts at run
time; never write their details into tracked files or public receipts.

## Integration and E2E

Run `make verify DOMAIN=<domain>` for a domain's explicit integration command,
or `make verify PROFILE=<profile>` for one Kubernetes E2E profile. CI uses the
same profile names from `tools/agent/domains.yaml` through
`.github/workflows/integration-test-k8s.yml`.
