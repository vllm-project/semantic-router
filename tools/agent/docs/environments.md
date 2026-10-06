# Environments

The shared `.venv-agent` contains pinned harness tools. Linked Git worktrees
reuse the primary worktree's environment through an ignored symlink; run
`make harness-bootstrap` to create it. The bootstrap needs Python 3.10 or newer
and uses `python3` from `PATH`; set `AGENT_BOOTSTRAP_PYTHON=<interpreter>` to
choose another one. It rebuilds an existing environment that runs an older
Python. Set `AGENT_VENV=<path>` only for an intentionally isolated tool
environment.

## Local runtime

| Environment | Build | Serve |
| --- | --- | --- |
| CPU | `make vllm-sr-dev` | `VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest vllm-sr serve --image-pull-policy never` |
| AMD | `make vllm-sr-dev VLLM_SR_PLATFORM=amd` | `VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:latest vllm-sr serve --image-pull-policy never --platform amd` |
| NVIDIA | `VLLM_SR_PLATFORM=nvidia make vllm-sr-build` | `VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest vllm-sr serve --platform nvidia --config <recipe> --image-pull-policy ifnotpresent` |

Make defaults to `latest`, while an editable CLI with a stable package version
defaults to release-tagged images. These overrides select the local builds.
If you customize the build tag, registry, or image variables, pass the actual
built images instead. The NVIDIA build target builds only the Router image;
`ifnotpresent` also allows the CLI to obtain missing companion images.

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
