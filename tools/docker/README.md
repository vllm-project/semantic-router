# Docker build files

This directory contains the Router image and lint toolchain Dockerfiles:

| File | Purpose |
| --- | --- |
| `Dockerfile.extproc` | Every Router image: the Go router with the built-in model runtime. |
| `Dockerfile.precommit` | Reproducible lint and agent-harness toolchain used by CI. |

`Dockerfile.extproc` builds the three published Router images from one stage
graph and one target, `vllm-sr`. The router binary and the image-routing
assets are shared; `ACCELERATOR` selects the runtime's PyTorch build. Each
image serves `vllm-sr serve`, the Helm chart and the Operator, in standalone
or extproc mode:

| Image | `ACCELERATOR` | Platforms |
| --- | --- | --- |
| `vllm-sr` | `cpu` (default) | amd64, arm64 |
| `vllm-sr-rocm` | `rocm` | amd64 |
| `vllm-sr-cuda` | `cuda` | amd64 |

For one release, `vllm-sr` is also published as `extproc` and `vllm-sr-rocm`
as `extproc-rocm`, with the same digests, for deployments that still name the
former Kubernetes images.

The entrypoint, `/app/start-router.sh`, runs the Router on
`/app/config/config.yaml` when it gets no arguments or Router flags (the Helm
chart and the Operator pass `-gateway=...`), and starts the Router of a
`vllm-sr serve` stack when it gets a config path, as the CLI passes.

Build from the repository root so the Dockerfile can access every module:

```bash
docker build -f tools/docker/Dockerfile.extproc -t vllm-sr:local .
docker build -f tools/docker/Dockerfile.extproc --build-arg ACCELERATOR=rocm \
  -t vllm-sr-rocm:local .
```

CPU images carry CPU-only PyTorch. GPU images take PyTorch from the official
ROCm or CUDA wheels, which include their user-space GPU libraries, so they
need only the host's GPU driver and device access at runtime. The router
process has no native model code: every model runs in the runtime it starts.

## CI and publication

Pull-request builds go through
[`build-artifacts.yml`](../../.github/workflows/build-artifacts.yml), and
main, nightly and release publication through
[`docker-publish.yml`](../../.github/workflows/docker-publish.yml). The image
definitions (Dockerfile, target, platforms) live in
[`tools/ci/image_artifacts.py`](../ci/image_artifacts.py); this README does not
duplicate that inventory.

When editing the Dockerfile, keep dependency-only layers ahead of source
copies: PyTorch and the runtime's dependencies change rarely, the router binary
and configuration often.
