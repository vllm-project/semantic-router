# Docker build files

This directory contains the Router image and lint toolchain Dockerfiles:

| File | Purpose |
| --- | --- |
| `Dockerfile.extproc` | Every Router image: the Go router with the built-in model runtime. |
| `Dockerfile.precommit` | Reproducible lint and agent-harness toolchain used by CI. |

`Dockerfile.extproc` builds all five published Router images from one stage
graph. The router binary, the Vela Omni bundle and the image-routing assets are
shared; `ACCELERATOR` selects the runtime's PyTorch build, and the target
selects the Kubernetes image or the `vllm-sr serve` stack image:

| Image | Target | `ACCELERATOR` | Platforms |
| --- | --- | --- | --- |
| `extproc` | `extproc` (default) | `cpu` (default) | amd64, arm64 |
| `extproc-rocm` | `extproc` | `rocm` | amd64 |
| `vllm-sr` | `vllm-sr` | `cpu` | amd64, arm64 |
| `vllm-sr-rocm` | `vllm-sr` | `rocm` | amd64 |
| `vllm-sr-cuda` | `vllm-sr` | `cuda` | amd64 |

Build from the repository root so the Dockerfile can access every module:

```bash
docker build -f tools/docker/Dockerfile.extproc -t extproc:local .
docker build -f tools/docker/Dockerfile.extproc --target vllm-sr -t vllm-sr:local .
docker build -f tools/docker/Dockerfile.extproc --build-arg ACCELERATOR=rocm \
  --target vllm-sr -t vllm-sr-rocm:local .
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
