# Docker build files

This directory contains shared development and Router image Dockerfiles:

| File | Purpose |
| --- | --- |
| `Dockerfile` | CentOS Stream development environment with the project toolchains. |
| `Dockerfile.extproc` | Multi-architecture ExtProc Router image; Candle is the default binding. |
| `Dockerfile.extproc-rocm` | AMD ROCm 7.0 Router image with the ONNX Runtime ROCm provider. |
| `Dockerfile.precommit` | Reproducible lint and agent-harness toolchain used by CI. |

Build from the repository root so each Dockerfile can access all required
modules:

```bash
docker build -f tools/docker/Dockerfile.extproc \
  -t semantic-router-extproc:local .

docker build -f tools/docker/Dockerfile.extproc-rocm \
  -t semantic-router-extproc-rocm:local .
```

The ROCm image is x86-64 only and requires the host GPU devices at runtime.
See the public AMD installation guide for the supported run command and model
layout.

## CI and publication

Pull-request image validation is defined by
[`docker-validate.yml`](../../.github/workflows/docker-validate.yml). Main,
nightly, and release publication use
[`docker-publish.yml`](../../.github/workflows/docker-publish.yml). Those
workflows are the source of truth for the image matrix, platforms, tags, and
registry permissions; this README does not duplicate that changing inventory.

When editing a Dockerfile, keep dependency-only layers ahead of source copies
so Rust, Go, and container caches remain reusable. Run the repository's Docker
validation gate rather than relying on a successful build of only one target
architecture.
