# Dockerfiles

This directory contains Dockerfiles used across the project.

- `tools/docker/Dockerfile`: development base image (CentOS Stream) with toolchains (Rust, Go, Envoy, HF CLI).
- `tools/docker/Dockerfile.extproc`: builds the `extproc` (semantic-router external processor) image.
- `tools/docker/Dockerfile.extproc.cross`: cross-compilation optimized `extproc` Dockerfile.
- `tools/docker/Dockerfile.precommit`: pre-commit / lint tooling image for CI and local use.
- `tools/docker/Dockerfile.stack`: single-image “stack” build bundling router + dashboard + observability components.
