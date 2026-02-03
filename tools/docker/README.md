# Dockerfiles

This directory contains Dockerfiles used across the project.

- `tools/docker/Dockerfile`: development base image (CentOS Stream) with toolchains (Rust, Go, Envoy, HF CLI).
- `tools/docker/Dockerfile.extproc`: builds the `extproc` (semantic-router external processor) image.
- `tools/docker/Dockerfile.extproc.cross`: cross-compilation optimized `extproc` Dockerfile.
- `tools/docker/Dockerfile.precommit`: pre-commit / lint tooling image for CI and local use.
- `tools/docker/Dockerfile.stack`: single-image “stack” build bundling router + dashboard + observability components.

## Build optimization (CI)

The workflow [.github/workflows/docker-publish.yml](../../.github/workflows/docker-publish.yml) builds multi-arch images (e.g. vllm-sr, extproc) on push to `main`. To keep build times under control:

- **Job timeout:** The build job has a 90-minute timeout; long or stuck builds fail instead of blocking the pipeline.
- **Rust cache:** vllm-sr and extproc use GitHub Actions cache for `candle-binding/target/` and `ml-binding/target/` (and cargo registry). Cache keys include `Cargo.toml`, `Cargo.lock`, and source hashes so dependency layers reuse across runs when only app code changes.
- **Build time metrics:** Each run reports **Build time: Xm Ys** in the job step summary and in a notice. Use this to track regressions and confirm builds stay within targets.
