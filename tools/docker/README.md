# Dockerfiles

This directory contains Dockerfiles used across the project.

- `tools/docker/Dockerfile`: development base image (CentOS Stream) with toolchains (Rust, Go, Envoy, HF CLI).
- `tools/docker/Dockerfile.extproc`: builds the `extproc` image (single-platform, used for PR amd64-only builds).
- `tools/docker/Dockerfile.extproc.cross`: cross-compilation optimized `extproc` Dockerfile (used for multi-arch push/dispatch builds).
- `tools/docker/Dockerfile.precommit`: pre-commit / lint tooling image for CI and local use.

## Build optimization (CI)

The read-only reusable
[.github/workflows/docker-validate.yml](../../.github/workflows/docker-validate.yml)
workflow owns PR validation. The write-capable
[.github/workflows/docker-publish.yml](../../.github/workflows/docker-publish.yml)
workflow owns main, nightly, and release publication. Lifecycle dispatchers
pass an explicit image matrix from shared PR classification or the full release
inventory.

### Architecture: Affected images and lifecycle modes

- PRs build only classified affected images on amd64, with no registry login or
  package-write permission. Images already built by an active Kubernetes, CLI,
  Memory, or Operator suite are removed from the standalone matrix.
- A validator-only workflow change builds only representative `vllm-sr`;
  publisher-only changes use static release/image contract checks.
- Main publishes affected images with immutable commit tags plus `latest`.
  Generic core changes publish `extproc` and `vllm-sr`; CUDA and ROCm variants
  publish only for their platform-specific paths.
- Stable releases contain the eight production deliverables: `dashboard`,
  `extproc`, `extproc-rocm`, `operator`, `operator-bundle`, `vllm-sr`,
  `vllm-sr-cuda`, and `vllm-sr-rocm`.
- Nightly additionally publishes the `vllm-sr-sim` developer companion and the
  `anthropic-shim` and `llm-katan` test fixtures. Fixtures receive dated
  `nightly-YYYYMMDD` tags plus a mutable `nightly` tag used by maintained E2E
  references; they are not production release artifacts.
- Release, nightly, and main publication share the same image definitions and
  build-argument resolver.

### Multi-architecture builds

- Buildx and QEMU provide the multi-architecture builder used for publish modes.
- `vllm-sr` uses `TARGETARCH` in its Dockerfile so build stages select the
  target architecture correctly.
- CUDA and ROCm definitions explicitly remain `linux/amd64`.

### Rust dependency pre-caching

All Dockerfiles use a two-step Rust build pattern:

1. Copy `Cargo.toml` + `Cargo.lock`, create a dummy `lib.rs`, and build dependencies (cached Docker layer).
2. Copy real source, **delete stale `.so`/`.a`** from the dummy build, and rebuild (only recompiles application code).

The stale library deletion (`find target -name "libcandle_semantic_router.so" -delete`) is critical: without it, cargo's incremental compilation may reuse the empty library from the dummy build, causing linker errors in the Go build stage.

### Other optimizations

- **No `cargo clean`:** Dependency cache from the pre-build layer is reused; only application code is recompiled.
- **Job timeouts:** Publication builds have a 180-minute timeout; PR builds have
  a 120-minute timeout.
- **GHA cache:** Docker layer cache is scoped per image and lifecycle.
- **CARGO_BUILD_JOBS:** Set to 20 on push (8 on PR) for higher parallelism.
- **Symbol verification:** Rust build stages verify the `.so` has exported symbols using `nm -D` to catch linking issues early.
- **Build time metrics:** Each run reports build time in the job step summary and as a GitHub notice.
