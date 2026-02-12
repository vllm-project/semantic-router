# Dockerfiles

This directory contains Dockerfiles used across the project.

- `tools/docker/Dockerfile`: development base image (CentOS Stream) with toolchains (Rust, Go, Envoy, HF CLI).
- `tools/docker/Dockerfile.extproc`: builds the `extproc` image (single-platform, used for PR amd64-only builds).
- `tools/docker/Dockerfile.extproc.cross`: cross-compilation optimized `extproc` Dockerfile (used for multi-arch push/dispatch builds).
- `tools/docker/Dockerfile.precommit`: pre-commit / lint tooling image for CI and local use.
- `tools/docker/Dockerfile.stack`: single-image "stack" build bundling router + dashboard + observability components.

## Build optimization (CI)

The workflow [.github/workflows/docker-publish.yml](../../.github/workflows/docker-publish.yml) builds multi-arch images (e.g. vllm-sr, extproc) on push to `main`. To keep build times under **60 minutes** per image:

### Architecture: Per-platform parallel builds

Instead of building both amd64 and arm64 in a single Docker Buildx invocation (which serializes the builds), the workflow splits into **per-platform parallel jobs**:

1. **`build_platform` job** (matrix: image × platform): Builds each image for each platform independently. amd64 and arm64 builds run in parallel on separate runners.
2. **`create_manifest` job**: After all platform builds complete, creates multi-arch manifest lists that combine the per-platform images.

For **PR builds**, only amd64 is built (via `build_pr` job) for fast feedback.

### Cross-compilation (no QEMU emulation)

- **extproc** uses `Dockerfile.extproc.cross` for push/dispatch builds. The Rust and Go stages use `--platform=$BUILDPLATFORM` to run natively on amd64 and cross-compile for arm64 using `gcc-aarch64-linux-gnu`.
- **vllm-sr** uses `TARGETARCH` in its Dockerfile so Rust and Go stages run on `--platform=$BUILDPLATFORM` and cross-compile for arm64 when needed.
- This avoids the 10-40x slowdown of QEMU emulation for Rust/Go compilation.

### Rust dependency pre-caching

All Dockerfiles use a two-step Rust build pattern:

1. Copy `Cargo.toml` + `Cargo.lock`, create a dummy `lib.rs`, and build dependencies (cached Docker layer).
2. Copy real source, **delete stale `.so`/`.a`** from the dummy build, and rebuild (only recompiles application code).

The stale library deletion (`find target -name "libcandle_semantic_router.so" -delete`) is critical: without it, cargo's incremental compilation may reuse the empty library from the dummy build, causing linker errors in the Go build stage.

### Other optimizations

- **No `cargo clean`:** Dependency cache from the pre-build layer is reused; only application code is recompiled.
- **Job timeouts:** Platform builds have a 90-minute timeout; PR builds have 60 minutes.
- **Per-platform GHA cache:** Docker layer cache is scoped per image and platform (`scope=$image-$platform`) for better cache hit rates.
- **CARGO_BUILD_JOBS:** Set to 20 on push (8 on PR) for higher parallelism.
- **Symbol verification:** Rust build stages verify the `.so` has exported symbols using `nm -D` to catch linking issues early.
- **Build time metrics:** Each run reports build time in the job step summary and as a GitHub notice.
