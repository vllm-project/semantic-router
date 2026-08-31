# Candle native binding

`candle-binding` is the router's Rust/CGo inference layer for embeddings,
classifiers, multimodal encoders, hallucination checks, and MLP model selection.
It is a library used by the Go router, not a standalone server.

## Build and test

The module requires Go 1.24.1 or newer, Rust, Cargo, and a working C compiler.
The default Cargo feature enables CUDA. Use an explicit CPU feature set on a
machine without CUDA.

```bash
cd candle-binding

# Default CUDA build
cargo build --release

# CPU build on Linux
cargo build --release --no-default-features

# Metal GPU build on Apple Silicon
cargo build --release --no-default-features --features metal

# Rust tests
cargo test --no-default-features
```

Go tests link against the native library:

```bash
cd candle-binding
export LD_LIBRARY_PATH="$PWD/target/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
go test ./...
```

On macOS, use `DYLD_LIBRARY_PATH` instead of `LD_LIBRARY_PATH`. Build with
`--no-default-features --features metal` to run on the Apple Silicon GPU, or
`--no-default-features --features accelerate` for CPU inference through
Accelerate. Metal inference runs on a fixed thread pool of
`METAL_MAX_CONCURRENCY` threads (default 8, max 32) because candle's Metal
backend keeps one command buffer per OS thread and its single command queue
deadlocks at 64 in-flight buffers.

From the repository root, the maintained test entry points are:

```bash
make test-binding-minimal
make test-binding-lora
```

The multimodal suite needs model files and is documented in
[`tools/agent/docs/testing-strategy.md`](../tools/agent/docs/testing-strategy.md#model-gated-multimodal-tests).

## Public Go surface

[`semantic-router.go`](semantic-router.go) owns the supported Go wrappers. It
includes model initialization, text and multimodal embeddings, similarity,
intent and safety classifiers, LoRA classifiers, and MLP selection. Call each
model's initialization function before inference and release returned native
resources as documented by the wrapper.

## Troubleshooting

- `library 'candle_semantic_router' not found`: build the release library and
  add `target/release` to the platform library path.
- CUDA build failures on a CPU host: use `--no-default-features`.
- Model-loading or network failures: tests that use real models may download
  artifacts; use the repository Make targets to get the expected fixtures.

The feature matrix and dependency versions are defined in
[`Cargo.toml`](Cargo.toml); do not duplicate them in this README.
