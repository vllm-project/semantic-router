# ONNX Runtime native binding

`onnx-binding` exposes ONNX Runtime embeddings and classifiers to the Go router.
It supports CPU execution by default and optional ROCm, MIGraphX, CUDA,
DirectML, or OpenVINO execution providers selected at build time.

The binding includes:

- text embeddings and similarity search;
- mmBERT 2D Matryoshka layer and dimension selection;
- intent, PII, jailbreak, fact-check, feedback, modality, and hallucination
  classifiers;
- text, image, and audio multimodal embeddings;
- Go wrappers over the native Rust library.

This README describes the developer workflow. Router configuration belongs in
the [learned-signal documentation](../website/docs/tutorials/signal/overview.md).

## Build

The module requires Go 1.21 or newer, Rust, Cargo, CGo, and a C compiler.

```bash
cd onnx-binding

# CPU build
cargo build --release

# Choose one optional execution provider when its runtime is installed
cargo build --release --features rocm
cargo build --release --features cuda
```

The available features and the pinned ONNX Runtime version live in
[`Cargo.toml`](Cargo.toml). Provider builds may require system libraries in
addition to the Rust feature.

## Test

Model-backed tests read their paths from environment variables. For example:

```bash
cd onnx-binding
export MMBERT_MODEL_PATH=/absolute/path/to/mmbert-onnx
export LD_LIBRARY_PATH="$PWD/target/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
go test ./...
```

On macOS, set `DYLD_LIBRARY_PATH` instead. From the repository root,
`make test-binding-minimal` is the maintained cross-binding entry point.

## Go API

[`semantic-router.go`](semantic-router.go) is the public API source of truth.
A typical embedding flow is:

```go
if err := onnx.InitMmBertEmbeddingModel(modelPath, true); err != nil {
    return err
}

embedding, err := onnx.GetEmbeddingDefault("route this request")
```

Use `GetEmbedding2DMatryoshka` when the exported model supports a specific
layer and dimension. Supported values come from the model artifact, not a
universal list in this module. Initialize a classifier before calling its
matching `Classify*` function.

## Benchmarking

The examples under [`examples/`](examples/) measure the checked-out code and
the model available on your machine:

```bash
cargo run --release --example benchmark_mmbert_latency -- /path/to/model
cargo run --release --example benchmark_cpu_vs_gpu -- /path/to/model
```

Do not treat historical numbers from a different host, provider, or model
export as a deployment guarantee. Record the model revision, execution
provider, CPU/GPU, sequence length, and batch size with any result.

The optional CK Flash Attention custom operator has its own
[build guide](ort-ck-flash-attn/README.md).
