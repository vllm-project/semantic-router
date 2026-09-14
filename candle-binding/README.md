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

[`instances.go`](instances.go) provides owned task handles for sequence and token
classification, NLI, hallucination detection, and embeddings. The generative
wrappers are in [`instances_generative.go`](instances_generative.go). Loaders
capture configuration, tokenizer, labels and input policy during preparation.
Call `Close()` when the binding is released; it waits for calls on that Go handle
to finish. An in-flight native call also holds its own strong model reference.

```go
model, err := candle_binding.LoadSequenceClassifier(candle_binding.InstanceOptions{
    ModelPath: "models/mmbert32k-intent-classifier-merged",
    Device: "cpu",
    Precision: "float32",
    MaxInputTokens: 512,
    Overflow: "reject",
})
if err != nil { return err }
defer model.Close()
result, err := model.Classify("Explain general relativity.")
```

Each load creates a separate native resource. `Clone()` makes an independently
closable binding to the same model. For ModernBERT/mmBERT, `BindSequenceHead` and
`BindTokenHead` explicitly reuse the source backbone while loading a different
head, tokenizer and label mapping. Architecture compatibility is checked before
publication. Merely using equal paths never establishes sharing. `Info()` reports
the effective device, precision, limits and physical `ResourceID`.

`LoadBackbone(options)` loads a ModernBERT/mmBERT encoder with only its
`config.json` and `model.safetensors`. It requires neither a classifier nor a
tokenizer in the base artifact. Bind a sequence or token head using the returned
`Backbone`; each head artifact supplies its own config, tokenizer, and head
weights. The two task types can be bound in either order, and a bound head remains
usable after the original backbone handle closes.

The owned loaders cover ModernBERT/mmBERT sequence and token checkpoints, BERT,
DeBERTa sequence classifiers, merged BERT LoRA (`ModelType: "bert_lora"`), the
maintained LoRA token wrapper (`"lora_token"`), and BERT/Qwen3/Gemma/mmBERT/multimodal
embeddings. A multimodal embedding handle supports text, PNG/JPEG image bytes and
mel spectrograms. Classification and token tasks keep the existing maximum of
512 tokens, including special tokens; a checkpoint name containing 32K does not
raise this task limit. Results include actual token counts and truncation state;
span offsets are UTF-8 byte ranges in the original text or hallucination answer.

Qwen3Guard returns its generated text without inventing a confidence. Qwen3 label
scoring retains the existing first-token or full-label scoring method. The
existing Qwen3 multi-adapter forward currently uses base-model logits and does
**not** apply loaded LoRA deltas; `AdapterWeightsApplied` is false and
`ScoreSemantics` identifies the scoring method. This ownership API does not change
that model behavior. Generative tasks reject inputs that exceed the complete
prompt/candidate budget or required output reserve. CPU uses float32; the existing
Qwen3 generative accelerator paths use bfloat16. Unsupported device/precision
requests fail explicitly.

[`semantic-router.go`](semantic-router.go) retains the legacy role-based wrappers
for consumers being migrated, plus the existing MLP selector handles. New
lifecycle-managed integrations should use owned task handles.

The deterministic native ownership suites execute small transformers loaded from
real safetensors; they do not qualify maintained checkpoint quality or GPU use:

```bash
cargo test --no-default-features --lib ffi::instances::tests
go test -race -run '^TestOwnedNative' ./...
```

The ignored `maintained_checkpoint_instance_regression` Rust test requires
`CANDLE_INSTANCE_SEQUENCE_MODEL` to name a prepared maintained checkpoint. It must
be run explicitly with `--ignored` when collecting that evidence.

## Troubleshooting

- `library 'candle_semantic_router' not found`: build the release library and
  add `target/release` to the platform library path.
- CUDA build failures on a CPU host: use `--no-default-features`.
- Model-loading or network failures: tests that use real models may download
  artifacts; use the repository Make targets to get the expected fixtures.

The feature matrix and dependency versions are defined in
[`Cargo.toml`](Cargo.toml); do not duplicate them in this README.
