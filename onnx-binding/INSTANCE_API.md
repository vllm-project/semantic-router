# Owned ORT instances

The `instance` Go package prepares independent sequence, token, embedding, and
multimodal model handles. `Clone` shares the physical native session; closing an
owner cannot unload a session used by another owner or an active call. Inputs
and outputs are task-specific, and actual token truncation is reported.

```go
model, err := instance.LoadSequenceClassifier(instance.Options{
    ModelPath: "models/classifier", Provider: "cpu", Precision: "native",
    MaxInputTokens: 128, Overflow: "reject",
})
if err != nil { return err }
defer model.Close()
result, err := model.Classify("Explain photosynthesis")
```

Build with `cargo build --release --lib --locked --no-default-features --features dynamic`
for CPU, or `--features migraphx-dynamic` for AMD. These builds export only
`ort_instance_*` symbols and can link in the same process as Candle. The default
`legacy-ffi` feature preserves the older standalone binding build.

Set `ORT_DYLIB_PATH` to the actual ONNX Runtime shared library. Its provider
libraries and ROCm libraries must also be on the dynamic loader search path.
CPU requires device 0 and `native` precision. MIGraphX accepts an explicit device
index and `native` or `fp16`; native preserves the graph's existing tensor types.
MIGraphX registration and session preparation are strict: CPU execution fallback
is disabled. `Info` reports the runtime build and session configuration;
`FinishProfiling` returns real ORT profiles when `ProfilePrefix` is configured.
Owned ROCm sessions use `kSameAsRequested` arena growth so bounded attention
subgraphs do not reserve geometrically larger GPU buffers. The allocation
policy is recorded in execution evidence and representation identity.

The ROCm 7.0 ORT 1.22.1 build at `ROCm/onnxruntime@2716b9b93a` changes the upstream
MIGraphX options C layout despite keeping API version 22. Its bridge is selected
only by the runtime's matching build commit. Other builds must support the
stable named provider-options API, otherwise preparation fails explicitly.
There are no registry patches or guessed provider layouts.

`go test ./instance` uses small input-dependent ONNX fixtures for ownership,
concurrent close, token spans, modality paths, and CPU execution evidence.
`ORT_TEST_MIGRAPHX=1` adds strict GPU execution checks.
`ORT_MAINTAINED_MODELS_ROOT` enables opt-in maintained-checkpoint correctness
checks; callers must supply revision-locked artifacts. Set
`ORT_PROFILE_EVIDENCE_DIR` to retain the resulting real GPU profiles.

`LoadOmni` loads the versioned `vela_omni_manifest.json` contract. It verifies
four graphs (text, image, CLAP, audio fusion), all external tensors, tokenizer,
processor digests, and the source-bound reference parity receipt before warming
all modalities. A raw Hugging Face snapshot is not a prepared artifact. Build
bundles with `tools/models/vela_omni/prepare.py`; the image includes Nano at
`/opt/router-model-artifacts/vela-1.0-omni-nano`, with Mini available through the
`VELA_OMNI_VARIANTS="nano mini"` build argument. Router provisioning validates and
atomically copies these bundles into the configured model cache. Set
`ROUTER_MODEL_ARTIFACTS` to use another prepared bundle directory.

`EncodeText`, `EncodeImageBytes`, and `EncodeAudioPCM` share the artifact's native
normalized space: Nano is 384 dimensions and Mini is 768. Dimension zero selects
that space; unsupported dimensions and early exit layers are errors. Text
inputs exceeding the prepared budget are rejected. Mini applies the published
whitespace stripping without an implicit instruction prefix. An optional fixed
execution budget pads according to the artifact and must be explicitly selected;
Mini's 32768-token maximum does not cause every request to be padded to 32768.
CPU deployments keep dynamic sequence lengths. Router GPU deployments require
an explicit binding `input.max_tokens`; the adapter resolves that value as the
fixed execution shape before partitioning and includes it in session identity.
Direct binding callers set both `MaxInputTokens` and `ExecutionMaxInputTokens`.
GPU preparation retains the strict prohibition on CPU execution fallback.

Raw PCM is finite float32 in channels-first order, one to eight channels, at most
30 seconds. Nano accepts 16000, 44100, and 48000 Hz; Mini accepts positive integer
rates up to 384000 Hz. Each branch resamples the original waveform independently,
then converts to mono: Whisper uses 16 kHz and CLAP uses 48 kHz. `Info` reports the
actual modalities, dimensions, token budget, and audio limits. Processor contents
participate in runtime identity alongside graph and provider contents.

`VELA_OMNI_ARTIFACT=/path/to/bundle go test ./instance -run TestPublishedOmniParity`
checks real text, image, and raw-audio vectors against source-reference goldens.
Prepare that test bundle with `--keep-golden`. Normal unit tests use tiny synthetic
ONNX fixtures and compact numeric DSP references without model downloads.
For GPU qualification, also set `VELA_OMNI_PROVIDER=rocm`,
`VELA_OMNI_EXECUTION_TOKENS` to the explicit deployment budget, and
`VELA_OMNI_PROFILE` to a writable file prefix. The test audits every graph's
profile, including nested control-flow bodies, and rejects CPU execution.
Goldens beyond an explicitly smaller deployment budget must return an input
limit error and do not count as numerical qualification. Run
`TestPublishedOmniFullContext` separately with a full-context reference bundle:
CPU uses dynamic shapes, and GPU requires the explicit maximum execution budget.
This test compares the longest golden at the artifact's exact public token limit.
An independent pinned-source reference can be supplied with
`VELA_OMNI_FULL_CONTEXT_REFERENCE`; its source revision must match the sealed
artifact. This does not change or replace the artifact's export parity receipt.

`TestPublishedGroundedParity` checks Halu answer spans against the export's
`halu_reference.json`. Set `VELA_HALU_ARTIFACT`, and for strict GPU evidence also
set `VELA_HALU_PROVIDER=rocm` and `VELA_HALU_PROFILE`. It executes the explicit
8192-token physical budget, preserving original answer offsets and token usage.
