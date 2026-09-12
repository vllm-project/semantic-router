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
