# AMD MIGraphX ONNX Runtime Provider Strategy

This note defines the AMD provider policy for router-owned ONNX signal models.
It covers mmBERT embedding, mmBERT sequence classifiers, PII token classifiers,
and multimodal ONNX paths owned by `onnx-binding`.

## Provider Order

AMD ONNX sessions use this provider order:

```text
MIGraphXExecutionProvider -> ROCmExecutionProvider -> CPUExecutionProvider
```

The Rust binding registers MIGraphX through ONNX Runtime's generic
`SessionOptionsAppendExecutionProvider` API. This avoids relying on the typed
`OrtMIGraphXProviderOptions` layout, which can drift across dynamically loaded
AMD ORT builds.

The `Auto`/GPU paths for embedding, classifier, token-classifier, and multimodal
sessions all use the shared AMD helper in `src/core/ort_migraphx.rs`.

## Classifier Artifact Order

mmBERT sequence and token classifiers choose ONNX artifacts in provider-aware
order:

```text
AMD Auto/ROCm: model_sdpa_fp16.onnx -> model.onnx -> classifier.onnx -> model_optimized.onnx
CPU/CUDA/OpenVINO: model.onnx -> classifier.onnx -> model_optimized.onnx -> model_sdpa_fp16.onnx
```

`model.onnx` remains the CPU baseline artifact. AMD paths prefer
`model_sdpa_fp16.onnx` when it is present because MIGraphX 2.14 with AMD's ORT
MIGraphX 1.23.x wheel compiles and runs that artifact for the mmBERT32k intent
classifier across seq=1/32/512. Raw sequence-classifier baseline artifacts such
as intent, factcheck, and feedback `model.onnx` compile for seq=1 but fail at
seq=32/512 with a MIGraphX `MULTIBROADCAST` shape error in the final masked
mean-pooling pattern. To avoid runtime failures, sequence classifiers force CPU
for baseline artifacts on AMD paths when no `model_sdpa_fp16.onnx` or compatible
optimized artifact is present. Token classifiers keep raw baseline artifacts
eligible for MIGraphX because the PII token classifier does not use the same
sequence pooling graph and has been validated under MIGraphX.

## CK FlashAttention Exception

CK FlashAttention optimized ONNX artifacts are ROCm-only today because the
custom op library is registered through ONNX Runtime's ROCm path. When
`ORT_CK_FLASH_ATTN_LIB` is set and the selected ONNX filename is an FA artifact
such as `model_fa_fp16.onnx`, the AMD helper skips MIGraphX and uses:

```text
ROCmExecutionProvider -> CPUExecutionProvider
```

This is intentional provider ownership, not an accidental fallback. Portable
`model.onnx` artifacts remain below `model_sdpa_fp16.onnx` on AMD paths even
when `ORT_CK_FLASH_ATTN_LIB` is set.

## Runtime Build

The AMD runtime choice is:

```text
ROCm 7.1 + onnxruntime_migraphx 1.23.x + onnx-binding migraphx-dynamic
```

The production ROCm Dockerfiles install AMD's `onnxruntime_migraphx` wheel and
build `onnx-binding` with `migraphx-dynamic`. The dynamic ORT path is provided
through `ORT_DYLIB_PATH`, and `LD_LIBRARY_PATH` includes both ROCm libraries and
the ORT `capi` directory.

## Operator Diagnostics

To verify the runtime inventory inside an AMD image or validation host:

```bash
migraphx-driver --version
python -c 'import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())'
echo "$ORT_DYLIB_PATH"
```

To verify provider ownership for mmBERT ONNX artifacts:

```bash
cd onnx-binding
./scripts/run_mmbert_migraphx_bench.sh \
  --providers cpu,auto,migraphx,rocm \
  --layers 6,11,16,22 \
  --seq-lens 1,32,128,512 \
  --batch-sizes 1,4 \
  --concurrency 2,4 \
  --jsonl /tmp/mmbert-provider-bench.jsonl \
  --summary-md /tmp/mmbert-provider-bench.md
```

The JSONL emits `provider`, `selected_provider`, `fallback_reason`,
session-create time, first inference time, warm P50/P95/P99, throughput,
concurrency throughput, VRAM percentage samples from `rocm-smi`, output shape,
and CPU parity drift.

Additional router-owned classifier artifacts can be included with repeated
`--model name=/path/to/model.onnx` flags, for example:

```bash
./scripts/run_mmbert_migraphx_bench.sh \
  --model intent=/models/mmbert32k-intent/onnx/model.onnx \
  --model pii=/models/pii-token-classifier/onnx/model.onnx \
  --providers cpu,auto
```

## Validation Matrix

Issue-level validation should cover:

- mmBERT embedding / 2D Matryoshka layer exits.
- mmBERT32k intent classifier.
- PII token classifier.
- factcheck classifier.
- feedback classifier.
- multimodal ONNX smoke path.
- portable ONNX with MIGraphX-first ownership.
- CK FlashAttention optimized ONNX with ROCm ownership.
- CPU fallback when AMD providers are unavailable.
- long-context and higher-batch/high-concurrency mixes.

Missing model artifacts must be recorded in follow-up issues rather than silently
skipped.

## Known Limits

- MIGraphX cold compile can be much slower than warm inference; benchmark both
  session creation and warm latency.
- Raw mmBERT32k sequence classifiers that use final masked mean pooling
  currently hit a MIGraphX `MULTIBROADCAST` shape error for seq=32/512. This
  has been reproduced for intent, factcheck, and feedback `model.onnx`
  artifacts. Use `model_sdpa_fp16.onnx` for AMD/MIGraphX acceleration when
  available; otherwise sequence classifiers fall back to CPU for baseline
  artifacts.
- CK FlashAttention custom-op artifacts are ROCm-owned until MIGraphX can load
  the same custom-op path.
- AMD's `onnxruntime_migraphx` 1.23.x wheel exposes MIGraphX and CPU providers,
  but not ROCm EP. CK FlashAttention validation therefore still needs a ROCm ORT
  runtime or a follow-up runtime image that carries both requirements.
- ORT/MIGraphX compatibility is tied to the ROCm and wheel combination; use the
  runtime inventory commands above before comparing benchmark numbers.
