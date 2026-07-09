# AMD MIGraphX ONNX Runtime Provider Strategy

This note defines the AMD provider policy for router-owned ONNX signal models.
It covers mmBERT embedding, mmBERT sequence classifiers, PII token classifiers,
and multimodal ONNX paths owned by `onnx-binding`.

## Provider Order

AMD ONNX sessions use this provider order when the loaded ONNX Runtime exposes
all AMD providers:

```text
MIGraphXExecutionProvider -> ROCmExecutionProvider -> CPUExecutionProvider
```

The AMD router image installs AMD's `onnxruntime_rocm` 1.22.x wheel together
with the system `migraphx` runtime package. That combination exposes all three
providers. AMD's newer `onnxruntime_migraphx` 1.23.x wheel exposes MIGraphX and
CPU providers but not ROCm EP, so it is useful for MIGraphX-only validation but
cannot run ROCm-owned CK FlashAttention artifacts.

The Rust binding registers MIGraphX through ONNX Runtime's generic
`SessionOptionsAppendExecutionProvider` API. This avoids relying on the typed
`OrtMIGraphXProviderOptions` layout, which can drift across dynamically loaded
AMD ORT builds.

The `Auto`/GPU paths for embedding, classifier, token-classifier, and multimodal
sessions all use the shared AMD helper in `src/core/ort_migraphx.rs`.

## Classifier Artifact Order

mmBERT sequence classifiers choose ONNX artifacts in provider-aware order:

```text
AMD Auto/ROCm: model_sdpa_fp16.onnx -> model_fa_fp16.onnx -> model_fa.onnx -> model.onnx -> classifier.onnx -> model_optimized.onnx
CPU/CUDA/OpenVINO: model.onnx -> classifier.onnx -> model_optimized.onnx -> model_sdpa_fp16.onnx
```

`model.onnx` remains the CPU baseline artifact. AMD paths prefer
`model_sdpa_fp16.onnx` when it is present so portable classifiers use MIGraphX
before considering ROCm-owned CK FlashAttention artifacts. Remote validation
with ROCm 7.0, ONNX Runtime ROCm 1.22.x, and MIGraphX 2.13 loaded and ran the
intent, jailbreak, factcheck, and feedback SDPA artifacts at seq=32/512 with
finite logits. Raw sequence-classifier baseline artifacts such as intent,
factcheck, and feedback `model.onnx` compile for seq=1 but fail at seq=32/512
with a MIGraphX `MULTIBROADCAST` shape error in the final masked mean-pooling
pattern. To avoid runtime failures, sequence classifiers force CPU for baseline
artifacts on AMD paths when no `model_sdpa_fp16.onnx` or compatible optimized
artifact is present.

PII token classifiers use a separate token-safe artifact order:

```text
AMD Auto/ROCm default with CK-FA available: model_fa_fp16.onnx -> model_fa.onnx -> model.onnx -> token_classifier.onnx -> classifier.onnx -> model_optimized.onnx -> model_token_sdpa.onnx -> token_classifier_sdpa.onnx -> model_token_sdpa_fp16.onnx -> token_classifier_sdpa_fp16.onnx -> model_token_eager.onnx -> token_classifier_eager.onnx -> model_token_eager_fp16.onnx -> token_classifier_eager_fp16.onnx
AMD Auto/ROCm default without CK-FA: model.onnx -> token_classifier.onnx -> classifier.onnx -> model_optimized.onnx -> model_token_sdpa.onnx -> token_classifier_sdpa.onnx -> model_token_sdpa_fp16.onnx -> token_classifier_sdpa_fp16.onnx -> model_token_eager.onnx -> token_classifier_eager.onnx -> model_token_eager_fp16.onnx -> token_classifier_eager_fp16.onnx
AMD Auto/ROCm with VSR_ENABLE_EXPERIMENTAL_MIGRAPHX_TOKEN_ARTIFACTS=1 and MIGRAPHX_MLIR_USE_SPECIFIC_OPS=~attention: model_token_sdpa.onnx -> token_classifier_sdpa.onnx -> model_token_sdpa_fp16.onnx -> token_classifier_sdpa_fp16.onnx -> model_token_eager.onnx -> token_classifier_eager.onnx -> model_token_eager_fp16.onnx -> token_classifier_eager_fp16.onnx -> model.onnx -> token_classifier.onnx -> classifier.onnx -> model_optimized.onnx
CPU/CUDA/OpenVINO: model.onnx -> token_classifier.onnx -> classifier.onnx -> model_optimized.onnx -> model_token_sdpa.onnx -> token_classifier_sdpa.onnx -> model_token_sdpa_fp16.onnx -> token_classifier_sdpa_fp16.onnx -> model_token_eager.onnx -> token_classifier_eager.onnx -> model_token_eager_fp16.onnx -> token_classifier_eager_fp16.onnx
```

Do not reuse sequence-classifier `model_sdpa_fp16.onnx` for token
classification. The public PII `model_sdpa_fp16.onnx` artifact emits
`[batch, num_labels]` logits rather than `[batch, seq_len, num_labels]`, so it
is a sequence-classification artifact and fails token-level quality validation.
The public PII `model_fa_fp16.onnx` artifact is token-shaped and ROCm-owned when
the CK FlashAttention custom op is available. Raw token baseline artifacts such
as public PII `model.onnx` remain CPU-owned on AMD paths unless a token-specific
optimized artifact passes output-shape and quality gates. Remote AI4Privacy
validation found `model.onnx` under MIGraphX can diverge from CPU token
entities, so it is not used as the default AMD fallback. Token eager and SDPA
artifacts are recognized for fixed-padding execution, but MIGraphX ownership is guarded by
`VSR_ENABLE_EXPERIMENTAL_MIGRAPHX_TOKEN_ARTIFACTS` plus either
`MIGRAPHX_MLIR_USE_SPECIFIC_OPS=~attention` or `MIGRAPHX_DISABLE_MLIR=1` until
entity-level validation reaches parity. The narrower `~attention` setting is
preferred over disabling all MLIR. Token FP16 artifacts remain supported for
experiments, but PII BIO boundaries are sensitive to FP16 logit drift; the
default token export emits `model_token_sdpa.onnx` in FP32, and
`--token-optimized-attn eager` can generate `model_token_eager.onnx` for
graph-workaround experiments.

## CK FlashAttention Exception

CK FlashAttention optimized ONNX artifacts are ROCm-only today because the
custom op library is registered through ONNX Runtime's ROCm path. When
`ORT_CK_FLASH_ATTN_LIB` is set, the loaded ORT exposes
`ROCMExecutionProvider`, and the selected ONNX filename is an FA artifact such
as `model_fa_fp16.onnx`, the AMD helper skips MIGraphX and uses:

```text
ROCmExecutionProvider -> CPUExecutionProvider
```

This is intentional provider ownership, not an accidental fallback. FA
artifacts are skipped when ROCm EP is absent, even if the custom-op library is
present in the image. Portable `model.onnx` artifacts remain below
`model_sdpa_fp16.onnx` on AMD paths.

## Runtime Build

The AMD runtime choice is:

```text
ROCm 7.0 + onnxruntime_rocm 1.22.x + system migraphx + onnx-binding migraphx-dynamic
```

The production ROCm Dockerfiles install AMD's `onnxruntime_rocm` wheel and the
system `migraphx` package, then build `onnx-binding` with `migraphx-dynamic`.
The ROCm wheel contains both `libonnxruntime_providers_rocm.so` and
`libonnxruntime_providers_migraphx.so`; the `migraphx` package supplies the
`libmigraphx_c.so.3` dependency needed to register MIGraphX. The dynamic ORT
path is provided through `ORT_DYLIB_PATH`, and `LD_LIBRARY_PATH` includes both
ROCm libraries and the ORT `capi` directory.

The default AMD runtime image sets `ORT_CK_FLASH_ATTN_LIB` because ROCm EP is
present. Sequence classifiers still try portable SDPA artifacts first; embedding
and other CK-FA-only artifacts use the ROCm-owned path.

## Cold Compile Mitigation

MIGraphX is a graph compiler, so cold session creation or first use can be much
slower than warm inference. The router-owned sequence classifier path mitigates
the most common source of repeated compile cost by bucketing AMD
`model_sdpa_fp16.onnx` inputs. The default buckets are:

```text
[batch, 128]
[batch, 512]
```

This applies only to AMD portable SDPA sequence-classifier artifacts. CPU paths,
CK FlashAttention artifacts, and PII token classifiers keep their existing input
contracts. Operators can override the buckets, use a single 512-token bucket, or
opt out for dynamic-shape debugging with:

```bash
VSR_AMD_MIGRAPHX_SEQUENCE_BUCKETS=64,128,512
VSR_AMD_MIGRAPHX_SEQUENCE_BUCKETS=512
VSR_AMD_MIGRAPHX_SEQUENCE_BUCKETS=dynamic
```

To move the remaining compile or first-run cost from the first user request into
startup, enable explicit warmup:

```bash
VSR_AMD_MIGRAPHX_WARMUP=1
```

Warmup runs one dummy sequence-classifier inference per configured bucket after
the ONNX session is created. It intentionally increases startup time when
enabled, but makes the first real request for warmed buckets more predictable.

The MIGraphX EP also exposes cache-related environment variables such as
`ORT_MIGRAPHX_CACHE_PATH` and `ORT_MIGRAPHX_MODEL_CACHE_PATH`. Keep those as
operator experiments until cache invalidation has been validated across model
hashes, GPU architecture, ROCm, ORT, and MIGraphX versions.

Do not set `MIGRAPHX_MLIR_USE_SPECIFIC_OPS=~attention` globally in the AMD
runtime image yet. That setting is currently a token-classifier workaround, not
a validated global runtime default for every router-owned ONNX artifact. Operators
who explicitly validate PII token optimized artifacts can set it together with
`VSR_ENABLE_EXPERIMENTAL_MIGRAPHX_TOKEN_ARTIFACTS=1`; otherwise token optimized
artifacts remain CPU-owned on AMD.

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

Provider/performance validation is not enough to prove a migration is safe. Run
the quality regression harness on labeled task data before switching a signal to
an optimized artifact:

```bash
python onnx-binding/scripts/eval_router_signal_artifacts.py \
  --task intent \
  --task-type sequence \
  --model-dir /models/mmbert32k-intent-classifier-merged \
  --dataset e2e-domain \
  --run old-runtime-old-onnx=model.onnx:cpu:old-runtime \
  --run new-runtime-old-onnx=model.onnx:cpu:new-runtime \
  --run new-runtime-new-onnx=model_sdpa_fp16.onnx:cpu:new-runtime \
  --run migrated-auto=model_sdpa_fp16.onnx:auto:new-runtime \
  --baseline-run old-runtime-old-onnx \
  --jsonl /tmp/intent-artifact-eval.jsonl \
  --summary-md /tmp/intent-artifact-eval.md \
  --fail-on-regression
```

For token classifiers, compare `model.onnx` on CPU and only compare AMD against
a token-specific optimized artifact after it has passed both output-shape and
quality gates:

```bash
python onnx-binding/scripts/eval_router_signal_artifacts.py \
  --task pii \
  --task-type token \
  --model-dir /models/mmbert32k-pii-detector-merged \
  --dataset e2e-pii \
  --run old-runtime-old-onnx=model.onnx:cpu:old-runtime \
  --run new-runtime-old-onnx=model.onnx:cpu:new-runtime \
  --baseline-run old-runtime-old-onnx \
  --jsonl /tmp/pii-artifact-eval.jsonl \
  --summary-md /tmp/pii-artifact-eval.md \
  --fail-on-regression
```

The required comparison is:

- `new-runtime-old-onnx` vs `old-runtime-old-onnx`: the new runtime architecture
  must not regress the existing artifact.
- `new-runtime-new-onnx` vs `old-runtime-old-onnx`: the optimized artifact must
  not regress baseline task quality on CPU.
- `migrated-auto` vs `new-runtime-new-onnx`: AMD provider execution must not
  change semantic results beyond acceptable fp16 drift.
- `migrated-auto` vs `old-runtime-old-onnx`: the final deployed path must not
  regress the original baseline.

Use task-native validation sets:

- intent/domain: MMLU-Pro gold JSONL from `perf/scripts/export_mmlu_intent_gold.py`
  plus `e2e-domain` smoke cases.
- feedback: `hf:llm-semantic-router/feedback-detector-dataset:validation`.
- factcheck: `factcheck-nisq`, a deterministic label-stratified slice from
  `YaoSun0422/NISQ_dataset` where `ISQ` maps to `FACT_CHECK_NEEDED` and all
  other labels map to `NO_FACT_CHECK_NEEDED`.
- jailbreak: `jailbreak-toxic-chat`, `jailbreak-salad`, or
  `jailbreak-mixed`. `jailbreak-toxic-chat` uses the
  `lmsys/toxic-chat:toxicchat0124:test` `jailbreaking` label, `jailbreak-salad`
  uses `OpenSafetyLab/Salad-Data:attack_enhanced_set:train` as positive
  jailbreak prompts, and `jailbreak-mixed` builds a deterministic balanced
  slice from ToxicChat negatives plus ToxicChat/Salad positives.
- PII: AI4Privacy/Presidio validation rows with entity spans, plus `e2e-pii`
  smoke cases. PII must be judged by entity-level precision/recall/F1, not only
  by blocked/not-blocked detection rate.

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

Current remote quality evidence from the ROCm/MIGraphX validation host:

- factcheck NISQ20 fixed-shape matrix:
  `/tmp/sr-quality-factcheck-nisq20-fixed-20260708-171406`. `model.onnx` CPU,
  `model_sdpa_fp16.onnx` CPU, and `model_sdpa_fp16.onnx` auto/MIGraphX all
  matched the baseline labels at `1.0`; auto selected
  `MIGraphXExecutionProvider, CPUExecutionProvider`; max absolute logit drift
  was `0.02563619613647461`.
- jailbreak mixed20 fixed-shape matrix:
  `/tmp/sr-quality-jailbreak-mixed20-fixed-20260708-171644`. `model.onnx` CPU,
  `model_sdpa_fp16.onnx` CPU, and `model_sdpa_fp16.onnx` auto/MIGraphX all
  matched the baseline labels at `1.0`; auto selected
  `MIGraphXExecutionProvider, CPUExecutionProvider`; max absolute logit drift
  was `0.1847209930419922`.

## Known Limits

- MIGraphX cold compile can be much slower than warm inference; benchmark both
  session creation and warm latency. AMD SDPA sequence classifiers now use
  `128,512` input buckets by default to avoid one compile per observed request
  length while keeping short-request latency lower than an always-512 shape. Set
  `VSR_AMD_MIGRAPHX_WARMUP=1` to pay first-run warmup before serving real
  requests for the configured buckets.
- Raw mmBERT32k sequence classifiers that use final masked mean pooling
  currently hit a MIGraphX `MULTIBROADCAST` shape error for seq=32/512. This
  has been reproduced for intent, factcheck, and feedback `model.onnx`
  artifacts. Use `model_sdpa_fp16.onnx` for AMD/MIGraphX acceleration when
  available; otherwise sequence classifiers fall back to CPU for baseline
  artifacts.
- Public PII `model_sdpa_fp16.onnx` is not token-classification compatible: it
  emits `[batch, num_labels]` logits. Public PII `model.onnx` is also not
  promoted to MIGraphX by default: a small AI4Privacy span-level validation
  matched CPU for 4/5 dynamic-shape rows but failed one row, and fixed 512
  padding produced empty MIGraphX entities for 4/5 rows. Use a token-safe
  `model_token_*.onnx` artifact for AMD/MIGraphX acceleration once it passes
  entity-level quality gates; keep `model.onnx` as the CPU fallback artifact.
- Token-specific FP32 exports are not MIGraphX-safe with default MIGraphX 2.14
  codegen. `model_token_sdpa.onnx` and `model_token_eager.onnx` both match the
  old CPU ONNX artifact on the sampled AI4Privacy rows when run on CPU, but
  default MIGraphX produces non-finite logits. Native
  `migraphx-driver verify --onnx --gpu` reproduces the `model_token_eager.onnx`
  NaN and `--bisect` reports the failure starts at a ModernBERT attention slice
  feeding the fused RoPE/QK/softmax/value codegen path. Setting
  `MIGRAPHX_MLIR_USE_SPECIFIC_OPS=~attention` avoids that fused attention path:
  `model_token_sdpa.onnx` matched old CPU ONNX entities for 500/500 AI4Privacy
  rows, `model_token_eager.onnx` matched for 20/20 rows, and native
  `migraphx-driver verify` passed for `model_token_eager.onnx`. The Rust
  classifier keeps token optimized artifacts CPU-owned on AMD unless both the
  experimental token-artifact opt-in and the attention workaround are present.
  Keep token optimized artifacts experimental until this workaround is validated
  at a larger scale and exposed through a documented runtime strategy or
  replaced by a graph rewrite/MIGraphX fix.
- PII token SDPA performance with the `~attention` workaround is promising but
  still has cold-compile and harness limits. A fixed-shape batch=1/seq=512 run
  measured CPU warm avg/P95 at `415.136/575.118 ms` and MIGraphX warm avg/P95 at
  `11.837/16.430 ms` with max/mean CPU drift `0.000053/0.000008`. MIGraphX
  concurrency 2/4 reached `373.5/395.8` items/s after sessions were created,
  but the current benchmark cold-compiles one MIGraphX session per worker, so
  concurrency=8 was interrupted before completion. Treat high-concurrency
  numbers as incomplete until the harness can separate compile cache behavior
  from steady-state throughput.
- CK FlashAttention custom-op artifacts are ROCm-owned until MIGraphX can load
  the same custom-op path.
- AMD's `onnxruntime_migraphx` 1.23.x wheel exposes MIGraphX and CPU providers,
  but not ROCm EP. The default AMD router image therefore uses AMD's
  `onnxruntime_rocm` 1.22.x wheel plus the `migraphx` runtime package to expose
  MIGraphX, ROCm, and CPU in one image.
- ORT/MIGraphX compatibility is tied to the ROCm and wheel combination; use the
  runtime inventory commands above before comparing benchmark numbers.
