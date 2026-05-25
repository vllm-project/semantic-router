# probe-2026-05-25-image-drift-isolation

Goal: localize where the residual ~0.8% post-normalization drift between Python
reference and candle-binding's mmes pipeline actually lives.

Companion to the 2026-05-21 sister-ships AAR / session log, which named the
investigation path verbatim:

> "Dump 384-d image embeddings from both (candle-binding and Python), compute
> cosine on embeddings alone -> separates image-side vs text-side drift."

This probe goes further: within the image-side it also separates **preprocessing
drift** (Go bilinear vs PIL bicubic+antialias) from **model-forward drift**
(SigLIP vision tower + 768->384 projection) by feeding identical PIL-preprocessed
pixels into the Rust FFI directly via `MultiModalEncodeImage`, bypassing
`decodeAndResizeImage` at `semantic-router.go:1247`.

## The three vectors

All are 384-d L2-normalized image embeddings of the SAME input image:
`fixtures/inrule_identifier_passport.jpg` (the Wikipedia Taiwan MOFA passport
specimen used in the 2026-05-20 calibration pack and the 2026-05-21 mmes
reference run).

| Vector | Source | Pipeline |
|---|---|---|
| `embedding_python.npy`            | Python reference | PIL bicubic+antialias resize -> SigLIP normalize -> SigLIP vision -> 768->384 -> L2 |
| `embedding_candle_pilpipe.bin`    | candle-binding   | PIL bicubic+antialias resize -> Rust FFI `MultiModalEncodeImage` -> SigLIP normalize (Rust) -> SigLIP vision (Rust) -> 768->384 -> L2 |
| `embedding_candle_gopipe.bin`     | candle-binding   | Go `decodeAndResizeImage` (4-tap bilinear, no antialias) -> Rust FFI -> same Rust forward |

## The three diffs

| Comparison | What it isolates |
|---|---|
| Python vs Candle-PIL-pipe   | **Model-forward drift** alone. Same preprocessing on both sides. Difference is only fp32 vs fp32 accumulation across Python torch vs Rust candle. Should be < 1e-3 if model port is clean. |
| Candle-PIL vs Candle-Go     | **Preprocessing drift** alone. Same Rust forward on both sides. Difference is only PIL bicubic+antialias vs Go 4-tap bilinear. **This is the prime suspect for the 0.8%.** |
| Python vs Candle-Go-pipe    | **Total pipeline drift**. Should reproduce the on-record ~0.992 cosine. |

## Files in this probe

- `step1_extract_pixels_and_python_embedding.py` - PIL bicubic+antialias resize to 512x512, save as raw float32 CHW [0,1]; also runs Python mmes encode_image and saves the resulting 384-d embedding
- `probe_image_drift_isolation_2026_05_25_test.go` - lives in `candle-binding/`, not here. Reads the pixel buffer + the raw passport bytes, runs both Rust paths, writes two 384-d embeddings to this dir
- `step3_compare.py` - loads all three vectors, reports cosine sims + max abs diffs + L2 norms
- `run.sh` - orchestration: step1, then go test, then step3

## Hypotheses ranked (pre-experiment)

1. **Preprocessing is the entire residual drift.** Predicted result: Python vs Candle-PIL cosine ~0.999, Candle-PIL vs Candle-Go cosine ~0.992. **If this lands, the fix is to replace Go bilinear with a PIL-bicubic-antialias-equivalent resize.**
2. Model-forward drift contributes meaningfully. Predicted result: Python vs Candle-PIL cosine 0.995-0.998. Would point at attention scale order, gelu_erf vs gelu_pytorch_tanh, or layer-norm eps.
3. Both contribute roughly equally. Mixed signal; further bisect required.
