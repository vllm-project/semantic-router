# 0.6B Milestone 3: the padded-micro-batch "stall" — root cause

**Bottom line.** The 0.6B trainer has no padding defect. Padded and one-row
micro-batches give the same loss and gradients in FP32 (regression test and a
preflight gate now enforce it). The Milestone 2 claim that padded micro-batching
caused the stall is withdrawn. The bidirectional-Qwen and EuroBERT collapses are an
optimization failure of the fresh-head marker-readout recipe: they reproduce without
padding, in FP32, with a head trained first, and at a 4× lower backbone learning rate.
They are not implementation failures, and not evidence against those backbones either,
because the recipe never learned. Records: preregistration `m3-prereg-2026-09-28.md`,
amendment `m3a-amendment-2026-09-28.md`; probe `probe_padding.py` (commit `d87c98b94`),
parity gate `parity.py` and tests (commits `0738aeb90`, `a4f457f96`).

## 1. Parity: padded vs one-row on identical rows and weights

Probe on node A GPU0–1 (image runtime, Transformers 5.17 for the causal control and
4.57.6 for the encoder, as trained): both families rebuilt exactly as the trainer builds
them (`m2-qc-s1` causal endpoints + global query; `m2-qb-s1` bidirectional Qwen marker
readout), five multi-row micro-batches from the first two logical batches (3–8 rows,
29–3,643 tokens), each compared as one padded batch against the same rows one at a time.

| Setting | Loss rel. diff | Gradient rel. L2 error (padded vs one-row) | Hidden states (real tokens) |
| --- | --- | --- | --- |
| FP32, math or default SDPA | ≤ 5e-7 | ≤ 1e-4 | ≤ 1e-5 relative |
| BF16 autocast, default kernel (as trained) | ≤ 4e-3 | 6–61% | 1.6–3.1% relative |

Under BF16 each path must be measured against the FP32 one-row gradient instead:

| BF16 path vs FP32 reference | Causal control | Bidirectional Qwen |
| --- | --- | --- |
| padded, default (efficient) kernel | 9–13% | 5–21% |
| one-row, default kernel | 8–77% (77% on one Choice micro-batch) | 5–19% |
| padded, math kernel | 9–26% | 5–17% |
| one-row, math kernel | 7–13% | 5–28% |

The error sits in the attention and norm weights of layers 1–7, and which path is worse
flips with the kernel. The SDPA kernels alone, on random inputs of the same shapes, are
within 0.4% of FP32 whether padded or not. BF16 autocast on these backbones is simply a
5–25% gradient-noise regime; padding does not make it worse.

## 2. What the Milestone 2 runs actually show

- Same data order and head seed, padded QC s1 vs one-row QCMB1 s1: SELECT 255 vs 255
  at update 58 and 247 vs 258 at 116. QC s1 was still rising (311) when the preregistered
  collapse stop (350 at update 175) ended it; padded QCL s1 trained normally (521).
- Every collapsed run (bidirectional Qwen with marker, span or mean-token readouts;
  EuroBERT at both learning rates; padded QC) loses 5–20× of its gradient norm between
  updates 20 and 30, when the 5% warmup reaches the peak learning rate.

## 3. Pilots (Milestone 2 data and QB s1 recipe; one change each; stop SELECT < 350 at update 175)

| Pilot | Change | SELECT at 58 / 116 / 175 | Gradient norm after warmup | Outcome |
| --- | --- | --- | --- | --- |
| M2 QB s1 (reference) | — | 247 / 247 / 253 | 0.5–1.9 | collapsed |
| P1 `m3a-qbw-s1` | backbone frozen for updates 1–47 | 261 / 254 / 239 | 11–31 on release, 0.15 by update 80 | stop |
| P2 `m3a-qbmb1-s1` | one-row micro-batches (no padding) | 237 / 253 / 270 | 0.09–0.6 | stop |
| P4 `m3a-qbf32-s1` | FP32 compute (no autocast) | 238 / 279 / 260 | 0.1–1.1 | stop |
| P5 `m3a-qblr-s1` | backbone LR 5e-6, warmup 10% | 250 / 260 / 273 | 0.9–12 (no collapse) | stop |

In every pilot the per-batch training loss stays at the class prior. With the default
learning rate the backbone then makes all candidate vectors of a request identical (the
Milestone 2 probe measured marker cosine .9998 after training vs .975 at step 0), which
zeroes the head gradient Σₖ(pₖ−yₖ)f(cₖ); at a 4× lower rate the gradient survives but the
model still does not learn. This matches the known result that switching a causal LM to
bidirectional attention without an adaptation stage (for example LLM2Vec's masked
next-token training) degrades its representations. EuroBERT, natively bidirectional,
uses its MLM mask token as the marker, whose state predicts a token identity rather than
the candidate. A future bidirectional attempt needs such an adaptation stage or a readout
that is discriminative at step 0; that is a research item, not a trainer fix.

## 4. A real defect found by the new test

The causal family's teacher mapping crashed on any micro-batch that mixed rows with and
without teacher targets (`original_probabilities(..., None)`). Milestone 2 always had full
teacher coverage, so it never fired; the Milestone 3 mixture (Lux1 targets on A0s rows
only, none on A6 rows) would have crashed arm (a) at its preflight. Fixed in `a4f457f96`.

## 5. Regression test and gate

- `tests/test_padding_parity.py` (CPU, tiny random backbones, the families' real `loss`):
  causal endpoints + global query, bidirectional Qwen markers and ModernBERT markers,
  micro-batches mixing lengths, candidate counts and partial teacher coverage; FP32
  padded ≡ one-row (loss ≤ 1e-5, gradients ≤ 1e-3). Two negative controls must fail: an
  ignored key-padding mask and left padding with unshifted readout indices. 26/26 tests
  pass in both runtimes (Transformers 5.17 and 4.57.6).
- Preflight gate (`train.py --preflight`, `PADDING_PARITY.json`) on the real model, on
  two eight-row micro-batches of the first logical batch: FP32 loss ≤ 1e-5 and gradient
  relative error ≤ 1e-3; BF16 loss ≤ 1e-2 and padded-vs-FP32 gradient cosine ≥ 0.95.
  Every Milestone 3 preflight passed it (FP32 loss ≤ 7e-7, gradient ≤ 5e-5; BF16
  padded-vs-FP32 cosine .981–.999).

## 6. Classification of earlier runs

- EB610, EB610 half LR (M2a), QB s1/s2, QBS s1/s2, QBQ s1: **invalid as architecture
  evidence** — optimization failures of the fresh-head bidirectional marker recipe, not
  implementation failures. Not rerun unchanged (P1, P2, P4 and P5 show the rerun would
  collapse again).
- QC s1/s2 and QCL s2 (collapse stops): same class; the causal endpoint readout escapes
  stochastically (Milestone 2: one-row 4/4 trained, padded 1/4).
- Milestone 2 finding 2 ("padded micro-batching is the one identified factor") is
  withdrawn. One-row micro-batches remain in arm (a) only because that exact recipe
  trained on both seeds, not because padding is wrong.

## 7. For other tracks

The 14:45 cross-track warning should read: the 0.6B trainer had no padding bug. A
padded-vs-unpadded gradient test must compare each path with an FP32 reference; under
BF16 autocast two correct paths can differ by 5–60% (relative L2) on these RMSNorm/SwiGLU
backbones.

GPU-hours (node A GPU0–1): probe 0.31, pilots 0.45 (P1 0.07, P2 0.12, P4 0.17, P5 0.09).
