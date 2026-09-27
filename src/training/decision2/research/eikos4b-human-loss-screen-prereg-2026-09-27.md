# Eikos 4B human-source loss screen: prospective protocol

**Frozen before the treatment optimizer.** This is one bounded development
ablation, not a model release or JevArena FINAL run. The treatment tests whether
giving the existing independently sourced human TRAIN rows more loss mass
improves transfer while retaining ordinal Score and typed reasoning. It does
not add data or use exposed benchmark items as gradients.

## Existing control and the single intervention

The untouched control is the completed Eikos clean-v2 run: pinned upstream
`caiovicentino1/Eikos-4B@582ffb13f19a4da3f455e3db198584190bd7755b`,
TRAIN/SELECT/CAL file SHA-256s `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
`32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`,
and `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`.
The rights receipt is `61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8`.
Its 7,455 TRAIN rows yield 7,418 admitted rows after the fixed >100-option
quarantine, 4,402,743 tokenized TRAIN tokens, 232 optimizer steps, and a
released-native selected checkpoint 0232. The original control run, SELECT
receipts, and 4B package are preserved; **do not retrain or modify them**.

One new treatment uses the exact same source, TRAIN row order, input encoding,
batch schedule, seed `20260926`, one epoch, max length 8192, microbatch 2,
accumulation 16, BF16 backbone/FP32 rank-8 LoRA, LR 2e-5, AdamW, Brier
coefficient .25, save interval 32 and 232 fixed updates. The only optimizer
change is `eikos4b-human-source-weight-v1`: each of the 3,974 admitted human
source rows has weight 1.5; the remaining 3,444 rows, including all 516
Score rows, retain weight 1.0. The human sources are frozen in
[`loss_profile.py`](../training/eikos/loss_profile.py). Each optimizer window
normalizes the weighted per-row CE+Brier sum by its fixed sum of weights.
Consequently rows, options, text tokens, steps, batch composition, and source
initialization remain equal; the treatment has a different gradient objective.
No gradient is taken from SELECT, CAL, CSS pilot, typed DEV, public231, or any
FINAL panel. This is one combined human-source weight factor, not a sweep.

The existing control's original trainer/source/data/rights/loss/plan functions
were compared by parsed Python AST with the current local implementation:
all functions that load, encode, schedule, train and select are semantically
identical, apart from this proposed weight change. The old source trainer SHA
is `092cfc70b3531d11e0f2a1acee24d7d3da97fe613ad1e577080067a3e9a0274e`.
The old model `lora_metadata` helper changed, but its used
`select_target_modules` function did not. Both runs must use the same pinned
ROCm image digest `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`.
The old initial LoRA tensor was not archived; deterministic seed, identical
source and used functions plus pre-optimizer prediction parity are the
available initialization evidence, with that limitation disclosed.

## Mandatory pre-optimizer gates

1. Verify the four input/rights hashes, upstream release manifest, 7,418
   admitted IDs in the original order, quarantine, 4,402,743 tokenized TRAIN
   tokens, 232 planned windows, and the 3,974/516 human/Score counts. Refuse
   any data copy or source mismatch.
2. Use a previously idle GPU and an isolated, new output directory. Inspect
   task processes, containers, image digest, disk and HBM immediately before
   launch. No existing task output is overwritten.
3. At **zero optimizer updates**, run the source native letter readout on the
   unchanged 700 SELECT prompts. Compare only IDs, input hashes, candidate
   domains, selected keys and probabilities with the original source baseline
   prediction SHA `7fef6e2fcaa42e17ed1187533230b7210d566129ee60721e051f45f1e17ec7ff`.
   Require zero categorical mismatch and maximum probability difference
   `<=1e-6`. `gold_key` and `correct` are ignored in this parity test. A failed
   check **stops before any optimizer step**, and the failure receipt is kept.

## Frozen selection and observation sequence

Complete exactly 232/232 updates; checkpoints are at 32, 64, 96, 128, 160,
192, 224 and 232. The released native Eikos adapter evaluates the unchanged
SELECT700 for source and these eight checkpoints and selects family-macro
accuracy, then lower Brier, then earliest step. The control's authoritative
native selection is checkpoint0232: 595/700, family macro .827222, human
GoEmotions subtotal 331/400 and Score targeted-median 75/90. The treatment
must gain at least **six human correct of 400**, keep overall family macro
`>=.822222`, and keep Score `>=74/90`, with zero missing/invalid and no
new source-length truncations. These are conjunctive **development screen**
conditions; failure stops before CSS/typed/public evaluation. The extra
training compute relative to the reused control is disclosed.

Only if that screen passes, calibrate the selected treatment on unchanged
CAL700 using the same native type-temperature fitter, then seal gold-free
predictions under the exact same native adapter/runtime for treatment and
the preserved control. Evaluate CSS pilot1430 first and the separate
100-base-ID XNLI/PAWS-X validation diagnostic second; those source IDs are
absent from TRAIN. A transfer signal requires CSS pilot correct at least
**control +20/1430**, median task macro-F1 at least **control +.01**, no
individual task macro-F1 decline exceeding .02, and no loss of more than
two *base IDs* on either XNLI or PAWS-X averaged across their available
languages. Source-ID resampling, not translated prompt count, is the
uncertainty unit for XNLI/PAWS-X. The source-independent panel is exposed
development data and has no Score; do not call it blind transfer.

Only after those conditions pass may the candidate run typed DEV1600 and
public231 once as **regression diagnostics**, with no adjustment to this
intervention. Require typed DEV at least control minus eight of 1,600,
Score at least control minus four of 400, and public231 no lower than the
open Eikos source by more than two of 231 under an identical current runtime.
The public subset is exposed and **never selects checkpoints or weights**.
Chinese Score25 from the exposed multilingual panel is a descriptive error
audit only; its small sample cannot set a training or release threshold.

Do not open the sealed FINAL labels or authored release labels, upload a
private dataset revision, publish a model/collection item, or adjust a
threshold after seeing any treatment result. At most one treatment run is
authorized under this protocol. A favorable result only motivates a
separate independent panel and full JevArena preregistration; it is not a
Decision 2.0 4B release claim.
