# 27B Score error audit on frozen typed DEV

Status: **method frozen before this aggregate diagnostic was run**. This
diagnostic reuses previously scored development predictions; it is not a new
model evaluation or a release score. No model inference, training, calibration,
candidate selection, or sealed-label access is involved.

## Inputs and fixed analysis

- Typed DEV gold SHA-256:
  `c7a8b86bda0d0d6120e572b94dfc756bf10264108554af76307141ae02fbf5dc`.
- Qwen3.8-27B clean-v2 BEST368 original CAL700 predictions SHA-256:
  `15ebc30b0207208e971e71deff2fceb4538fc74caf389ba47a2db9fdc9c622b1`.
- Decision 1.0 Lux 9B native predictions SHA-256:
  `41fb3655ae2e5e4f97d8d0bdc3c9492985dc909fe1e5caacf6c20e04945b335a`.
- Aggregate-only diagnostic script SHA-256:
  `c14e3b218fe0e4b8a86f8086d735ec638e852d61f48ee0f82157fc130afd9d0e`.

For each model, reuse `benchmark.score.evaluate_answer` and count invalid or
missing answers as wrong. Tabulate each family, true Score level to predicted
modal level, all-four-correct groups, and simultaneous correctness on
counterfactual, order and label pairs. Show row-matched win/loss between the
two models by family. The diagnostic writes only aggregates and hashes, not
raw prompts or private gold, and must reproduce the existing 27B overall
1,213/1,600 result. If the old Lux predictions are not byte-identical to
the above pinned file, stop rather than substituting another calibration.

Interpretation is limited to this synthetic DEV generator. It may motivate a
TRAIN-only ablation but cannot establish transfer, JevArena rank, or a 27B
release. Do not use sealed FINAL or 15-task heldout transfer labels to diagnose
or select a candidate.

## Private dry-run output correction before public reporting

The initial private execution produced report SHA-256
`1ab1d053db9e31c7b1a3c1ed7e67661f499269b16d7094198153eb9881d15606`.
Its `gold_level` histogram unintentionally split Choice by each synthetic
symbol, yielding a large per-label table. The file remains private and was
not published or used for selection. The analysis code now excludes Choice
from `gold_level` while retaining all 1,600 Choice rows in family and paired
totals; Noul and Score gold-level summaries remain. Revised source SHA-256
is `46ddfe589a3f7e19bb1952b14886684a514a1b07e527d9e2b03c8695e34ab936`.
The same three frozen inputs will be rerun once after this correction is
committed. This is an output-privacy fix, not an opportunity to tune a model.
