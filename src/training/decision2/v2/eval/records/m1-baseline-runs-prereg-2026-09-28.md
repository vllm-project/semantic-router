# Milestone 1 same-panel baseline runs — preregistration

Eval & peers track, 2026-09-28. Written and committed before any GPU run listed
here. Everything below is **post-key same-panel** evaluation of existing
packages; no model is trained or selected on these panels.

## Frozen evaluation plane

| Item | Frozen value |
| --- | --- |
| Panels | typed FINAL `e2a4a86b…` (1,600 items / 2,000 slots, gold `707dd28d…`); CSS15 `7a527357…` (6,547, gold `1cda9623…`); JevBench public 231 `642d3fac…` (targets `abc17b97…`, manifest `e0e7c677…`) |
| Scorers | typed `d02a3b2b…`, CSS `cfe199a1…`, public `aec840b6…`, paired `compare_v3` `bdb966ed…` (5,000 draws, seed 20260927) |
| Runner | `src/training/decision2/v2/eval/` at the commit containing this file |
| Runtime | node A image `decision20-train-fast:host2` `sha256:f83b1d10…`; Kai/Lex venv copy `/data/dev2/tools/envs/kai-lex` (library tree digest `ea64ae149066dc38`, identical to the qualified original) |
| Hardware | node A, one MI325X per process, GPU6 or GPU7 (`ROCR_VISIBLE_DEVICES` only), `--network none`, gold never mounted into inference |
| Denominators | missing, invalid and native over-budget answers count as failures; no truncation, candidate removal or chat conversion |

## Runs

| Run | Package | Native adapter | Purpose |
| --- | --- | --- | --- |
| R1 Lux1 repeat | `Decision-1.0-Lux-9B` current `main` `cdf4d3ef` weights (byte-identical to `bd45a30a`), native runtime of `bd45a30a` | `decision1-lux` (`inference.run`, over-budget invalid) | reproducibility of the sealed r4 comparator and current-package Lux1 comparator for the 9B track |
| R2 Kai1 repeat | `Decision-1.0-Kai-0.6B` main `9d6872cd` weights (= `7185f514`), native runtime of `7185f514` | `kai` (`inference.kai_lex`) | second reproducibility check on a different runtime family |
| R3 Lex | `Decision-1.0-Lex-0.6B` main `6c5e3d48` weights (= `ee8e74d9`), native runtime of `ee8e74d9` | `lex` | missing 0.6B own baseline |
| R4 Eos1 | `Decision-1.0-Eos-0.8B` main `363c4a5e` weights (= `3c2d6326`), native runtime of `3c2d6326` | `decision1-eos` | missing 0.8B own baseline |

New Decision Index peers are added by a separate amendment committed before
their launch, once the refreshed roster pins repository, revision, license,
native path and loaded parameters.

## Rules

- **Reproducibility (R1, R2):** PASS iff every answer category (Choice label,
  Noul side, Score argmax) is unchanged on all 8,378 prompts versus the sealed
  earlier predictions and T, H and public-231 correct counts are identical.
  The maximum absolute numeric drift is reported; drift above `1e-6` with zero
  category changes is disclosed but does not fail the check. A failed repeat is
  recorded and the earlier result is not reused for that model.
- **One shot:** each run is collected once. A technical failure (load, device,
  runtime qualification) is recorded with its GPU time and stops that run; no
  flag changes and in-place retries. A corrected run needs a new amendment.
- **Budget:** at most 0.5 GPU-hour per run including load; stop a run that
  exceeds it.
- **Outputs:** per run `COLLECT.json`, `GPU-TIME.json`, gold-free `SEAL.json`
  before scoring, then `REPORT.json`; paired v3 bootstrap versus the tier's own
  1.0 model (0.6B vs Kai1, 0.8B vs Eos1, 2B vs Sol1, 4B vs Nox1, 9B vs Lux1).

## Amendment A1 — Lux1 kernel-selection determinism (before D1/D2)

R2 (Kai1) passed: bit-identical to its sealed predictions. R1 (Lux1, node A)
**failed** the repeat rule against the node B r4 predictions: 7 typed and 36
CSS answer-category changes, maximum numeric drift 0.190, v3 65.808 versus
66.268, public 183 unchanged. Host kernel, amdgpu driver, image package trees
and package bytes are identical on both nodes; FLA's chunked gated-delta kernels
choose Triton configurations by runtime autotuning. Hypothesis: timing-based
autotune choices differ between runs/nodes and change reduction order.

- **D1** (node A, one GPU): Lux1 full three panels with
  `TRITON_CACHE_AUTOTUNING=1`, `TRITON_PRINT_AUTOTUNING=1` and a fresh
  persistent `TRITON_CACHE_DIR`. Output: predictions plus autotune cache C,
  frozen afterwards (tree SHA-256 recorded).
- **D2** (node A, the other GPU): identical run from a fresh copy of frozen C.
  PASS iff predictions are bit-identical to D1 (zero category changes, zero
  numeric drift) and no new `*.autotune.json` entries appear. PASS makes
  "image `f83b1d10…` + frozen cache C" the Lux1 frozen runtime and D1 the
  current-package Lux1 comparator; FAIL means autotune is not the only source
  and Lux1 is reported as the spread of R1/D1/D2 and r4 with no single number.
- Budget: at most 0.3 GPU-hour each; one shot each.

## Reused results (identity verified, no new GPU time)

Recorded predictions were located by SHA-256, adopted, sealed and re-scored with
the unchanged scorers; every recorded v3 score and public-231 count reproduced
exactly: Kai1 35.938/114, Bosun v3.1 38.524/133, DEV2.0-0.6B 38.520/143,
Sol1 45.580/161, Decider 2B 49.499/175, Nox1 56.470/173, Decider 4B 61.882/192,
JPT-9B 60.994/197, Lux1 r4 66.268/183, AutoJev 27B 72.310/200.
