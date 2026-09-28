# Own Kai 0.6B TRAIN-only teacher screen: negative feasibility result

**Decision: HOLD any own-Kai soft-target student arm on the current rights-clean
v2 data.** This is a TRAIN-only, same-source teacher diagnostic under the
signed [prospective 96-row protocol](small06-own-kai-teacher-train96-prereg-2026-09-28.md).
No student optimizer step, SELECT/CAL reading, typed DEV/FINAL, CSS pilot/FINAL,
JevArena or JevBench inference occurred. The current 0.6B private release
and all historical model scores are unchanged.

## Fixed execution and integrity

- Source, prior native conversion and quarantine match the preregistered
  SHA-256 values. The pinned own-Kai release is
  `llm-semantic-router/Decision-1.0-Kai-0.6B@7185f514f54b8f93c55998b1e8f9c5cc67f0d029`,
  with native manifest
  `c1bf07ab1c4c3fa1f819256d3de858d1ed87869bdfa663553280d7e78b88bee4`.
  The selected 96 independent TRAIN groups match roster SHA-256
  `4f56daee2b5f9f58451228114b6360ef62a8e503b3a61d361a9538d82dd4cebf`.
- The local signed implementation at `a9f0f2adb` and the exact mirrored probe
  have SHA-256
  `445a7261c36914df83bbbff5c080146f5ce0db47ab3a4dae576c7492a8bb3f29`.
  Local three focused tests, `make impact`, and `make check` passed. One first
  CPU inventory attempt caught an incorrect assumption that the Kai converter
  retained original row IDs; the builder deliberately assigns new IDs. That
  check was corrected locally before the preregistered teacher run, with the
  native manifest and unique native-row count still verified. A first
  read-only CPU container lacked a writable temporary directory; retry with a
  private temporary mount succeeded. Neither incident used a GPU or produced
  teacher predictions.
- Exactly one teacher GPU run used the pinned ROCm image digest
  `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`
  and qualified Kai/Lex runtime (Python `3.12.13`, torch
  `2.12.0+git6bbd260`, HIP `7.2.53211`, Transformers `4.57.6`, tokenizers
  `0.22.2`, safetensors `0.8.0`, NumPy `2.5.3`, `gfx942`). All **96/96** native
  responses were valid, finite, normalized and matched the original option
  keys; zero overflows, ties or schema exceptions. The container start-to-die
  duration was **27.73 seconds = 0.00770 GPU-hours**, under the fixed cap;
  the container was removed and its GPU is idle. The private aggregate receipt
  is mode `0600`, SHA-256
  `01f88a425e498133c8d93e5465d449865373311efd273d77d8ad6e189fa396b9`.
  No raw source text, labels, selected IDs or probability vectors were
  published.

## Frozen TRAIN evidence screen

Half-Brier is summed over options per row and averaged over the stratum. All
values below are **on the model's own TRAIN-source rows**, so they are neither
transfer accuracy nor new model scores.

| Slice | Correct / valid | Mean gold probability | Half-Brier |
| --- | ---: | ---: | ---: |
| Choice overall | **18/32**, gate ≥20 | .5084 | .2435, gate ≤.25 |
| Choice, stage-4 | 10/16 | .5179 | .2350 |
| Choice, other | 8/16 | .4989 | .2520 |
| Noul overall | **16/32**, gate ≥20 | .4563 | **.3857**, gate ≤.25 |
| Noul, stage-4 | 4/16 | .3088 | .5375 |
| Noul, other | 12/16 | .6039 | .2339 |
| Score overall | **8/32**, gate ≥12 | .2313 | **.3844**, gate ≤.25 |
| Score, stage-4 | 4/16 | .1959 | .4022 |
| Score, other three-level | 4/8, gate ≥3 | .3347 | .3320 |
| Score, other non-three-level | **0/8** | .1987 | .4013 |

The **technical gate passes** but the fixed Choice, Noul and Score evidence
thresholds fail, including probability quality in Noul and Score. The short
non-stage-4 Noul slice looks better, yet 16 selected TRAIN examples cannot
justify narrowing the mask after the result. The distinct prior AutoJev
Choice/Noul KL arm already excluded the weak stage-4 source and still failed
SELECT `510/700` versus the hard-label control's `562/700`. This result
therefore does not support another whole-TRAIN or opportunistically remasked
student distillation run.

## Capacity, source overlap and next action

Kai can process **6,262/7,455** original rights-clean v2 TRAIN rows, but only
**1,302,130/4,094,489 (31.8%)** Qwen-rendered TRAIN tokens; the excluded
1,193 whole-group long rows hold **68.2%** of token exposure. The eligible
teacher view has Choice 3,127/3,908, Noul 2,672/3,031 and Score 463/516
rows, including only 86 three-level Score rows. Thus the screen does not cover
most long composition evidence, and no truncation or row dropping is a valid
workaround. Kai1's older training source IDs are not fully published, with an
inherited 42-near-CSS warning; teacher source-disjointness from protected
transfer tasks is still unproven.

The next discriminating 0.6B work remains **source-quality admission** for
evidence-dependent Choice and independent three-level Score examples, with a
frozen, matched-token official-Qwen student control. Its CPU source, rights,
answer-blind evidence-necessity and overlap gates must pass before any new
training. Own-Kai distillation may be revisited only after a different,
prospectively defined short-source retention hypothesis and source-overlap
audit; this 96-row failure cannot be overturned by choosing another subset or
checkpoint after seeing these TRAIN diagnostics.
