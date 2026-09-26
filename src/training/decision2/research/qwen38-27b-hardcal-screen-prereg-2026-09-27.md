# Qwen3.8 27B independent hard-CAL transport screen

Status: **preregistered before this model was inferred on hard CAL**.
The clean-v2 BEST368 head has a severe calibration shift on typed DEV Score:
its existing CAL700 has just 90 quantized-median Score rows. This bounded
screen tests a separate, already frozen hard-CAL source with genuinely
different Noul and Score examples. It changes no weights, checkpoint,
selection set or existing original-CAL result.

## Frozen source, data and separation

- Candidate: pinned Qwen3.8-27B posttrained source
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, clean-v2 completed run
  `BEST.json` checkpoint-0000368, model fingerprint
  `d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2`.
- Original CAL700 fit SHA-256
  `e4e0d9fda575d807503299bdf7c67828a7c4d3c3b09e9fa1bf3750a51b79db78`;
  original temperatures Choice .9814940779855923, Noul 1.0952572366387938,
  Score .5646764464932256. All frozen original DEV prediction bytes remain
  unchanged.
- Hard CAL v2 JSONL SHA-256
  `bf5bbf29693928a2559ce0aff10e9d6b5b1541b50698634fcdb7725902412dcf`,
  manifest SHA-256
  `063c3fd1f2ddb0e2f08823e77961e5b356e18c9752c24b3ff84d3150270d3e0c`.
  Select only the 300 `cal_hard_archive_approval` Noul and 300
  `cal_hard_net_credit_grade` Score rows. The other 300 hard-CAL rows are
  retained CSS-pilot Choice examples, so they are **excluded** to prevent a
  CSS-pilot calibration/evaluation overlap. The hard-CAL manifest reports
  zero exact and screened near overlaps for the fresh Noul/Score rows against
  visible DEV and CSS pilot; approximate near detection is not a proof of no
  semantic overlap.

## Fixed method and decision gate

First validate the above bytes, BEST checkpoint, source and model fingerprint.
The same native encoder collects all 600 logits at the completed run's
4,096-token limit, batch size one, BF16 backbone/FP32 head on one AMD GPU;
any dropped, overlength or nonfinite row aborts the screen. Fit two positive
temperatures by the existing `fit_temperature` NLL objective on these 600
CAL labels, never on DEV or CSS labels. Freeze the fit and its output SHA
before looking at independent DEV scores. Keep Choice at the original CAL700
temperature. Do not alter the original run provenance or impersonate the
run's audited CAL700; this is an explicitly separate diagnostic fit.

Apply the new Noul and Score temperatures to **copies** of the existing
calibrated DEV predictions by probability power transform
`p_new ∝ p_old ** (T_old / T_new)`. Recompute Score's probability-weighted
mean exactly as the native adapter does. Choice stays byte-identical. Reuse
the frozen typed scorer on all 1,600 DEV items and report accuracy, Brier,
NLL and ECE by type against original CAL700. CSS pilot predictions need no
rescore because they are Choice only and Choice temperature is unchanged.

This calibration hypothesis is promising only if independent DEV Score Brier
improves by at least .02 and Score NLL by at least .10, while Score hard
correct decreases by no more than five of 400 and Noul Brier regresses by
no more than .005. Report every metric regardless of gate. A positive screen
would require a separate native inference/package parity test before any
release; it cannot repair the underlying 161/400 Score reasoning accuracy or
prove six-axis JevArena quality. Synthetic FINAL and 15-task CSS heldout
labels remain unopened.

## Pre-inference implementation freeze

The original calibrated DEV prediction file SHA-256 is
`15ebc30b0207208e971e71deff2fceb4538fc74caf389ba47a2db9fdc9c622b1`.
The isolated hard-CAL collector/transformer source SHA-256 is
`722fa198babbf9d6f09915ff710cf7b649d73c08722b4cccbbb4739d26ffbff8`.
It verifies the completed run's original CAL identity, model fingerprint,
frozen hard-CAL bytes, 300/300 eligible source families and full row
coverage. Its diagnostic output cannot be passed off as the original native
calibration receipt.
