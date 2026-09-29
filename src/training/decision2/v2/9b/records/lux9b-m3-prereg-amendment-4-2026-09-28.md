# 9B Milestone 3 preregistration, amendment 4: arm DW (weight interpolation of the D soup with Lux 1.0)

Frozen 2026-09-29 (UTC+8 early morning) after arm D's development readout and before any DW
checkpoint exists. Everything not named here is as in the preregistration and amendments 1–3.

## What arm D showed (development readout, already recorded)

Seeds P 66.69 / 65.02 / 59.02 (typed-DEV transition table 394 / 280 / 210 of 400); the uniform
soup recovers to P 69.35 (≥ the seed mean, so it is D's artifact) with typed-DEV Score 364 vs
331 but Choice 759 vs 799 (transition table 359) and CSS pilot H .5516 vs .5724: a proxy tie
that breaches the Choice (−5.0 points) and CSS-H (−0.021) floors, so arm D is not a finalist.
Full fine-tuning learned new Score behavior but drifted from Lux on other skills; the own-Lux KL
on recipe rows did not hold it.

## Arm DW

- **Artifact:** uniform FP32 average (`v2.dec.soup`) of the D soup (`9a1d7db8…`) and Lux 1.0 in
  the same full-checkpoint format — the zero-step checkpoint of arm D's preflight
  (`m3/pf-D-s1-zero/run/checkpoint-0000000`, whose in-process and cross-process reload equals
  the untouched Lux 1.0 source) — i.e. θ = ½·θ_D-soup + ½·θ_Lux (WiSE-FT with α = ½; Wortsman et
  al., 2022). **α = ½ is fixed here; no other α is built or read.** Equivalent to (1/6)(D1 + D2 +
  D3) + ½·Lux.
- CAL698 per-type temperatures (`v2.dec.calibrate_ckpt`), typed DEV + CSS pilot readouts, the
  preregistered finalist rule (P tie or better, typed-DEV type floors 3.0 points, CSS pilot H
  floor 0.015 vs Lux 1.0), then — only if it qualifies — the formal protocol and gate of the
  preregistration (16K, native Lux1 16K comparator), packaged `qwen-full`.
- **Why it is not outcome fishing:** a single, standard, pre-specified post-hoc trust region
  (weight-space interpolation toward the start), one artifact, one readout, the unchanged
  finalist rule; it is disclosed as chosen after arm D's development readout.
- **Budget:** ≈ 0.3 GPU-h (+ ≈ 0.3 if formal); Milestone 3 cap unchanged (38 GPU-h).
