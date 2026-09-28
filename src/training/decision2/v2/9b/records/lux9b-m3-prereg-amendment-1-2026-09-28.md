# 9B Milestone 3 preregistration, amendment 1: arm B inputs and the same-renderer control

Frozen 2026-09-28 before any arm B step and before any arm A readout. Everything not named here
is as in the [preregistration](lux9b-m3-prereg-2026-09-28.md).

## Arm B teacher files (published by the research & data track)

- `m3/teachers/autojev27/pk1/A0s-train.targets.jsonl` `97a071af…` (private revision
  `9bb9790b014d17d7dad50677b4f057f20e50460c`, 7,299 rows, from the pk1 A0s prompts) and
  `m3/teachers/autojev27/rp-v2/aj-m.targets.jsonl` `ba52dd86…` (revision
  `3a99bf1c26ac7924ec1216cdf1f703e1a5f21ae6`, 84,523 rows; 41,388 are full-M rows). Teacher
  `denis-pplx/autojev-27b@6f5b557e`, qualified runtime (qualification v2 PASS, node A).
- Spec `lux9b/specs/m3-b-full-M-autojev.json` differs from arm A's only in these two files, so
  the materialized TRAIN must be byte-identical to arm A's (`55abf2ad…`); the build is refused
  otherwise.
- On the 41,388 non-A0s full-M rows, argmax-vs-gold for AutoJev / own Lux: Choice .830 / .778,
  Noul .825 / .783, Score .606 / .551; the two teachers pick the same option on 81% / 89% / 66%
  of rows; normalized entropy is similar (Choice .41 / .34, Noul .45 / .47, Score .62 / .60).
- Provenance caveat (disclosed on every record and any card of an AutoJev-distilled candidate):
  AutoJev's public training pipeline includes SFT rows generated with a closed model.

## Same-renderer Lux1 16K control

Candidates are read through the shared `from_decision1` renderer (`v2.dec.infer_dec`); the
release comparator (65.808) used Lux's native runtime. The decoder track measured a renderer
cost of −0.78 v3 for Nox 1.0 at 8K. A **descriptive** control is therefore collected once:
untouched Lux 1.0 through the shared renderer (`v2.dec.infer_1p0`, package temperature) at
16,384 tokens, on node A with the frozen Milestone 3 autotune cache copy, before any
candidate's formal run (so candidates reuse its kernel choices). Candidates are compared with
both; **the release gate stays the native 16K comparator (65.808).**

## Measured cost

Arm A runs at ≈ 6,400 native tokens/s per GPU (≈ 1.5 s per 16-row update, 3,131 updates,
46 GiB peak): ≈ 1.4 GPU-h per seed including SELECT, CAL and readouts. Budget unchanged.
