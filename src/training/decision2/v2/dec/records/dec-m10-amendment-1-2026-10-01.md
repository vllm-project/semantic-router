# Decoder Milestone 10 — amendment 1 (LT stopped by its seed-1 zero-step; LT2 / NT2 = the same arms at 8,448 tokens)

Written 2026-10-01 ≈12:05 UTC+8 (04:05Z), **before any M10 readout**: the only M10 GPU jobs so far are preflight
stages (LT-s1's zero-step; LH / FB seeds' zero-step, one-step and gate jobs on node F). Preregistration
[`dec-m10-prereg-2026-10-01.md`](dec-m10-prereg-2026-10-01.md) (`2dea44d6f`); data lock `6a8092977`.

## What happened

- `m10-LT-s1` (node E GPU0, 03:45:53Z) stopped in its zero-step run (`zero-step FAILED`, 27 s, before any model
  output): the label-token prompt of TRAIN row `a7:stage3:stage3_replay_high_k-000656` is 8,252 tokens, over the
  preregistered `--max-length 8192`, and the encoder never truncates.
- Measured on node E with the base tokenizer, over the locked TRAIN file: the head prompt's longest row is 7,754
  tokens (29,404,539 tokens in total, M7 / M9's count); the label-token prompt's longest is **8,272** and **9 of
  58,739 rows exceed 8,192** (30,388,389 tokens in total, +3.3%: one label and one separator token per option, and
  the longer answer cue). SELECT700 / CAL698 are at most 240 tokens under either prompt.
- The cause is the preregistered prompt's length on many-option rows, not data, randomness or the runtime. The rule is
  not reinterpreted: **LT stops** (recorded `FAILED`, preflight); LT-s2 / s3 are not started.

## Amendment

- **New arm LT2** = LT exactly as preregistered (start, LoRA, LR, seeds, readout, data, targets, batching, gates)
  with `--max-length 8448` (≥ 8,272, a multiple of 128) — the only change, so the same 58,739 rows train under
  either readout. **NT2** = NT with the same change (NT starts from the same TRAIN file).
- LT2 replaces LT in every contrast, gate and recommendation of the preregistration (readout contrast LT2 − LH). Its
  extra tokens (+3.3%) are the readout's own prompt cost and are disclosed with the contrast.
- Placement: node E GPU0–2 (`m10-chains.sh` wave `w2`: "LT2:i NT2:i"); NT2 starts only if LT2's three preflights pass
  and node E has used < 40 GPU-h, as preregistered for NT. Caps unchanged (LT2 6 GPU-h, NT2 6 GPU-h); LT's 0.01 GPU-h
  counts toward the total.
- The head arms (LH, FB) are unaffected: their longest row is 7,754 tokens.
