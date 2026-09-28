# 9B Milestone 2: Lux 1.0 continuation result (development readout)

**Disposition: own-Lux soft replay (L2) is the only arm that improves on the
continuation control, and it ties Lux 1.0 under the calibrated proxy; no arm
reaches the 72.33 scalar gate. Formal post-key run held for a coordinator
decision.** Protocol: [preregistration](lux9b-m2-continuation-prereg-2026-09-28.md)
(with its preflight r1 amendment). All numbers are typed DEV 1,600 / CSS pilot
1,430 development readouts from the same-runtime native adapter, never release
scores.

## Arms, primary seed (paired 10,000-draw bootstrap)

| Arm | BEST | T | H | Proxy | Choice / Noul / Score | Δ proxy vs L0 [95%] |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Lux 1.0 zero-step (pinned cache) | — | .8763 | .5724 | 70.82 | 799 / 272 / 331 | — |
| L0 control | 466 | .8663 | .5598 | 69.64 | 795 / 272 / 319 | — |
| L1 zero-gated ordinal Score readout | 408 | .8706 | .5677 | 70.30 | 797 / 277 / 319 | +0.67 [−0.70, +2.02] |
| L2 own-Lux soft replay (KL 0.5) | 466 | .8794 | .5898 | 72.02 | 797 / 273 / 337 | **+2.38 [+0.63, +4.05]** |
| L3 A6g Score data (25% substitution) | 291 | .8700 | .5289 | 67.84 | 800 / 290 / 302 | −1.80 [−3.88, +0.20] |
| L4 A6h human Score (25% substitution) | 350 | .8694 | .5484 | 69.05 | 793 / 282 / 316 | −0.59 [−2.44, +1.32] |

## Seeds (control versus L2, paired by seed)

| Seed | L0 proxy | L2 proxy | Δ [95%] | Score L0 → L2 | Noul L0 → L2 | CSS H L0 → L2 |
| --- | ---: | ---: | --- | --- | --- | --- |
| 20260926 | 69.64 | 72.02 | +2.38 [+0.63, +4.05] | 319 → 337 | 272 → 273 | .560 → .590 |
| 1 | 71.11 | 70.82 | −0.30 [−1.99, +1.43] | 314 → 326 | 282 → 266 | .581 → .577 |
| 2 | 69.81 | 71.42 | +1.61 [−0.12, +3.36] | 329 → 338 | 265 → 272 | .561 → .580 |
| **Mean** | **70.19** | **71.42** | +1.23 | 320.7 → 333.7 | 273.0 → 270.3 | .567 → .582 |

- **Score:** L2 beats its paired control in every seed (+18, +12, +9). Against
  Lux 1.0 (331) it is +6, −5, +7: not a clear gain over the 1.0 model.
- **Noul:** seed 1 drops 16 items against its control (−4.0 points, beyond the
  −3.0 floor); the other seeds are flat. Against Lux 1.0, Noul is 273/266/272
  versus 272.
- **CSS pilot:** L2's H exceeds Lux 1.0 in every seed (.590/.577/.580 vs .572);
  the mean-H proxy interval against Lux 1.0 is above zero for all three seeds.
- **Calibrated proxy** (eval track: v3 ≈ 19.12 + 0.629·P, |ΔP| < 4 is a tie):
  L2 seed mean 71.42 predicts v3 ≈ 64.0, Lux 1.0 70.82 predicts ≈ 63.7 (its
  measured node-A v3 is 65.808). L2 and Lux 1.0 are a tie; the rule would send
  L2 to the formal runner rather than decide on P.

## Mechanism notes

- Continuing Lux on current A0 alone (L0) loses a little (T interval versus
  Lux 1.0 below zero; Score −11 in the primary seed). Soft replay acts as a
  trust region: it keeps Lux's behavior while fitting A0 and slightly
  improves CSS and Score.
- The zero-gated ordinal readout never opened (gate ≈ −0.001 for all 466
  updates) because Lux already fits A0's Score rows (Score CE ≈ 1e-4), so L1's
  Score answers equal L0's. The readout needs Score data Lux does not already
  fit to engage.
- Both Score data arms (A6g generated levels 2–10, A6h human ordinal) lowered
  typed-DEV three-level Score and CSS H at this substitution dose; A6g raised
  Noul (+18). Neither is a Score lever for Lux on the current panels.
- Hard-negative objective: not run (joint CE already normalizes over own
  distractors; the A4 data arm is still in repair).

## Failures and technical records

- L1 preflight r1 failed the cross-process zero-step drift gate (0.0528 >
  0.05) under concurrent Triton autotuning; after pinning one persisted
  autotune cache (seeded from the eval track's frozen node-A Lux1 cache) every
  preflight passed (L0, L1 r2, L2, L3, L4).
- Runtime noise: uncached versus cached Lux 1.0 readouts differ on 3 typed and
  13 CSS answers; all arms were paired against the cached readout.
- No ROCm backward fault in 9 full 466-update runs (FLA path, gradient
  checkpointing, peak 36 GiB, 4.5–5 s per update).

## Teacher cross-check

Decoder-track Lux labels `752b7c8f…` (7,455 rows) versus the Milestone 1
frozen-path labels `abaa1113…` on the 7,324-row overlap: argmax agreement
Choice 3,811/3,824, Noul 2,991/2,993, Score 503/507; mean max |Δp| 0.0011–0.0032,
largest 0.071; all input hashes equal. M2 used the decoder-track file.

## Resources and digests

Milestone 2: **7.40 GPU-hours** on node A GPU2–4 (training 6.15, CAL and dev
readouts 0.68, preflights 0.47, Lux 1.0 readouts 0.10). Code `00b105bdf`
(trainer and readout unchanged from the decoder track at the integration
merge); mixes `6e674b32…` (A6g) and `a4db1f5b…` (A6h); final readout report
`a57bbac4…`. Triton cache seed tree `e215f8bd…`.

## Proposal

Run L2 (primary seed 20260926, the preregistered primary) through the eval
track's frozen formal runner on node A against Lux1 (65.808 / 183), about 0.2
GPU-hours, because the calibrated proxy calls it a tie. Report it as a
Lux-continuation candidate only if its post-key v3 paired interval against
Lux1 is above zero; otherwise the 9B tier stays on Lux 1.0 until admitted data
arms that Lux does not already fit are available.
