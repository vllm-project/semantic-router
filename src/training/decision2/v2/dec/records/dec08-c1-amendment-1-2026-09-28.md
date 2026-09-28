# C1 amendment 1: adopt experiment matrix v1, add N0, revise preflight gates

Status: **committed before any C1 optimizer step.** Amends
[`dec08-c1-factor-screen-prereg-2026-09-28.md`](dec08-c1-factor-screen-prereg-2026-09-28.md)
(commit `1a5241981`). Work under C1 before this amendment: four zero-step
SELECT runs (A0, A1, A3, A4) and the Lux teacher-labeling job, all on code
`11f02b8d1`. From here every C1 job runs code commit
`cea7cd7ca0b3360f4f6fb70867fc6bfe6464d9e2` (exact mirror, content manifest
verified). Nothing else in the preregistration changes.

## 1. Adopt the research & data track's matrix v1 by hash

Matrix [`experiment-matrix-v1-2026-09-28.md`](https://github.com/vllm-project/semantic-router/blob/1ff80eaf3b5cecb4bad6a80ce1cb200347fa2e86/src/training/decision2/v2/data/records/experiment-matrix-v1-2026-09-28.md)
(commit `1ff80eaf3`) binds these shared-protocol items, now adopted by C1:

- **Checkpoints and selection:** 8 evenly spaced SELECT checkpoints (updates
  58, 116, 175, 233, 291, 350, 408, 466); BEST = SELECT700 family-macro
  accuracy, ties → earlier step (replaces every-32 and the Brier tie-break).
- **Statistics:** 10,000-draw paired group bootstrap. A factor effect counts only
  if its 95% lower bound is > 0 **and** |Δ| > 2σ_seed, in addition to the
  preregistered Δproxy ≥ +1.0 threshold for "confirmed".
- **Retention floors vs control:** no typed-DEV native type falls by more than
  3.0 points; `H_pilot` falls by at most 1.5 points; CAL Brier worsens by at most
  0.010; invalid/over-budget answers do not increase. A failed floor makes the
  factor negative.
- **H definition:** C1's primary proxy keeps the task-**median** CSS pilot
  macro-F1 (the v3 convention and the Eos 1.0 30.8286 baseline); the matrix's
  task-**mean** `H_pilot` and its proxy are reported beside it.

## 2. Add N0, the 0.8B seed-noise floor

**C1-N0** = C1-A0 with data-order seed `20260927` (everything else identical).
σ_seed for each dev proxy = |A0 − N0| / √2. Run concurrently with the arms.

## 3. Matrix IDs

| C1 arm | Matrix mapping |
| --- | --- |
| A0 | control (CE + Brier already = M4 treatment) |
| N0 | N0 (T1b) |
| A1 | new factor, proposed **M8 multitask loss balancing** (not in matrix v1) |
| A2 | new factor, proposed **M9 own-family larger-teacher soft targets**; distinct from D7-R2 (start's own distribution) and M6 (third-party) |
| A3 | M5-variant: head-only ordinal Score readout on A0. Matrix v1 places M5 on A0 ∪ A6; A3 is rerun on A0 ∪ A6 once A6 is frozen, whatever its A0 result |
| A4 | M1 readout, run on T1b at the user's request |

## 4. Preflight gates (schema `dec-arm-preflight/2`)

Measured before any gate was evaluated: the four concurrent zero-step SELECT
runs of the same Eos 1.0 model were bit-identical for A0/A3/A4, while A1 drifted
up to .027 with 5/700 argmax flips. The model is identical in all four, so this
is runtime noise (timing-dependent Triton autotuning under concurrent jobs), not
a code difference. The committed 1e-5 cross-process tolerance would fail on noise
alone; no `dec-arm-preflight/1` check was run. New gates, all required:

- in-process: source vs reloaded zero-step checkpoint exact (700/700, drift ≤ 1e-6);
  reloaded one-step checkpoint differs from source;
- cross-process (trainer vs preflight process): same argmax ≥ 693/700 and drift
  ≤ .05, for zero-step and one-step reload;
- reloaded adapter, head and residual tensors bit-equal to the saved files;
  LoRA B, head and any residual gate moved; finite loss and gradient.

All arms (including N0) rerun zero-step and one-step runs on the amended code.
The earlier zero-step runs are kept as the runtime-noise measurement.

## 5. A2 teacher fidelity (before A2's full run)

The Lux teacher path (shared renderer and head, `infer_1p0.py`) is run once on
typed DEV1600. It must reproduce the stored native Lux 1.0 DEV report
(1,388/1,600 = 86.75%) within ±1.0 point overall and ±2.0 points in each
family; otherwise A2 stops. This touches only the teacher, never a student.

## 6. Same-runtime Eos 1.0 control readout

Eos 1.0 is read out once on typed DEV1600 and CSS pilot1430 through
`infer_1p0.py` (temperature 1, as shipped) in this runtime. It is the paired
comparator for the formal rule; its T/H are reported beside the prior native
readout (T .495, H .192).
