# 0.6B Milestone 3 amendment (M3c): matched control for the A6 Score effect

Frozen 2026-09-28 before any (a2c) run.

## Why

(a2) is the first 0.6B recipe whose typed-DEV Score is not one constant level (s1: 92/400,
levels 0 and 2; s2: 123/400, levels 0, 1, 2; every Milestone 1–2 model predicted one
level). But (a2) changed the schedule and the data together relative to Milestone 2, so
the Score gain cannot be attributed to A6 yet. Matrix v1.1 template S names the control:
`C(ρ) = A0s ∪ A0s-resample(ρ)` at the same ρ, same recipe. Milestone 3 has used about 2.5
of its 5.0 GPU-hours.

## Arm (a2c)

`m3-a2c-causal-s1|s2` = (a2) with the treatment slice replaced by a whole-group resample
of A0s stratified by source × task type × language (`mixture.resample_groups`, seed
`decision2-06b-m3-a6-rho\0A0s-resample`): 10,900 rows, 6,099,692 Qwen3 tokens (A6 mixture
6,101,929; −0.04%), 3,601 resampled rows in 2,927 groups, 682 updates. Resampled copies
get new ids and groups and keep their original row's Lux1 targets (`teacher_source_id`),
so the teacher covers every control row while A6 rows had none; that asymmetry is part of
the factor and is disclosed. Collapse stop: SELECT < 350 at the 3/8 milestone (update
256). Everything else equals (a2): model, seeds, schedule, padded micro-batches, loss,
checkpoints, readout.

## Contrast and rule

(a2) − (a2c) per seed on typed-DEV Score accuracy (primary), Score levels predicted,
Score slice of SELECT, `T_dev`, `H_pilot`, P and typed Choice/Noul. Paired bootstrap over
typed-DEV items (10,000 resamples, treatment vs control of the same seed label); the A6
effect counts only if the 95% lower bound is > 0 on both seeds and |Δ| exceeds twice the
measured seed noise σ_seed = |s1 − s2|/√2 (the larger of the two arms' values). Retention floors of matrix v1.1 §1.6 are reported. (a2c) is a control,
never a finalist.
