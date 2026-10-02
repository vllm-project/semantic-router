# Decoder M17 stage 2 — preregistration (4B toward the class leader; 2026-10-02)

Written 2026-10-02 ≈05:55Z (13:55 UTC+8), before any stage-2 file was built and before any stage-2 GPU job, on the
coordinator's 13:55 note. `4b-LHS17SD` (M17 stage 1) is the current Nox-4B release (`b285e7a1`); M17 is now the 4B
trainer **and** the single Nox-4B publisher. Stage 1: prereg `da770d98a`, amendment 1 `367cdfa67`, data lock
`4f68f1eb0`, hand-over [`dec-m17-handover-2026-10-02.md`](dec-m17-handover-2026-10-02.md).

## Question

The 4B class leader still leads on RAGTruth, PhishNChips, VAST, iSarcasmEval, GSM8K, GPQA, ACOS and ANLI (private gap
analysis). Stage 1 showed more swapped IB helped (S17 above S10). Stage 2 asks which of four cheap moves raises the
Index further: new IB families aimed at the deficits, typed upweighting, a higher swap dose, and interpolations.

## Arms (two seeds each, uniform FP32 soup; every arm is stage 1's recipe with one stated change)

Recipe (stage 1, unchanged): LoRA r128 / α256 / dropout .05 from Qwen3.5-4B-Base@`1001bb4d`, fresh head (seed
20261001), LoRA / head LR 1e-4, KL 1.0 to the released LH's self-distillation targets on the kept released rows,
`--teacher-partial` (IB rows gold only), seeds 20260926 / 20260927; token batching as stage 1.

| Arm | Change vs `4b-LHS17SD` | Wave / node F GPUs |
| --- | --- | --- |
| **(b) `4b-LHS17UP`** | the same TRAIN and teacher (hard links of the locked S17 files) plus `--example-weights`: every kept released row ×1.5, every IB row ×1 (M14's 4B weights; M14's recipe won at 2B). `ops/m17/m17_weights.py`. | 1 / GPU2, 3 |
| **(c) `4b-LHS23SD`** | swap share .23 instead of .17 (`m17_data.py`, same inputs, flags and seed). The IB1-r3 + IB2 pool without `sentfin` is .232 of T, so .23 is the highest dose the pool supports (S25 is not buildable). | 1 / GPU6, 7 |
| **(a) `4b-LHS17IB4`** | S17's TRAIN plus, appended (above T), every TRAIN row of IB4 phase 1 (`llm-semantic-router/decision-2.0-training-data` `m6/ib4/p1` @ `76cea510`: `sqa2`, `isarc2` (kept as its own family; in-distribution), `sentfin3`, `fc_pick`) and the IB3-r2 maths family. IB1 `sentfin` is already dropped in S17. New rows gold only. Files and hashes in a data-lock amendment before training. | 2 / the first free pair |
| **(d) interpolations** | uniform FP32 averages of finished stage-2 soups and `4b-LHS17SD`, at most three points, chosen only from arms that finished training (no Index value is used to pick them; the choice rule is: the pairs of the two newest finished soups with `4b-LHS17SD`, weight .5). CPU only. | after waves |

If wave 1 finishes before arm (a)'s data is locked, the next wave may instead train `4b-LHS23UP` (S23 TRAIN + UP
weights), preregistered here.

## Measurement and release (Index-first rule, 09:55 / 11:35 / 12:30 notes)

- **Each candidate is measured once, on its BF16 release copy** (`v2.release.bf16_copy`, restaged onto
  `DEV2.0-4B-13d42143`), with the IX1 harness on the eval fast lane (node E GPU0–3 / 6–7, lease `track=eval-fast`,
  ≤ 2 h per lease; never node E GPU4–5) or free node C / D GPUs, sharded wide. No development readouts, MLX-DEV2 or
  formal v3 are run for release (references only; not run).
- **Release gate:** paired-bootstrap Index 95% lower bound > 0 vs **the current release's** Index run (base: the
  `DEV2.0-4B-LHS17SD-bf16` run), plus the integrity checks: IX1 parity, Hub smoke, contamination audit of the new
  TRAIN files (IX1 audit method, planted controls), and typed-FINAL with no collapsed type (formal item 3).
- **IB4 arms** also need the C1 recheck r3 PASS (`v2/eval/records/c1-recheck-r3-2026-10-02.md`) before release.
- Among candidates ready at the same time, the largest lower bound is released (5e7b8132's release ops,
  `release/records/dev2-4b-indexfirst-2026-10-02/`, `dec/ops/4bif/`); a later candidate must beat the then-current
  release. Index values stay private (`decision2-program/private/m17/`, node private stores).

## Budget and stops

≤ 50 GPU-h for stage 2. Per-arm cap 5.0 GPU-h of training; the chain gate on node F is 50 GPU-h of M17 receipts
(stage 1 included). A failed preflight stops its arm (no rerun); a failed seed is not replaced. Polls ≤ 30 min with
state commits in [`dec-m17-state.md`](dec-m17-state.md); hand-off near 5 h.
