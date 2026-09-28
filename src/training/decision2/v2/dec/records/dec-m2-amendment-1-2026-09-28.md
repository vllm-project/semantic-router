# Decoder Milestone 2 — amendment 1 (before any amended arm is launched)

Parent: [M2 preregistration](dec-m2-prereg-2026-09-28.md) (`143cfc214`). Adds
two arm families; nothing already launched changes. Written 2026-09-28 ≈17:45
UTC+8, after X4C-s1's development readout and the diagnostic below, before
any X4K / E4F-h / B4F-h job exists.

## Evidence that motivates the amendment

1. **A7g held-out diagnostic** (`sample_rows`, 1,120 rows = 80 per family ×
   type cell of A7g AHO, ≤ 3,072 tokens; `eval_rows`, node A): Nox 1.0
   972/1,120 (family macro .8594) vs E1-X2 973/1,120 (.8578); no cell moves by
   more than 4 of 80 (`stage4_scope` Noul 80/80 for both). X2 did **not** lose
   Nox's own Stage4 skills, so its post-key typed-FINAL loss (exception stack
   Noul −37) is a shift on a held-out family, which hard-label replay of Nox's
   training distribution (arm R) may not address.
2. **9B Milestone 2** (coordination note 16:45): own-model soft replay is the
   only arm that beats plain continuation; it acts as a trust region.
3. **0.8B mid-run:** after 254 of 2,030 updates the official Qwen3.5-0.8B-Base
   start matches the Eos 1.0 start on SELECT (502 vs 496 of 700), so a fresh
   start with the fixed data is a live option, and the coordinator's (d)
   question is worth a paired test at 4B.

## Added arms

| Arm | Tier / start | Mixture | Recipe | Cap (GPU-h) |
| --- | --- | --- | --- | ---: |
| X4K-s1, X4K-s2 | 4B Nox 1.0 `@cde2a68d` | control A0s-r (`156d8dab…`) | as X4C, but the soft targets are **own Nox 1.0** labels (package per-type temperatures, `teacher_label` on the A0s-r inputs; all 6,547 rows covered) at KL 0.5 instead of own Lux | 0.6 each |
| E4F-h-s1 | 4B Nox 1.0 | half mixture (`m2-half-a7-v1.json`: A0s-r whole + 50% whole-group sample of every other component of the full mixture) | E8F recipe: full fine-tuning, backbone lr 1e-5, head lr 1e-4, no teacher, 32,768 tokens / ≤ 64 rows per micro-batch, ≥ 64 rows per update | 5.0 |
| B4F-h-s1 | 4B `Qwen/Qwen3.5-4B-Base@1001bb4d` (fresh head) | half mixture | same as E4F-h | 5.0 |

E4F-h vs B4F-h replaces the preregistered single B4F arm on the full mixture:
at an equal GPU cost (two half-data runs ≈ one full-data run, ≈ 3.2 h each) it
answers the coordinator's question — does a fresh start with fixed data beat
Nox continuation? — with a same-data, same-recipe pair, and gives a
full-fine-tuning Nox continuation candidate. One seed each in Milestone 2; a
second seed precedes any promotion.

## Rules

Same preflight, selection, CAL, readout, screen and release reading as the
parent. X4K joins the 4B formal slots on the same terms as X4R (both seeds
pass the screen → both seeds formal). E4F-h / B4F-h: development readout only
in Milestone 2 unless one passes the 4B screen with P ≥ P(Nox 1.0 same
runtime) + 4 (outside the tie band), in which case its s1 takes one formal
run; release still needs a second seed. Budget after this amendment ≈ 20
GPU-hours (≈ 2 over the coordinator's estimate, reported).
