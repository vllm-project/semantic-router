# 9B Milestone 3: v2 data at scale with soft replay (preregistration)

Status: frozen 2026-09-28 before any Milestone 3 training step. Goal: a 9B candidate that
clearly beats Lux1 — post-key same-panel v3 paired 95% interval lower bound > 0 against the
node-A Lux1 comparator at the same 16,384-token input limit (65.808, eval run
`m1/d1-lux1-autotune-cache`). Development readouts are never release scores; v3 is post-key.

Handoff evidence (Milestones 1–2): the CLM disaggregated readout and frozen-backbone heads
are closed; Lux continuation on A0 with own-Lux soft replay (L2) tied Lux1 post-key
(+0.130 [−2.357, +1.480] at 8K) with typed Choice −19 / Noul −9 and most of its transfer
gain from FLUTE. Soft replay stays the trust region; the lever is data Lux does not fit.

## Fixed configuration (every arm)

| Field | Value |
| --- | --- |
| Start | own `llm-semantic-router/Decision-1.0-Lux-9B@bd45a30a…` (bundle `985ade73…`), 7,940,895,744 deployed parameters |
| Trainer | decoder track `v2/dec/train_dec.py`, unchanged; LoRA r16 / α32 / dropout .05 on every Qwen3.5 projection plus the shared candidate head |
| Objective | CE + 0.5·Brier over offered options, + 0.5·KL(teacher ‖ student) on every recipe row (`--teacher-partial`: replay rows have no teacher term) |
| Optimizer | LoRA lr 5e-5, head lr 2.5e-5, weight decay .01, warmup .05, one epoch (as Milestone 2) |
| Batching | token-budget micro-batches (≤ 16,384 padded tokens, ≤ 16 rows), ≥ 16 rows per update — the decoder track's validated padded path, keeping Milestone 2's 16 rows per update and learning rates; max length 8,192 (longest TRAIN row 7,751 tokens), no truncation |
| Selection / calibration | SELECT700 `32a4352d…` at eight evenly spaced checkpoints, family-macro accuracy with earliest tie (`matrix-v1`); **CAL698** `19cc1a8c…` per-type temperatures for BEST |
| Runtime | pinned image `sha256:f83b1d10…` with FLA 0.5.2; one persisted Triton autotune cache (Milestone 3 copy of the Milestone 2 cache, itself seeded from the eval track's frozen node-A Lux1 cache); node A GPU6–7, and GPU2–4 when the research & data track returns them |
| Seeds | 20260926 (primary) and 1; arm decisions use the two-seed mean |

## Data (materialized by `lux9b.m3_data`, spec `lux9b/specs/m3-a-full-M.json`)

- **Recipe mx-v2-full-M** (ids `5b8d5d29…`): 48,687 rows, 19,686,750 native tokens (Choice
  23% / Noul 51% / Score 26% of tokens; 21 languages). Rows joined by id from the m2 arms
  (revision `5c602c5a…`, hashes equal to the `ed87a03a…` registry), V1S from the v1 A6g/A6h
  files (`5c0255ed…`), **A0s from pk1** `35c1f98f…` (`d8eae3e4…`).
- **Own-Lux targets on every recipe row**: pk1 canonical A0 `56627939…` (7,299 A0s rows) and
  RP-v2 waves 1 `47049a6b…` (12,695), 2 `2c6ab38d…` (28,647), 4 `084adacd…` (46); wave 3 is
  not needed for M. Input hash and option keys checked per row.
- **A7 retention replay**: view `dec10-natural24k` (Lux's own final-stage pool) at A7 v3
  (`3a4efc77…`, sub-arm files hash-verified), families `natural_cosmos_qa` and
  `natural_squad2_answerability` excluded (5,056 rows), rows repeating pk1 A0 or the recipe
  by current or pre-renumbering input hash dropped (2,817), then whole groups stratified by
  source × task type × language to 5.0M tokens: **4,711 rows, 5,010,720 tokens** (Choice
  3,295 / Noul 1,004 / Score 412; en 3,135 / zh 1,576; Stage4 generated 4,154, SNLI 557).
  Gold labels, no teacher term. This protects Choice (the recipe is Noul-heavy) and the
  Stage4 skills behind Lux's typed reasoning.
- **Guards**: no source matches a sealed C1 candidate (38 name keys from the C1 registry
  `db00d4fe…`) or MASSIVE / PAWS-X / XNLI; isolation against SELECT700 and CAL698 passes.
- TRAIN: **53,398 rows, 24.70M tokens** (Choice 15,143 / Noul 26,007 / Score 12,248 rows).
  Expected outputs: train `55abf2ad…`, teacher `f986ed39…` (a scratch build of this exact
  code produced them; the recorded build must reproduce them byte for byte).
- **Deviation from the handoff pin (decision within scope):** A7 at v3 `3a4efc77…` instead of
  v2 `39a120ca…`. v3 is v2 minus 323 rows the A7 embedding/lexical rescreen quarantined
  (A7h rows all inside the two excluded families; A7m 12, A7i 12, one of them near an
  `mlx-diag` item), and the A7 track designates v3 for release candidates. A7g/A7p/A7o are
  byte-identical. The recipe keeps A0s whole as frozen by the research & data track
  (including its legacy Cosmos QA / SQuAD 2.0 rows, which Lux was trained on).

## Arms

| Arm | Factor | TRAIN | Teacher term |
| --- | --- | --- | --- |
| **A** | Lux + v2-M + own-Lux KL + A7 retention replay | `a-full-M` (above) | own Lux on all recipe rows |
| **B** | as A with **AutoJev-27B** targets instead of Lux | same TRAIN rows | AutoJev (`m3/teachers/autojev27/`: AJ-M + pk1 AJ-A0s) on all recipe rows; built as soon as those waves are published |
| **C** | v2-L pass for the better of A / B | mx-v2-full-L + A7 natural24k replay at 25% of recipe tokens | as the chosen arm (waves 1–4 incl. wave 3, or AutoJev AJ-M + AJ-SL + AJ-A0s) |

- A is B's matched Lux-target control (same TRAIN rows, seeds, budget, runtime).
- B discloses AutoJev's provenance caveat (its public pipeline includes SFT rows generated by
  a closed model) on every record and on any card of an AutoJev-distilled candidate.
- C runs only if A or B passes the development screen below; its budget scales with tokens.

## Readout, comparison and promotion

- Per run: CAL698 temperatures for BEST; gold-free typed DEV 1,600 + CSS pilot 1,430
  predictions (`v2.dec.infer_dec`), scored by `v2/dec/dev_readout.py` (unchanged scorers'
  answer functions; 10,000-draw paired bootstrap), paired against the same-runtime Lux 1.0
  zero-step readout re-collected under this milestone's mirror.
- Calibrated proxy P = 100·√(T_dev·H_pilot) (eval track; v3 ≈ 19.12 + 0.629·P). Tie rule:
  |ΔP| < 4 is a tie. **Finalist rule:** an arm goes to the formal runner if its two-seed mean
  P is not below Lux 1.0 by 4 or more (a tie or better) and, on the two-seed mean, it keeps
  typed-DEV Choice, Noul and Score each within 3.0 points of Lux 1.0 and CSS pilot H within
  0.015. Between arms the same tie rule applies; tied finalists all go formal (at most three
  formal candidates in this milestone). The formal checkpoint is the preregistered primary
  seed's BEST (no best-seed selection).
- **Formal run** (eval track's frozen runner on node A, persisted autotune cache): the
  candidate packaged at **16,384 tokens** (`v2/dec/adapter-spec-infer-dec.json`, CAL698
  temperatures), typed FINAL + CSS15 + public 231, then `mlx-diag`; seal, report and paired
  compare against the Lux1 16K run (65.808). Reported: v3, T/H, Choice/Noul/Score, typed
  families, per-task CSS15 with **H with and without FLUTE** (and GoEmotions, a same-source
  task present in A0s, flagged), public 231 E/S/H, typed/CSS calibration, invalid counts,
  mlx-diag. **Release gate:** paired 95% lower bound > 0. A passing candidate is uploaded to
  a private HF staging model repo (profile `qwen-adapter`, base pinned to Lux 1.0
  `bd45a30a…`) and reported immediately with the scored run directory.

## Preflights, stop rules and budget

- Per arm (primary seed, before its first full run): zero-step parity (untouched source =
  reloaded zero-step), one update + bitwise reload, cross-process drift with the persisted
  cache (`v2.dec.preflight_dec`). The second seed starts only after the preflight passes.
- A failed preflight, nonfinite loss or gradient, identity drift or a native fault stops that
  arm (recorded, no blind retry); a reproduced ROCm backward SIGSEGV stops all 9B training.
- Budget (estimated from Milestone 2 at 3.5× the one-row throughput): A and B ≈ 1.5 GPU-h per
  seed, C ≈ 2.8 GPU-h per seed, formal runs ≈ 0.5 GPU-h each. **Milestone 3 cap: 20 GPU-hours**
  including preflights, readouts and formal runs; GPUs lent for eval formal/C1 events are
  returned on request.
