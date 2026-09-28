# 9B Milestone 2: Lux 1.0 continuation arms (preregistration)

Status: frozen 2026-09-28 before any Lux training step. Route approved by the
coordinator after the [Milestone 1 result](clm-frozen-ablation-result-2026-09-28.md):
the CLM disaggregated readout is a completed negative finding; Lux
continuation is the 9B route. Development readouts are never release scores;
v3 is post-key and only runs through the eval track's frozen runner.

## Fixed configuration (every arm)

| Field | Value |
| --- | --- |
| Start | own `llm-semantic-router/Decision-1.0-Lux-9B@bd45a30a…` (bundle `985ade73…`; weights byte-identical to Hub head `cdf4d3ef…`), 7,940,895,744 deployed parameters |
| Trainer | decoder track `v2/dec/train_dec.py`, unchanged, at this branch's merge of the integration branch; LoRA r16 / α32 / dropout .05 on every Qwen3.5 projection plus the shared candidate head |
| Objective | CE + 0.5·Brier over offered options (shared `per_example_loss`) |
| Optimizer | LoRA lr 5e-5, head lr 2.5e-5, residual lr 5e-4, weight decay .01, warmup .05, one epoch, microbatch 1 × accumulation 16, max length 8,192 without truncation, gradient checkpointing |
| Order / seed | shared length-bucketed deterministic order, seed 20260926 (extra seeds 1, 2) |
| SELECT / CAL | SELECT 700 (`32a4352d…`) at eight evenly spaced checkpoints, family-macro accuracy with earliest tie (`matrix-v1`); CAL 700 (`3e34f6cb…`) per-type temperatures for BEST only |
| Runtime | pinned image `sha256:f83b1d10…` with the FLA 0.5.2 overlay (the eval track's pinned runtime), node A GPU2–4 only |

This is the decoder track's C1 configuration applied at 9B, so arm effects are
comparable across tiers.

## Arms (one factor each versus L0)

| Arm | Factor | Data |
| --- | --- | --- |
| L0 control | — | A0, 7,455 rows (`61740be4…`) |
| L1 ordinal Score readout | `--residual ordinal_score`: zero-gated absolute level readout (query-predicted location and precision on the normalized level axis; exactly Lux's logits at step 0) | A0 |
| L2 own-Lux soft replay | + 0.5·KL(Lux ‖ student) on every TRAIN row; teacher = decoder-track Lux labels `752b7c8f…` (7,455 rows) | A0 |
| L3 A6g Score data | 25% of A0 rows replaced by whole A6g groups (content `ac94b7f4…`, L=2..10 generated, gates at chance) | A0 ⊖ 25% ∪ A6g |
| L4 A6h human ordinal Score | as L3 with A6h (content `23440f0c…`) | A0 ⊖ 25% ∪ A6h |

Substitution (`lux9b/mix.py`, seed 20260928) removes random whole A0 groups
until ≥25% of rows are gone and adds random whole arm groups until the added
rows reach the removed rows, so updates match L0 to within one group. This is a
budget-driven deviation from matrix template S (A0 ∪ X(ρ)), disclosed as such:
S would add 50–70% more tokens per arm at 9B.

**Hard-negative objective: not added.** The CLM gain came from putting a
question's own distractors into an in-batch InfoNCE denominator; the joint CE
already normalizes over exactly those own options, so there is no cheap,
distinct objective to add. The data-level form (A4 hard vs random distractors)
is still in repair at the data track and is deferred until it is admitted.

## Sequence (cheapest and most informative first)

1. Same-runtime Lux 1.0 zero-step typed DEV / CSS pilot readout (the paired
   control for every arm).
2. Preflights per arm: zero-step parity (untouched source equals reloaded
   zero-step, exact) and one-update + bitwise reload (the decoder track's
   `preflight_dec`); L1's one-update window must contain Score rows.
3. Wave 1: L0, L1, L3 in parallel (readout versus data for Score, Lux's
   weakest v3 axis at 228/400).
4. Wave 2: L2, L4, and one combination of the best Score factor with the best
   other factor only if both wave-1 effects are positive.
5. Seeds 1 and 2 for L0 and the best treatment (paired); a second treatment
   only while the Milestone 2 total stays ≤ 10 GPU-hours.

## Readout, comparison and promotion

- After a run: CAL temperatures for BEST, one typed DEV 1,600 and CSS pilot
  1,430 readout with the same-runtime native adapter (`infer_dec`), scored by
  `v2/dec/dev_readout.py` (unchanged scorers' answer functions, 10,000-draw
  paired bootstrap, typed groups within family, CSS items within task).
- Reported per arm: typed-DEV Choice, Noul and Score counts, CSS pilot per task
  and median macro-F1, T, H and the scalar proxy, each paired against L0 and
  Lux 1.0. A factor counts only with a 95% lower bound > 0 on its targeted
  metric (Score for L1/L3/L4, CSS for L2) and no floor breach: typed type
  −3.0 points, CSS pilot −1.5, CAL Brier +0.010 (data-track matrix floors).
- Formal post-key v3 + public 231 (eval track runner, node A, persisted Triton
  autotune cache, against the node-A Lux1 comparator 65.808 / 183) only for an
  arm with scalar proxy ≥ 72.33 and a multi-seed positive paired interval. An
  arm with a clear multi-seed Score gain and no floor breach but proxy < 72.33
  is reported to the coordinator before any formal run. The eval track's
  calibrated proxy replaces the scalar one if it is published first.

## Stop rules and budget

Preflight failure, nonfinite loss or gradient, identity drift, or a native
fault stops that arm; a reproduced ROCm backward SIGSEGV stops all 9B
training and is recorded without a blind retry. Per-arm cap 1.5 GPU-hours;
Milestone 2 total ≤ 10 GPU-hours including preflights, readouts and any formal
run.
