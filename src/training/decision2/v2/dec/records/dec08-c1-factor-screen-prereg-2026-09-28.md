# DEV2.0-0.8B contrast set C1: equal-budget factor screen from own Eos 1.0

Status: **pre-registered before any C1 GPU launch** (2026-09-28, decoder track).
Development evidence only; typed DEV and CSS pilot are open development panels,
not release scores. JevArena v3 and public JevBench 231 are touched only by a
frozen candidate under the formal rule below and are then labeled post-key
same-panel comparisons.

## Why this set

Completed evidence this set builds on (not rerun):

- Own Eos 1.0 on typed DEV answers Score by a constant prior: level 0 receives
  mean probability .884 (p10 .835, p90 .923) on all 400 three-level rubric
  Score items, so Score is 85/400 = the gold level-0 share. Noul
  `rule_precedence` is 197/400 (chance). Dev proxy `100·sqrt(T·H)` = 30.8286
  (T .495, H .192).
- Prior Eos LoRA continuation on rights-clean v2 (hard 512-repeat arm) moved
  SELECT from 461/700 to 578/700 but typed DEV only 792→800 and H .192→.216
  (proxy 32.82); the matched Eos self-KL soft replay added +0.163. SELECT fit
  does not transfer; Score stayed all-level-0.
- Official Qwen3.5-0.8B Base / general full-backbone clean-v2 arms reached
  typed DEV 392/1600 and 509/1600 (H .270 / .210), far below Eos 1.0 on `T`.
  With only the rights-clean v2 mix they are not competitive starts; they are
  not rerun here and return only with a new admitted data arm.
- JevArena v3 typed FINAL Score (`resource_ledger`) is a five-level numeric
  scale, like TRAIN's quantized-median rows; DEV Score is a three-level rubric.
  Score competence has to transfer across both shapes.
- TRAIN has almost no counterfactual groups (75 quantized-median, 30
  COSMOS/SNLI), so a contrastive/hard-negative objective is deferred until the
  research & data track delivers grouped counterfactual arms.

## Common frozen configuration (control and every treatment)

| Item | Frozen value |
| --- | --- |
| Start | `llm-semantic-router/Decision-1.0-Eos-0.8B@363c4a5e56afc115b1c78c837633956d0bbb63ab` (weights byte-identical to the `3c2d6326` revision used by prior Eos arms: backbone `613f491d…98abd`, head `75f4ee8b…d72f6`, tokenizer `06b95093…e0a523`); shared `from_decision1` loader; 753,446,208 loaded parameters (text 752,393,024 + head 1,053,184) |
| TRAIN / SELECT / CAL | rights-clean v2 `61740be4…3755` (7,455) / `32a4352d…f2a6` (700) / `3e34f6cb…af60a` (700); manifest `61aa8830…e9ee8` |
| Budget | one epoch, 466 updates, microbatch 1 × accumulation 16, the shared deterministic length-bucketed order (seed 20260926), 4,194,465 TRAIN tokens, max length 8,192, no truncation |
| Trainable | LoRA r16/α32/dropout .05 on every Qwen3.5 projection (shared `select_target_modules`), plus the shared candidate head |
| Optimizer | AdamW wd .01; LoRA LR 5e-5, head LR 2.5e-5; 5% warmup then cosine to 10%; grad-norm clip 1.0 |
| Objective | CE + 0.5 × summed Brier over offered candidates (shared `per_example_loss`) |
| Precision/runtime | FP32 master weights, BF16 autocast backbone, FP32 head/loss; image `sha256:f83b1d10…c2d54` (torch 2.12.0+git6bbd260, transformers 5.17.0, peft 0.21.0, fla 0.5.2); node A GPU5 only |
| Selection | SELECT every 32 updates and at 466; BEST = SELECT family-macro accuracy, then normalized Brier, then earliest step |
| Calibration | per-type temperatures fit once on CAL for BEST only (shared `fit_report`) |
| Readout | one calibrated BEST pass on typed DEV1600 (`a17ec4b6…cf79a`) and CSS pilot1430 (`598319a4…3def3dda`); temperature changes no argmax, so T and H are calibration-invariant |

Code: `v2/dec/train_dec.py`, `dec_model.py`, `teacher_label.py`,
`infer_dec.py`, `calibrate_dec.py`, `preflight_dec.py`, `dev_readout.py`,
`launch.sh` at commit `11f02b8d1b8f11f745ce1d8af9836eb0a59605e2` (tree
`a8b7b022…`), exact-mirrored on node A; its nine contract tests pass in the
pinned image. Model snapshots (Eos 1.0, Lux 1.0, Qwen3.5-0.8B Base and
general) were HF-CLI downloaded at the pinned revisions and all files match
the Hub hashes. No DEV, FINAL, CSS or public item or label enters training,
selection or calibration.

## Arms (one registered factor each)

| Arm | Factor | Only change vs A0 |
| --- | --- | --- |
| C1-A0 | control | none |
| C1-A1 | multitask loss balancing | per-example weights giving each native type equal total TRAIN weight: Choice .63587 (3,908 rows), Noul .81986 (3,031), Score 4.81589 (516); window-normalized as in the shared trainer |
| C1-A2 | own-family soft targets | + 0.5 × KL(teacher ‖ student) on **all** 7,455 TRAIN rows; teacher = own `Decision-1.0-Lux-9B@cdf4d3ef2dda21518e599fe99ebbe468486b197c` scoring the same shared rendering, softened by its published temperature 2.00544 |
| C1-A3 | Score head: ordinal absolute-probability residual | + zero-gated query-predicted unimodal level prior on Score rows only (`OrdinalScoreResidual`); residual LR 5e-4 |
| C1-A4 | decision readout | + zero-gated second candidate readout over a learned softmax mix of all decoder layers (`LayerMixResidual`); residual LR 5e-4 |

A2's teacher adds no rows or tokens; its labeling is a separate GPU job whose
only gate is 7,455/7,455 finite, normalized, key-complete distributions. Its
TRAIN-label agreement by type is reported, not gated. Replay from Eos 1.0's
own distribution is not repeated (prior +0.163).

## Preflight (per arm, before its full run)

Two throwaway runs of the exact arm configuration: `--zero-step-only` and
`--max-steps 1`. `preflight_dec.py` must PASS all gates on the full SELECT700:
source vs in-memory zero-step arm and vs reloaded zero-step checkpoint (same
argmax 700/700, drift ≤ 1e-5); reloaded one-step checkpoint vs trainer
in-memory outputs (700/700, drift ≤ 1e-4); finite loss/gradient; LoRA, head
and residual gate all moved. Also: pinned-image torch contract tests pass;
Eos 1.0 zero-step typed DEV/CSS readout in this runtime reproduces the prior
readout's T/H (reported). A failed gate stops that arm; it is recorded and not
retried under the C1 name.

## Stop rules

- Nonfinite loss or gradient: stop the arm.
- Projected exclusive-GPU time above 3.0 GPU-hours per arm from the one-step
  timing: stop before the full run.
- Fixed horizon: no early stopping; no checkpoint other than BEST is read out.

## Decision rules

Factor effect = arm − A0 on the dev proxy with a 5,000-draw paired bootstrap
(seed 20260928; typed generator groups within family, CSS items within task).
**Confirmed** if Δproxy ≥ +1.0 and lower 95% bound > 0; **suggestive** if
Δproxy ≥ +1.0 otherwise; else **negative**. Report Choice/Noul/Score, Score
level distribution, per-family and per-task results and calibration for all.

Conditional combination C1-A5: if at least two factors have Δproxy ≥ +0.5,
a union arm is pre-registered in a committed amendment before launch (same
budget and rules), run once and read out once.

**Formal rule.** The single best arm by dev proxy (A0–A5) becomes a formal
candidate only if its proxy ≥ 33.8286 (Eos 1.0 + 3.0), its paired lower 95%
bound versus the same-runtime Eos 1.0 readout is > 0, and all answers are
valid. Its package fingerprint and calibration are then frozen in a committed
lock before any v3 prediction; typed FINAL 1,600 + CSS15 6,547 and public
JevBench 231 predictions are sealed before scoring with the eval track's
frozen runner (or, absent one, the existing `arena_v3` / `compare_v3` /
`jevbench_public` path at this commit), against Eos 1.0 on the same panel.
Results are labeled **post-key same-panel**. No second candidate is scored on
v3 in Milestone 1.

## Resources

Estimated GPU5 wall-clock: preflights ≈0.4 h, teacher labeling ≈0.3 h, five
concurrent arms ≈2.5 h, calibration and readouts ≈0.6 h. Actual start/end of
every container is recorded by `launch.sh` receipts; GPU-hours = wall-clock ×
1 GPU, with per-arm attribution when jobs share GPU5.
