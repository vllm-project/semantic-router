# Sol 2B matched inline soft-replay screen

Status: **prospective; no optimizer or selector result from this arm**. This is
one bounded development experiment. It preserves the completed targeted160 →
rights-clean-v2 control; that control must not be retrained or reset. The
synthetic FINAL, CSS15 gold, authored release panel and public JevBench are
excluded from training and checkpoint selection.

## Question and prior evidence

The completed targeted160 → clean-v2 2B control scored 933/1,600 on typed DEV
and 629/1,430 on the CSS three-task pilot. Its same-source merged targeted160
warm start scored 939/1,600 before continuation. The completed 270-row A1
short hard-replay screen added only 3.12% input-token mass and failed its
predeclared step-128 SELECT threshold (540 rather than 545/700); its source
was Sol 1.0, not the merged targeted160 source used here. Repeating that mix
cannot isolate forgetting in the transfer-favorable sequential arm.

This screen tests whether preserving the **merged targeted160 source's native
option distribution on existing structured TRAIN rows** prevents typed
reasoning regression during the same clean-v2 continuation, without changing
the training prompts, order, token mass, or optimizer horizon. It is not new
independent supervision and cannot qualify a release by itself.

## Fixed source, data and intervention

* Initialization is the existing FP32 materialized targeted160 Decision 2.0
  full source, merged-model SHA-256
  `2f4bb061e0881d2d5f29da1cedee8655655ce339bacfa4f6971bcd845add5ae9`.
  The source materialization receipt SHA-256 is
  `eae632fba65bc3b208aa3698d2334a88cbd5a0a4b7521ade9398fa3be3aacd1c`.
  Its BF16 output is numerically distinct from the original unmerged adapter;
  only the prior merged-source continuation is a causal control.
* The existing rights-clean-v2 TRAIN7,455 is byte-identical, SHA-256
  `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`.
  SELECT700 SHA-256 is
  `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`;
  CAL700 SHA-256 is
  `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`.
  All protected-panel isolation claims inherit the audited frozen data
  manifest. Neither SELECT nor CAL enters a gradient update.
* Teacher probabilities are generated in native BF16 inference by this exact
  merged source, before any treatment optimizer step, on the following ten
  complete, source-group-disjoint TRAIN families, with no gold-based filtering:
  `stage4_scope`, `stage4_automaton`,
  `stage4_replay_stage3_transition_set`, `stage4_replay_policy`,
  `stage4_replay_authorization`, `stage4_replay_stage3_evidence_scope`,
  `stage4_replay_cross_candidate_evidence_binding`, `stage4_replay_rubric`,
  `stage4_ordinal`, `stage4_replay_stage3_logic_score`.
  This is 894 existing rows/groups: Choice441, Noul142, Score311; English443,
  Chinese451. The ordered `[{id,input_sha256},...]` canonical-JSON roster
  SHA-256 is
  `4917cfb648e60929b7b24d9948d4c8e685510b2dff8b2429c4ca0e4a319933c8`.
  The private teacher artifact contains only these identities and option
  probabilities, no raw text or evaluation gold. Verify its source file map,
  receipt, train hash, roster, option keys and normalization before use.
* Every TRAIN row still receives hard-label CE plus 0.5 Brier. The 894 selected
  rows additionally receive `0.3 * KL(source_teacher || student)` on the
  native option probabilities during their **existing** training pass.
  There is no appended replay pool, extra sample or changed prompt. All
  nonselected rows receive zero KL. Teacher and student candidate keys must be
  aligned by key, never by sorted JSON order.

## Equal-schedule and start gates

Use the same rank-16/alpha32/dropout0.05 LoRA, LoRA LR `1.5e-5`, head LR
`7.5e-6`, microbatch1, accumulation16, CE+Brier0.5, max length8192,
seed20260926, step32 SELECT, warmup/cosine schedule and one-epoch **466-step
planned horizon** as the completed control. The sole intended optimization
difference is the inline KL term. Verify in a private preflight that all
7,455 encoded prompt/token IDs, all SELECT/CAL hashes, deterministic batch
order, number of updates and per-step token counts match the control recipe.
Any mismatch aborts rather than changing the control or training budget.
The historical control source and the current source differ in formatting and
later direct-LoRA support. Before this optimizer, AST comparison must establish
that canonical input, native segment/encode/collate, per-example loss,
deterministic batch/step planning, SELECT evaluation and LR schedule functions
are identical; a separate two-runtime token/schedule audit must confirm the
actual frozen 7,455/700 inputs and all 466 update windows. The original
control's provenance and trainer hashes remain preserved, not overwritten.

Before optimizer start, the new loader must pass contract tests and the exact
teacher/source/data verification. The zero-step parity sample is the first
32 SELECT rows ordered by SHA-256 of UTF-8 row ID, with ordered
`[{id,input_sha256},...]` roster SHA-256
`4c247f30d6a879ce20d77acce1a0d9fdbd0d7c9b5a120f6e9b8febd499ad199a`.
The existing control's frozen, uncalibrated `select-baseline-predictions.jsonl`
has SHA-256
`ceda618a37069832af69375bc5905b28da952f8386be972c88dc6a30671c8cb3`.
Compare only identities, prompt/token hashes and predictions; do not read its
gold or aggregate SELECT accuracy. Native zero-step treatment must match the
prior merged-source control's Choice/Noul/
Score categorical outputs on all 32 with maximum option-probability drift
at most `1e-4`, no invalid response and no truncation. Abort if a different
runtime cannot achieve this parity. Freeze the container digest, local source
commit, source/teacher/calibration hashes, exact command and zero-step receipt
before the first update.

The historical SELECT output used batch size two in original partition order.
The parity probe must retain each chosen row's original batch mate and padding;
single-row BF16 inference is not an equal-shape comparison. This was corrected
after a pre-optimizer dry run exceeded the probability-drift threshold, before
any teacher artifact or treatment update was written. The strict `1e-4`
threshold and fixed 32-row roster remain unchanged.

Start the full one-epoch optimizer schedule; **do not** set `max_steps=128`
because that changes the learning-rate schedule. Externally stop only after
the complete atomic checkpoint128 and its SELECT receipt are sealed. The
existing control at step128 scored 592/700, family macro `.82462963`,
GoEmotions Choice144/200 and Noul175/200. Continue the same optimizer run
only if treatment step128 reaches **at least 598/700**, family macro **at
least .830**, GoEmotions Choice **at least 142/200** and Noul **at least
173/200**, with 700/700 valid. This screen asks for more than a two-item
fluctuation while protecting the human slices. If it fails, stop: no alternate
checkpoint, CAL/DEV/CSS, or public-panel selection.

If it passes, complete at most step466. Freeze BEST from SELECT family-macro
accuracy, normalized Brier then earliest step. Fit native Choice/Noul/Score
temperatures on CAL700 only after BEST is frozen, and evaluate the same typed
DEV1600 and CSS pilot1430 once. A development advance requires all of:
DEV at least **946/1600** (59.125%, above Sol1 and the completed sequential
control), CSS micro at least **629/1430**, CSS median task macro-F1 at least
**.36183**, no decrease on any of the three CSS task accuracies against the
completed sequential control (195/497, 160/498, 274/435), 100% native-valid
responses and no truncation. Report paired independent-group uncertainty,
calibration, every task/type/family and failure cases regardless of outcome.
The CSS pilot contains same-task TRAIN families, so it is not a blind transfer
claim. Public JevBench may be run only after development selection is frozen,
for descriptive same-panel reporting; it is never a selector or a promotion
condition. No sealed FINAL/CSS15 or authored release gold is opened.

## Abort and interpretation

Abort on source/teacher/roster/hash mismatch, malformed probabilities,
over-budget prompt, zero-step parity failure, nonfinite gradient, missed
step128 checkpoint, GPU conflict or incomplete fixed protocol. Preserve
completed control and private output artifacts. A SELECT gain with DEV loss
is forgetting, not success. A positive CSS pilot alone does not prove new-task
transfer. There is one KL weight and one predeclared cohort; do not tune either
on this selector. No HF upload, model publication or release-score claim may
follow this arm alone.
