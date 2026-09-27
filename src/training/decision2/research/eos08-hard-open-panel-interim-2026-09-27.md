# Eos 0.8B matched replay: hard-arm open-panel interim

This is a **development-only** readout under the frozen [v2 matched replay
protocol](eos08-own-source-soft-replay-v2-prereg-2026-09-27.md) and its
[cross-node addendum](eos08-cross-node-dev-readout-addendum-2026-09-27.md).
The matched soft arm is still running. Nothing here is a JevArena FINAL,
CSS15, public JevBench, release, or causal soft-versus-hard conclusion.

## Identity and gates

- Initializer: our `Decision-1.0-Eos-0.8B` at revision
  `3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd`; the source package
  manifest SHA-256 is `4f1886e3df327cb0726086df1f41f1ab584ecc7ab80c0e065e68c7cb8dc96ec6`.
- Rights-clean v2 TRAIN/SELECT/CAL and the deterministic reference backend
  passed the prospective zero-step and one-step gates. The hard-label arm
  completed its frozen 498 updates; optimizer time was 1.076 GPU-hours and
  container wall time was 1.188 GPU-hours. The soft arm uses the same
  initializer, roster and update budget, with replay KL as the sole training
  intervention.
- The fixed-498 and SELECT-selected 480 hard checkpoints were copied with
  SHA-256 manifest checks. Cross-node parity passed independently for each:
  all 700 ordered SELECT rows, including the frozen 32, matched in prompt,
  tokens, input length, option domain and category; maximum offered-option
  probability drift was zero. The same deterministic native adapter was used
  on both nodes.
- SELECT at fixed 498: 576/700, family macro accuracy 0.78778, normalized
  family Brier 0.12897. SELECT-selected step 480: 578/700, family macro
  accuracy 0.79176, Brier 0.12866. The latter was selected only by the
  prospective SELECT rule, before open-panel scoring.

## Open development comparison

| Model and view | Typed DEV correct / 1,600 | Choice / 800 | Noul / 400 | Score / 400 | Typed Brier ↓ | CSS pilot median macro-F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Own Eos 1.0, native control | 792 | 510 | 197 | 85 | 0.36178 | 0.19200 |
| Hard replay, fixed step 498 | 797 | 510 | 202 | 85 | 0.43032 | 0.21621 |
| Hard replay, SELECT-best step 480 | 800 | 512 | 203 | 85 | 0.43174 | 0.21560 |

The 1.0 control was rerun on the same prompts through its published native
adapter. The best hard checkpoint's paired typed improvement is 8/1,600
or 0.5 percentage points. A bootstrap over 400 independent four-item groups
gives a 95% interval of −0.75 to +1.75 points. Its paired Brier increase is
+0.06996 (95% interval +0.06012 to +0.07947; higher is worse). The CSS pilot
has only three fixed tasks; the median macro-F1 improvement is +0.02360,
with within-task paired bootstrap interval −0.00388 to +0.05228. This is not
evidence of transfer across new tasks.

All 400 typed DEV Score questions are predicted as level 0 by both the own
1.0 control and hard step 480, while the answer distribution is 85 level 0,
107 level 1 and 208 level 2. The hard arm sharpens the wrong decision:
its expected score has median 0.00592 versus the 1.0 control's 0.129, and
its Score Brier rises from 0.66224 to 0.77941. SELECT Score uses a different
targeted-median mechanism; its improvement did not transport to typed DEV.
This mechanism and calibration failure must be resolved before calling the
arm a release candidate.

CAL-only temperatures fitted after the step-480 selection leave typed DEV
accuracy at 800/1,600 and worsen its Brier slightly to 0.43219. They improve
CSS pilot median probability quality, but do not repair the typed Score
mechanism. No open-panel metric was used to select a checkpoint or alter the
frozen soft arm.

## Decision

Keep the completed hard arm as the matched control. Await the soft fixed-498
and SELECT-best results, then run the already frozen cross-node parity and
same-panel readout. Do not open protected formal labels or treat the typed
eight-item gain as a release result. The first-node GPU used for this interim
readout has been released.
