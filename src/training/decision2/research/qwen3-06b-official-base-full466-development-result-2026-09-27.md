# Official Qwen3 0.6B full arm: selected-checkpoint development result

**Development gate: PASS; release evidence: incomplete.** This result follows
the frozen [full-arm protocol](qwen3-06b-official-base-full-clean-v2-prereg-2026-09-27.md).
The candidate is the prescribed BEST step 466 of a complete, one-epoch run
initialized from official `Qwen/Qwen3-0.6B-Base` revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, with a newly initialized
shared option head. No third-party decision-model weights initialized this arm.
The complete model has 597,103,104 trainable parameters. Its 700-item SELECT
result is 562 correct, with family-macro accuracy .77259. SELECT chose the
checkpoint before typed DEV or CSS pilot was evaluated. This is not a sealed
JevArena result or a reason by itself to publish the model.

## Native checkpoint and panel integrity

The selected full checkpoint loaded in the pinned ROCm runtime from its own
saved backbone, head and tokenizer. Reloading the first 32 SELECT examples
with the trainer's batch size of two reproduced the in-process predictions:
**zero category changes, zero p99 probability drift and zero maximum drift**.
The pass receipt is `3a8e165b2a3a1ce188072a477216a19357025ee6b8e33d6ada3240b1f2a5ee77`.
The model's inference fingerprint is
`5380e01e3fbb5f541d6548144dfb9e0776292d28a52d490f90c8929f764f37ab`.
All six training-core source files exactly matched the training provenance;
the inference source came from local commit `1414e6d6d150fbde9d977945a4be23a7b8597368`.
The runtime image ID was
`sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`.

The gold-free typed DEV prompt bytes hash to
`a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a`;
the CSS pilot prompt bytes hash to
`598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda`.
Every one of 1,600 typed and 1,430 CSS pilot requests received an answer
without truncation or overflow. Predictions were sealed before their
development labels were scored. The typed prediction/report SHA-256 values
are `d45df6e0207c13daf93310ca9e5ca3c6557f8b3efe6bdbeeb61c0a31e4493156`
and `4c4cbb3235992199325ef0ce387809bd6b292f3db9b8e8acd9a3ca215e20f3be`;
CSS values are `a99a1a01f003571f671e17cb032a1fbb5988e492d4ed0c3522c08082ba4b32a3`
and `91bc7ed495b73c887a5aa4bb85ca02a0cbd4dab023001176ee1f99b2786e3acc`.
Predictions, labels and individual errors remain private.

## Frozen development gate

| Same development panel | Official Qwen full step 466 | Own Kai 1.0, archived exact panel |
| --- | ---: | ---: |
| Typed DEV family-macro / correct | .289375; 463/1,600 | .265625; 425/1,600 |
| CSS pilot task-median macro-F1 | .290959 | approximately .1748 |
| CSS pilot correct / valid | 504/1,430; 1,430/1,430 | 418/1,430; 1,408/1,430 |
| `100 × sqrt(T_dev × H_pilot)` | **29.02** | approximately **21.55** |

The CSS pilot candidate correct total is 169 discourse + 138 implicit-hate +
197 stance = **504**. Its task macro-F1 values are .290959, .248810 and
.366483 respectively. The archived Kai checkpoint and this candidate used
the same panel question hashes and fixed scorers; Kai's native 1,024-token
limit caused 22 pilot invalids, which remain failures in the denominator.
The Kai task-median value is retained at the four-decimal precision of the
earlier signed comparison, so the composite difference is approximately
**+7.47 points**, safely above the frozen +2.0 development threshold. Typed
accuracy increased and CSS task-median macro-F1 did not decrease. The gate
therefore passes without changing its thresholds.

The candidate is still far below verified same-panel near-size references on
typed DEV: Laya 638/1,600 and GLiNER2.5-Decide 652/1,600. These are open
development comparisons through their own native adapters, not formal ranks.

## Mechanism and calibration failures

Typed DEV by type is Choice **186/800**, Noul **192/400**, Score **85/400**.
The four families are attribute gate 110/400, rule precedence 192/400, set
reconciliation 85/400 and transition table 76/400. All **400 Score point
predictions select level 0**; the 85 correct answers are exactly the 85 gold
level-zero rows. Thus the SELECT improvement did not establish ordinal
transfer. Counterfactual relation consistency is only 29/400 and paired
joint accuracy 9/400; option-order and label-renaming joint accuracy are
69/400 and 83/400. These related variants are not independent items.

The completed run's isolated CAL700 fitted temperatures Choice 1.07163,
Noul .72328 and Score .38298. On CAL, total NLL improved .44706→.42745
and normalized Brier .12003→.11902. On independent **development** typed
DEV, however, that same fixed fit worsened normalized Brier .38385→.44577,
ECE10 .24614→.32686, and Score expected-value MAE .95866→1.20803.
The uncalibrated diagnostics were reconstructed algebraically from the sealed
calibrated probabilities, without another model run or changing answer ranks.
No calibrated reliability or Score advantage is claimed.

The rights-clean v2 TRAIN has only 516 Score examples across varying ordinal
scales, whereas typed DEV Score uses three levels and a distinct set
reconciliation mechanism. This is a plausible distribution-shift explanation,
not a causal conclusion. A separate, equal-budget Score curriculum arm and
new-source transfer comparison would be more discriminating than selecting
another checkpoint from this completed run.

## Resource and next boundary

The completed training run spanned 17 minutes 30 seconds wall time on one
GPU, an approximately .292 GPU-hour occupancy proxy. The current reload,
CAL, DEV and CSS validation used one reserved GPU for under .19 GPU-hour; the
two full gold-free panels accounted for 61.84 seconds of per-request measured
inference latency, excluding model loads. The task reservation was released
after scoring. No protected typed FINAL, 15-task transfer, public JevBench,
HF upload or model publication was run in this development gate.

**Decision:** retain the exactly selected full checkpoint as a research
candidate. Its predeclared development gate passes, but this does not prove a
first-release gain. Before promotion, compare a frozen package on JevArena v3
against own Kai and a near-size open peer, and independently test the observed
Score and counterfactual weaknesses. Because earlier 0.6B formal labels were
already accessed by this project, any later v3 result is a post-key same-panel
comparison rather than an untouched blind result.
