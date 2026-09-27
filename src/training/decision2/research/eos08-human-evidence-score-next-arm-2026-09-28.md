# 0.8B: one Score-only human-evidence substitution

**Disposition: CPU admission first; no optimizer is authorized by this note.**
This is a prospective experiment, not a model result or a release claim. Its
purpose is to separate an ordinal-data failure from an optimization/head
failure without repeating the completed 0.8B controls. It supersedes neither
the failed Eos hard/soft replay result nor the broader, still unadmitted
multi-mechanism Score proposal.

## Why this contrast

Our eligible `Decision-1.0-Eos-0.8B` initializer at revision
`3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd` has a same-source, fixed
498-update hard-label control. That control answered 797/1,600 typed DEV
questions, including 510/800 Choice, 202/400 Noul and **85/400 Score**;
all 400 Score outputs were level 0 although the gold levels are
85/107/208. Its development `100 sqrt(T×H)` proxy was 32.8177, compared
with 30.8286 for Eos 1.0. The matched soft replay gained only 0.1627 proxy
points over hard replay, missed its preregistered gate and retained the same
all-zero Score behavior. CAL temperature did not change those decisions
([fixed readout](eos08-own-source-replay-final-hold-2026-09-27.md)).

The existing [shared 0.8B/2B proposal](small-model-08b-2b-next-arm-audit-2026-09-27.md)
would replace 384 of the 512 repeated Eos TRAIN occurrences with newly authored
Score rows. It is blocked on source quality. It also changes whatever
Choice/Noul exposure lies in those 384 slots. This narrower contrast changes
**only Score repeat slots** and preserves every Choice/Noul and original human
row. The frozen hard arm is the control; do not retrain it.

## Data admission and exact intervention

Use the publisher's **original OCNLI TRAIN** as the candidate human-label
source for a native System One Score question: given premise evidence and a
claim, rate `refuted / undetermined / supported` as ordered levels 0/1/2.
This is an evidence-relation rubric, not a general ordinal-effect or
rule-precedence label. The [publisher source](https://github.com/CLUEbenchmark/OCNLI)
and the [existing source audit](score-evidence-nli-source-audit-2026-09-28.md)
identify 39,762 eligible non-news pairs in 6,314 conservative premise groups
after setting aside the news genre's additional text-rights issue. The audit
also found a 51.81% hypothesis-only balanced-accuracy shortcut and did **not**
admit any row. The 9,444 long protected leaves excluded from its near scan,
suspected source overlaps and publisher-ID collisions remain unresolved.

Before training, recover the exact 512-repeat manifest, row order and native
tokenizer from the archived Eos hard arm. Let `K` be its count of Score
occurrences; do not infer it from another model's 512-row sample. Retain every
non-Score occurrence unchanged. Replace precisely those `K` Score occurrences
with `K` distinct OCNLI **non-news** premise groups, with the three relation
counts differing by at most one. Prespecify deterministic group selection by a hash
of source revision plus group ID; choose one row per group. Preserve the
original 7,455 TRAIN rows, SELECT700, CAL700, native question schema,
optimizer, seed, hard CE+Brier objective, 498 updates, LoRA/head topology,
checkpoint rule and no-truncation cap. No teacher KL or temperature change.

Admission is conjunctive:

1. Reverify original OCNLI TRAIN revision, file hash, source rights and
   required attribution. Quarantine the news genre and any inconsistent
   ID/normalized-premise component. Inspect a blinded, class-balanced rubric
   packet for mapping/ambiguity; a claim that is both supported and refuted
   under different unstated readings is rejected.
2. Complete a **gold-free** group-level exact/near audit against TRAIN,
   SELECT, CAL, typed DEV/FINAL prompts, CSS pilot/15 prompts and public
   supplements, including the protected long leaves omitted in the earlier
   heuristic scan. Quarantine suspicious components, not isolated pair rows.
   Source labels or formal answers must not be used to choose near matches.
3. Require `K >= 128`; otherwise the intervention is too small for this
   fixed-budget test. After selecting `K`, render and tokenize full native
   inputs. Preserve the control's total encoded TRAIN tokens within **1%**
   without padding, truncation or longer training. If this fails, record a
   token-matching failure; do not silently change `K`, the seed, steps or a
   different source after seeing results. Freeze replacement IDs, whole-group
   manifest, source/target token sums, package and code hashes before step 1.

The earlier CPU audit provides feasibility evidence only: OCNLI is short,
has a strong hypothesis-only cue, and has not passed these admission gates.
No OCNLI row is presently approved for this arm.

## Single-run preflight and readout

At the eligible source's zero step, require a 32-item native Choice/Noul/Score
reference/package parity test with zero category changes and the previously
frozen numerical tolerance, unchanged input/option identities, finite loss
and finite nonzero trainable gradients. One throwaway optimizer step verifies
reload and numeric stability; discard that smoke state. If the archived hard
control's exact sample/runtime/selector cannot be reconstructed, stop rather
than calling the treatment matched.

Run **one** fixed 498-update treatment. Save the final step for the causal
contrast; any intermediate checkpoint is diagnostic and cannot replace it.
Report SELECT700 by type and Brier, but do not adjust the arm or choose a
better step from SELECT. Freeze the package, then run the same raw native
typed DEV1,600 and CSS pilot1,430 once for treatment, archived hard control
and Eos 1.0; reuse a control's predictions only when prompt, package and
runtime fingerprints match exactly. Invalid/missing/over-budget answers fail.
Report each Score level's recall, the 400-item Score confusion matrix,
Choice/Noul changes, three human-task F1 values, proxy, Brier and paired
uncertainty at original group/task level. Fit CAL only after candidate choice.

For this **mechanism test**, advance to package consideration only if the
fixed treatment's open proxy exceeds Eos 1.0 by at least **2.0 points**,
Score improves by at least **40/400 correct** over the archived hard control,
both nonzero levels receive at least one correct prediction, and native
validity is no worse. Report all individual regressions; these thresholds
do not imply that every axis must improve for a later product release.
Before any post-key v3/public231 run, also lock a separate source-disjoint
three-level evidence diagnostic (for example, a rights-cleared ANLI portion)
and a distinct rule/state Score diagnostic, with whole-group overlap checks.
If their source identity or rights cannot be verified, the arm may remain a
development diagnostic but cannot support a transfer claim.

**Cost/stop:** CPU provenance, QA, dedup and token-matching first; GPU use is
zero until they pass. A passed arm needs one approximately 1.1-GPU-hour
optimizer run, based on the completed 1.076-GPU-hour hard control, plus at
most 0.6 GPU-hour for parity and the matched open readout. Record actual
GPU-hours. Stop on any failed rights, quality, overlap, token, source-parity,
numeric or validity gate; keep the failure. A null result despite balanced
human evidence favors examining the native Score head/learning dynamics
before importing more same-family labels. A positive result establishes only
this evidence-rubric intervention; it does not prove broad Score or human-task
transfer. Formal JevArena v3 and JevBench remain separate release gates.
