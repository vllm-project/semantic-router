# Sol 2B: formal gap audit and next data-arm admission

**Status: CPU audit and prospective hypothesis; no new training, scoring, or
release candidate.** The eligible own-Sol BEST160 package remains HOLD. Its
same-panel v3 score is 43.9596 against Sol 1.0 at 45.5804 and the pinned
Decider 2B peer at 49.4992; public231 counts are 162, 161, and 175. The
development proxy gain of 2.191 points did not transport to v3. These are
post-key comparisons, not a never-opened test or a new peer ranking.

This audit read the existing nine private typed/CSS/public score reports from
the completed [formal experiment](sol2b-own-source-formal-v3-result-2026-09-27.md).
The three typed reports share gold SHA-256
`707dd28dfbab10d124d437434023729f319501542e999e536fce9b7ff7f2361e`
and identical metric policy; the three CSS reports share gold SHA-256
`1cda9623032138bb7b124be0c1b0a4239c06bed7be3169264e6eb31805c19ba4`
and identical metric policy. Candidate typed/CSS score-file SHA-256 values are
`975f456ff1eb1a1860480802c1ff199355ee2defd375c32ab75980273475f6af`
and `373efaf30575ac96c3540638dad0cb22fe29fd511405a78f01a8ee85d44263af`.
No original examples, protected answers, private paths, or predictions were
copied into this note. This is analysis of revealed existing results, so none
of the observations below may become a checkpoint selector on v3.

## Where the gap actually is

| Typed FINAL family | Sol 1.0 | Own 2.0 candidate | Decider 2B peer | Candidate − Sol 1.0 |
| --- | ---: | ---: | ---: | ---: |
| Constraint competition | 58/400 | 74/400 | 346/400 | +16 |
| Evidence connection | 572/800 | 544/800 | 484/800 | −28 |
| Exception stack | 177/400 | 162/400 | 212/400 | −15 |
| Resource / Score | 154/400 | 137/400 | 133/400 | −17 |

The **272-item constraint-competition gap to Decider** is the largest
same-size typed gap; the candidate improves this family over own 1.0 but
remains at 18.5% versus the peer's 86.5%. Its overall typed correct count
falls from own 1.0's 961/2,000 to 917/2,000. By request type, the candidate
versus own 1.0 scores Choice 357 versus 366/800, Noul 423 versus 441/800,
and Score 137 versus 154/400. Decider scores 545, 497, and 133 on those
types. The candidate's counterfactual joint-correct count is 103/500 versus
own 1.0's 121 and Decider's 138; label-renaming joint-correct is 207/500
versus 218 and 271. On the abstention view, both own models identify the
same 218 true positives, but the candidate has 117 false positives versus
own 1.0's 92. These are associated failure modes, not a diagnosis of the
model's internal mechanism.

Transfer is mixed rather than uniformly worse: the candidate improves on
eight of 15 CSS tasks, and micro accuracy rises from 0.5007 to 0.5062,
while task-median macro-F1 falls from 0.492462 to 0.479368. Selected
per-task macro-F1 values illustrate the most useful contrasts:

| CSS task | Sol 1.0 | Own 2.0 candidate | Decider 2B | Candidate − Sol 1.0 |
| --- | ---: | ---: | ---: | ---: |
| IBC | .4359 | .4041 | .3222 | −.0318 |
| TalkLife | .3245 | .2976 | .3480 | −.0268 |
| Reddit Humor | .4203 | .3988 | .5693 | −.0215 |
| Emotion | .4935 | .4794 | .7768 | −.0141 |
| FLUTE | .6628 | .7236 | .4430 | +.0608 |

Emotion has the largest candidate-to-Decider task gap (−.2974 macro-F1),
but Decider's task-median `H=.420180` is below own Sol 1.0's `.492462`.
Thus its v3 advantage is driven primarily by typed reasoning, especially
constraint competition, rather than a uniformly better transfer profile.
The candidate's typed normalized Brier `.3534` versus own 1.0 `.3453`
further argues against describing the modest public231 +1 as broad progress.

## One new, falsifiable hypothesis

**Hypothesis:** With the Sol 1.0 direct initialization and the archived
rights-clean-v2 optimizer, replacing a fixed token share of old, nonhuman
synthetic TRAIN with *new source-disjoint, program-oracled policy conflicts*
will improve native Choice/Noul constraint and exception decisions while
retaining the existing human-label and Score exposure. This is a data
substitution test, not a new backbone claim. It is materially different from
the stopped A1 replay: A1 added 270 mostly arithmetic/ordinal rows, only
seven explicit transition-set cases, and 3.12% input-token mass. It is also
different from the stopped own-Sol soft-KL arm, whose exact BF16 zero-step
probability gate failed before optimization, and from the unadmitted clinical
directional-Score proposal. No teacher KL, new Score interpretation, or
revealed v3 example is part of this treatment.

The proposed source is authored from fresh structured policy/state records:
eligibility, permission, scheduling, and inventory are four separate domain
families. A deterministic executable oracle fixes priority, exception scope,
state updates, and the correct candidate before natural-language rendering.
Each independent situation yields one native Choice, one Noul, and one
three-level Score request with different evidence and candidate orderings;
counterfactual and label/option permutations remain inside its single group.
The Score rubric must encode an ordered policy severity or priority level
that the oracle can actually verify, not an arbitrary three-class label.
Scenarios, wording and generator code must not derive from JevArena typed
FINAL, CSS15, JevBench or the existing stage3/stage4 TRAIN templates. Fresh
rendering alone is not evidence of independent situations.

### Admission before any GPU hour

1. Build one frozen roster of **512 independent groups / 1,536 native
   requests**, exactly 128 groups per domain family and balanced Choice/Noul/
   Score token shares within ±5 percentage points. Preserve all existing
   2,800 human GoEmotions rows and all 516 old Score rows in the 7,455-row
   clean-v2 TRAIN. Replace only whole old nonhuman, non-Score groups/slots.
   Target new-source exposure of **20–25%** of the archived 4,194,465
   Sol-token TRAIN budget and total run tokens within **±1%** of that budget,
   at the same 7,455 row slots and **466 updates**. If group completeness,
   replacement capacity or token matching fails, this arm is HOLD; do not
   append rows, pad/truncate inputs, loosen the range, or switch source.
2. Independently reexecute every oracle answer; reject contradictory rule
   orderings, unresolvable states, ambiguous severity levels and any case
   whose answer changes only because of option position. A blinded review of
   48 groups, 12 per family, must find at least **44/48 unambiguous** across
   all three native outputs. Keep all rejected IDs and reasons. A state-
   removed/label-only shortcut probe must stay at most **5 percentage points
   above each type's fixed majority baseline** on an independent 128-group
   diagnostic; failure stops the source. Freeze this diagnostic before
   training and keep it outside TRAIN/SELECT/CAL.
3. Register origin, rights and redistribution scope for every text shell.
   Quarantine exact, near and suspected semantic overlap at source/group level
   against TRAIN, SELECT700, CAL700, typed DEV/FINAL gold-free prompts,
   CSS pilot/15-task gold-free prompts and public231. Do not inspect formal
   labels. Any unresolved cross-split group or license blocks admission.
   Require zero detected exact/near cross-split groups and report the
   quarantine count, generator/template families, label balance, language,
   length, per-type tokens and source hashes.
4. Verify the direct Sol 1.0 source revision, original 7,455-row control,
   tokenizer, trainer and original fixed batch order. Run two independent
   zero-step native SELECT700 readouts in the **same runtime and batch shape**;
   require 700/700 valid, identical categories and max option-probability
   drift ≤.005. Compare the zero-step output to the archived direct-v2
   control under the same bound. If that historical parity cannot be
   recovered, stop before training; do not silently rerun or redefine the
   control. A one-update save/reload smoke must be finite and meet the same
   category/probability bounds on the fixed first 32 SELECT rows.

Only after those gates pass, freeze the exact replacement IDs, new roster and
oracle, per-step token counts, seed, source/data/code/image revisions and
SELECT rule in a signed receipt. Reuse the direct clean-v2 control's rank-16
LoRA/head, CE + 0.5 Brier, one-epoch 466-update budget, and SELECT every 32
updates. The archived direct-v2 run is the no-new-source control; do not
retrain it. CAL700 is fit once only after one BEST checkpoint is selected by
family-macro accuracy, normalized Brier, then earliest step. The new-source
diagnostic is reported separately and must not pick checkpoints.

**Frozen development stop rule to register with the admitted roster:** stop
at the complete step-128 checkpoint if SELECT700 has fewer than **545/700**
correct or family macro below **.713** (the failed A1 screen's prospective
threshold), or if either GoEmotions Choice/Noul slice drops below **140/200**
and **174/200**, respectively. If it continues, evaluate the single BEST
package on typed DEV1600 and CSS pilot1430. Advance only if their frozen
`100 × sqrt(T_DEV × H_pilot)` proxy is at least **47.15** (about +4 over
own Sol 1.0 and above the previously misleading +2.19 candidate), all
in-budget answers are valid, Score is at least **280/400**, and CSS task-
median macro-F1 is at least **.33**. This allows tradeoffs while rejecting
severe Score or transfer collapse. The independent 128-group source-disjoint
diagnostic must show a positive paired group-bootstrap lower confidence
bound against own Sol 1.0; otherwise no formal run. If any gate fails, record
the failed arm and stop without new calibration search, alternate checkpoint,
public231 selection, v3 comparison, or HF upload. If all pass, freeze a new
gold-free formal roster and seek an additional source-disjoint external check:
the already revealed v3 panel is comparable but cannot alone establish
independent generalization.

This proposal is **not admitted yet**. In particular, the 20–25% token
replacement capacity, oracle quality and zero-step parity are unverified;
they are cheap CPU/short-smoke gates, not reasons to reserve a long GPU run.
