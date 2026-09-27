# Sol 2B: human evidence Score next-arm screen

**Disposition: data admission HOLD; no optimizer or model evaluation.** This is
one prospective intervention from official Qwen weights, not a new 2B score.
The completed own-Sol, official Base and official Posttrained controls and
the stopped own-Sol soft-replay zero-step gate remain immutable. Do not rerun
those controls, relax their numerical gates, or use keyed v3/public231 labels
to choose this mixture.

## Why this arm

On the same open DEV/CSS pilot, our Sol 1.0 scored proxy **43.14498**, typed
Score **311/400** and CSS task-median macro-F1 **.315173**. The completed
official `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`
Posttrained arm scored proxy **41.91133**, Score **95/400** (predicted levels
0/1/2 on 390/0/10 rows) and CSS H **.358025**. Official Base scored proxy
**40.78168**, Score **241/400** and CSS H **.289243**. Both full official
arms completed 466 updates at **4,194,465 native TRAIN tokens**; neither
passed the frozen +2 development-promotion gate. Separately, own-Sol's
BEST160 v3 **43.9596** trailed Sol1 **45.5804**; public231 was 162 versus
161, far below the same-panel Decider2B's 175. These formal numbers are
post-key same-panel evidence, not fresh blinded estimates.

The official Posttrained start is the clearest *conditional* 2B test: its
real-task transfer is stronger, while its three-level Score has collapsed.
Preserving its human-task exposure and injecting independently labeled
ordinal evidence can falsify the hypothesis that new Score supervision
recovers the typed axis without losing H. It does not test a new architecture.
This is a single treatment against the archived official-Posttrained control;
any positive effect combines a new source with greater Score exposure and
cannot be called a source-only causal effect.

## Original human-label source screen (CPU only)

The [Evidence Inference 2.0 organizer](https://evidence-inference.ebm-nlp.com/download/)
publishes expert annotations, intervention/comparator/outcome prompts,
evidence spans, article text and article-level splits. Its [paper](https://aclanthology.org/2020.bionlp-1.13/)
describes human evidence inference rather than synthetic templating. The
downloaded organizer archive has SHA-256
`6abe0d4ec0d331834981c0171c3c79d47515761867f82f1dc6066e43863a1586`.
The aggregate screen read the official archive on an authorized private
CPU environment; it wrote no model prediction and opened no protected key.

| Archive aggregate | Observation |
| --- | ---: |
| Annotation / prompt rows | 24,686 / 12,865 |
| Official article split IDs | TRAIN 3,562; validation 443; test 449; disjoint |
| TRAIN prompts / distinct TRAIN articles with text | 10,326 / 2,674 |
| TRAIN prompt groups with one accepted canonical label | 10,071 |
| TRAIN groups with accepted abstract evidence and verified reasoning | 4,944 |
| Above groups with all selected evidence offsets present | 4,845 |
| Distinct articles per eligible label: decrease / no significant difference / increase | 832 / 1,075 / 910 |
| Publisher README caveat IDs / those in TRAIN | 106 / 95 |
| Full-article whitespace words, median / p90 / p99 | 3,766 / 5,990 / 8,374 |

The three canonical TRAIN labels among the 10,071 one-label groups are
2,463 decrease, 4,563 no significant difference and 3,045 increase. Eight
other TRAIN prompt groups have conflicting accepted canonical labels; 71
have no accepted canonical label. A publisher spelling variant, “significantly
increase”, occurs in 549 TRAIN groups and is **excluded** unless its mapping
is separately verified. The official README also warns of wrong,
questionable and malformed prompts and missing evidence offsets. The 95
flagged TRAIN IDs are to be excluded by article group, not silently corrected.

A deliberately simple prompt-only artifact probe used *only* the outcome,
intervention and comparator words, lowercased alphanumeric unigrams, a
multinomial Naive Bayes classifier, and a fixed article-group split by
SHA-256 modulo five. On 2,082 held-article TRAIN prompt groups, majority-class
accuracy was **41.98%**, and prompt-only accuracy **42.80%**. This is an
aggregate TRAIN-source shortcut check, not a Decision model baseline or proof
that the full source is artifact-free. The selected 384-case treatment subset
must repeat the same check after filtering and balancing.

The source's `no significant difference` label describes a reported
statistical result. It is **not** lack of evidence. A faithful System One
native Score request should ask for the reported intervention-versus-
comparator effect on the stated outcome, with ordered levels `decrease / no
significant difference / increase`. Do not relabel the middle level as
`unknown`, invent a free-text chat target, or expose annotated evidence spans
as answer hints. These labels are directional *reporting conclusions*, not
ordered measurements of effect magnitude: `no significant difference` does
not prove zero effect, and whether an increase is beneficial depends on the
outcome. Do not use an ordinal-distance loss, interpret expected level as a
clinical effect estimate, or write `better / same / worse` without a separate
outcome-direction annotation. An independent rubric review must approve this
restricted three-level Score interpretation; otherwise this source is only a
categorical Choice diagnostic, and the proposed Score training arm stops.
The abstract-only evidence route is bounded, not evidence of long-document
ability. Full-article input would need a separate length admission and trial.

The organizer repository has a root MIT license, but the trial articles
originate in PMC and their text rights may differ. Do not infer that all
article text is MIT-redistributable. Check the rights of each selected article
and keep raw text private where its conditions require. The archive's
annotation license and the individual article terms must be recorded
separately.

## Conditional matched intervention, not yet authorized to train

Use exactly the pinned official Posttrained direct source above, its same
dynamic-option head initialization, native Score adapter, BF16 backbone /
FP32 head, fixed seed, LoRA/head optimizer, CE + 0.5 Brier, 8,192-token
limit, **466-update** schedule and SELECT700/CAL700 from the completed arm.
Retain all original human-label and all original **516 Score TRAIN** rows.
Choose exactly **384 additional** publisher-TRAIN cases from **384 distinct
articles**, balanced 128 per canonical effect label, after the gates below.
Replace 384 nonhuman, non-Score old TRAIN slots chosen by a frozen gold-free
hash. Preserve 7,455 row slots, deterministic step order, the old optimizer
horizon, and whole-run native token exposure within **±1% of 4,194,465**.
Match old and new lengths without filler or truncation; if this cannot be
done while retaining the human and old Score rows, do not launch the arm.

For each replacement, require one accepted, internally consistent label;
verified reasoning; evidence certified in the abstract with usable offsets;
no publisher-caveat ID; and the full, correctly extracted abstract plus the
ICO prompt. Enforce one case per article and freeze the mapping from source
label to the three-level System One rubric before inference. Article-level
splitting controls sibling prompts; duplicate and near-identical articles
must be grouped. The selection and train-set overlap audit must consume
gold-free protected prompts only, not their answer files.

Before any optimizer step:

1. Pin candidate article licenses, archive/derived-file hashes, annotations,
   abstract extraction, group IDs, quarantine decisions and native Qwen
   tokenizer totals. Check exact and near overlap against TRAIN/SELECT/CAL,
   typed DEV/FINAL, CSS pilot/15-task prompts and public231. Source-family
   overlap, including clinical examples already embedded in the backbone's
   unknown pretraining, remains a stated limit.
2. Verify the source revision/weight/tokenizer and trainer inputs. Repeat the
   official-Posttrained zero-step **SELECT700 twice**, with the historical
   same-batch ordering: 700 valid each, zero categorical changes and
   p99/max probability drift at most **.005/.02** between fresh starts.
   Compare source identity and encoded input hashes with the archived
   official arm. Do not transplant the own-Sol BF16 gate, which failed for a
   different merged source/runtime.
3. On a throwaway source initialization, run one finite optimizer update,
   save/reload, and check the same 32 SELECT rows at the same batch shape:
   zero category changes and p99/max probability drift at most **.005/.02**.
   This smoke checkpoint is never a full-arm start. Stop for nonfinite loss,
   missing rights/roster, overlength, output invalidity or numerical drift.

Only then freeze the exact 384 case IDs, 384 replacement IDs, per-step token
counts, private data/code/image hashes, source package, controls and stop
rules; run one complete 466-step arm. Keep the original SELECT family-macro,
normalized-Brier, earliest-step BEST selector, then fit temperature on CAL
only after the single BEST is fixed. Bound full training plus selection to
**2.5 GPU-hours** on one reserved device. The archived official-Posttrained
run is the no-new-source control; do not rerun it. If prechecks fail, record
why and stop without choosing another source, seed or checkpoint.

**Frozen development stop gate proposed for this treatment:** after native
package reload, the same typed DEV1600 + CSS pilot1430 proxy must reach
**45.15** (at least +2.0 over own Sol1), CSS task-median macro-F1 at least
**.340**, typed Score at least **200/400**, and 100% in-budget valid answers.
These are a combined-gain and severe-collapse guard, not a demand that every
Choice/Noul/task value increase. Report all per-type/task scores, three-level
Score recall/confusion, Brier/ECE, invalids, and paired uncertainty, including
regressions. If the gate fails, no CAL rescue, v3/public231, HF upload or
alternate checkpoint follows. If it passes, obtain one separately frozen,
source-disjoint evidence-Score diagnostic and a fresh human-task diagnostic
before any post-key formal comparison. Within-archive validation is useful
for source quality but cannot establish cross-source transfer.

## Present blocker and next exact CPU task

The archive identity, class supply, article grouping and crude artifact
probe pass a *feasibility screen*; **zero new cases have been admitted**. The
next task is a reproducible private TRAIN-only extractor and admission
receipt that (a) excludes all publisher caveats and disputed labels, (b)
checks abstract extraction against evidence offsets and obtains the restricted
Score-rubric review, (c) resolves per-article text rights, (d) matches 384
native row lengths and the full token budget, (e) quarantines exact/near
protected matches, and (f) reruns the prompt-only shortcut probe on the
selected 384. Only a passed, signed receipt can promote
the conditional arm to a GPU preregistration. The previous synthetic Score
v8.4/v8.5 candidate, HelpSteer2 correctness projection, short SNLI/OCNLI
screen and SICK similarity source remain nonadmitted alternatives; none is a
shortcut around this gate.
