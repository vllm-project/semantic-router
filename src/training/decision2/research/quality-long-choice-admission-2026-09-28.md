# QuALITY long Choice: source and gold-free admission follow-up

**Decision: TRAIN HOLD, 0 admitted rows, 0 GPU-hours.** This follow-up to the
[native-length screen](quality-long-choice-source-screen-2026-09-28.md) did not
train or evaluate a model, inspect protected answers, select a checkpoint, or
repeat the tokenizer length audit. It examines whether pinned QuALITY TRAIN
could supply genuinely long Choice examples for a future matched 0.6B arm.
The current private 0.6B package and its v3/public231 results do not change.

## Frozen inputs and method

The [official QuALITY repository](https://github.com/nyu-mll/quality) was
read at commit `f84977c40dbfef70c9cab48037b7becfc8e45f73` with corrected
`v1.0.1.htmlstripped` TRAIN/DEV/TEST files, SHA-256
`4011e9952d5395beb8ff7637b963481a400630c1bbe2f40dc0d83f5d59f926ed`,
`99852d874994078e4b4112b71ceca4dd35aa3a24ff6d3a35c051be25295b4fef`,
and `ca103a953741c56888124a14958460b07941ee40a844914d39e851f1c3099897`.
No official DEV/TEST target was loaded. The strict input-only protected
manifest SHA-256 was
`26bbaf82eb1c30c0f2093c70d27718fab6731ea8c80e9a451547b03bd30897e1`.
It contains eight independently hash-checked projected roles: rights-clean v2
TRAIN/SELECT/CAL, typed DEV/FINAL, CSS pilot/15-task, and JevBench public231.
Five native sealed roles and all three rights-clean partitions are present.
Twenty-seven optional historical roles were deliberately excluded from this
strict projection and are **not** claimed as cleared.

The local aggregate auditor is `training/data/audit_quality_admission.py`,
SHA-256 `7056c064852b7d7f83ee7b061db9083a727ff6c1ce1148763125646f397d9db3`.
Its private aggregate receipt SHA-256 is
`4beb44483be07af5a2e955f417d876c6c69a95f0993985c8e1997e6b8457cf48`.
The private review artifact receipt was separately sealed at
`f272e2c9f2e0effbce944f0df47ec355e0dbcff2d6aba84d140b3eb04b4fd63a`;
the later aggregate-only run reproduced its relevant counts and added a
within-TRAIN article duplicate screen.
The script enforces the source Git revision and file bytes, projected role
bytes/counts, and input-only schema before comparing normalized full article
states, article excerpts in complete inputs, and question text. Five-word
shingle Jaccard ≥0.60 or shorter-text containment ≥0.80 with at least 12 shared
shingles flags an entire source article group. Three-word question shingle
Jaccard ≥0.90 or exact normalized question text also flags the group. These
lexical gates cannot establish semantic independence from paraphrases or
pretraining exposure.

## Source independence and rights

The official TRAIN split has **150 independent articles / 2,523 questions**.
One TRAIN article title recurs in official DEV with different authors; its
entire 16-question TRAIN group remains quarantined. Full-article five-gram
near search found no additional TRAIN versus DEV/TEST match at Jaccard ≥0.80,
and no distinct within-TRAIN article near match or same-title group.
This leaves 149 article groups / 2,507 questions before native length, rights,
or protected-panel checks. The previous exact native-length audit limits any
untruncated whole-article arm to **144 groups / 2,424 questions** after the
title hold. That is an upper bound, not an admitted roster.

For the 149 source-disjoint candidate articles, the source-level license
metadata classifies **117 Project Gutenberg**, **22 OANC-pointer**, and
**10 article-level CC BY 4.0** groups. Title, author and year fields are
present for all; the 22 OANC-pointer rows lack a source URL. The
[QuALITY project](https://nyu-mll.github.io/quality/) distributes its dataset
under CC BY 4.0, while the underlying article terms remain separate.
[Project Gutenberg](https://www.gutenberg.org/policy/license.html) and
[OANC](https://anc.org/OANC/license.txt) have their own conditions. A private
149-article attribution and rights-review ledger was sealed at SHA-256
`0b538d9960b8f064aed80272715352dcc52ccdedb2360155b0cd02492ced23ac`.
It contains no article body or question and assigns each article a pending
work-level review. Source metadata alone does not clear the 149 underlying
works for a final data recipe or redistribution.

## Protected overlap and shortcut diagnostic

The **eight strict core roles** contained 20,263 row comparisons. Within the
149 source-disjoint TRAIN article groups, normalized exact full-article state,
five-gram near/excerpt state and complete-input comparisons, exact question
text, and three-gram near question text each flagged **zero** protected rows
and **zero** source groups. This result covers the independently projected
eight roles only. Source-level and semantic overlap, missing optional roles,
and upstream pretraining exposure remain possible.

One question from each of 24 deterministically selected independent TRAIN
articles was checked for two simple answer cues. The correct option occurred
verbatim in the question **0/24** times; the correct option was uniquely
longest **5/24** times. These two cues do not prove that the article is needed.
A separate 48-row answer-blind review packet, with and without article, and
separate key were sealed privately. Packet/key SHA-256 values are
`37ae7833977d53b617f876952aedd4800e108d50fabaa6a6afe46218f09aeac8`
and `395fd5ff136ed22548205f0b0264770ba5693fd5d70fbd49151346a9155216f5`.
An independent answer-blind reviewer assessed all **24 paired questions / 48
views** under a criterion frozen before opening the packet: identifiable
article evidence, inability to answer the article-removed view from shortcuts,
unambiguous options, and legible document. Only **7/24 pairs** passed all
four conditions; **17/24** were rejected. This is a small, conservatively
selected diagnostic, not a full-corpus quality estimate or a comparison to
the publisher answer key. The reviewer did not open the key or rights ledger,
run a model, or use GPU. Its private item-level receipt SHA-256 is
`927d89f63c9fdc2274a585183932a49abb7d44f97b036f4745d0b1527deaccf2`;
the public-safe private summary SHA-256 is
`d5be46ca721453dbefcb39fa02b3fa297bd210ba6903bc83ff5540fad5fdc23f`.
The [authors' question-only baseline](https://nyu-mll.github.io/quality/)
reinforces the need to test shortcuts rather than assuming long text creates
long-context reasoning.

## Next decision

The blind review keeps the **entire proposed QuALITY training arm on HOLD**.
Before reconsidering it, quarantine or repair ambiguous and shortcut-prone
questions, expand independent paired review beyond 24 articles, and complete
work-level rights and attribution review. Then recheck any resulting
whole-article roster against optional historical roles and a separate
semantic-duplicate review. Only a source-disjoint, untruncated,
evidence-dependent roster can enter a new signed matched-token training plan.
Its initial gate must check both Choice recovery and Score retention on the
already fixed development panels. No optimizer run is authorized by this
screen alone.
