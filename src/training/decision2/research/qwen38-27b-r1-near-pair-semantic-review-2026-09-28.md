# Locked 27B r1: independent input-only near-pair review

**Disposition: r1 remains HOLD; no GPU or model evaluation.** This review does
not edit its 2,560-row schedule or reinterpret the registered zero-near-state
gate. The [sampling plan](qwen38-27b-r1-near-pair-semantic-review-plan-2026-09-28.md)
was signed as `21115ed28` before the private pair receipt was opened. Only the
pinned TRAIN inputs, gold-free SELECT/CAL projections, and input-only
source/group metadata were inspected. No TRAIN, SELECT or CAL target, teacher
output, model prediction or formal answer key was inspected.

## Evidence identity and scope

The source receipt matched its published SHA-256
`3f3da6e6ba616565a5ceb0cc7a4552d7c5aa563577598354b7aa29b7e88ae49d`.
The core gold-free inventory manifest matched
`26bbaf82eb1c30c0f2093c70d27718fab6731ea8c80e9a451547b03bd30897e1`.
The TRAIN partition matched
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`.
The ignored mode-0600 input-only sample packet and per-pair judgment receipt
have SHA-256 `1d560e70e53d476c52ab01c34d5d35c40a7b1b423d75d0d00eb216eff8822037`
and `c07780807c33e99da5cb94036b0b51cd595a4d99bed91705ee119627e7a55e60`.
No pair IDs or row text are reproduced here.

| Input-only finding | Result |
| --- | ---: |
| Registered near-state pairs reviewed | 2/2: one SELECT, one CAL |
| Fixed complete-input sample | 36/606: 12 hash-drawn plus six highest state-similarity pairs in each split |
| Sample with distinct raw state and distinct source/group ID | 38/38 including state pairs |
| Sample with identical task family and instruction | 38/38 |
| Sample with identical options | 37/38; the code-transformation state pair has different candidate codes |
| Sampled median pairs with distinct numerical readings multiset | 36/36 |
| Sampled median pairs classed common template/different instance | 31/36 |
| Sampled median pairs left unresolved for substantial shared numerical evidence | 5/36 |

Both near-state pairs have distinct input-specific evidence: one changes the
code operated on, the other changes instrument readings while retaining the
same grading rule. They are **template collisions**, not established
same-record duplicates. They still fail r1's registered **zero** near-state
condition; the semantic judgment is not a waiver.

The fixed 36-pair complete-input sample all concerns the same five-reading
median-to-grade mechanism. Its five unresolved pairs share either at least
three readings or both the same four cutoffs and at least two readings. One
pair shares four of five readings. The other 31 have different deciding
numbers even though scaffold, instruction and grade options match. These
input-only judgments do not establish independent original corpus records or
measure a population duplicate rate.

An explicitly **exploratory follow-up**, separate from the fixed sample,
checked all 606 pair metadata and structural numerical facts, then read the
seven non-median outliers. Of the 606, **599** are generated median-grade
pairs: every raw state and source/group ID differs, and none has both the same
five-reading multiset and identical cutoffs. Yet **8/599** share at least
three readings; **5/599** share at least two readings and all four cutoffs
(the sets overlap). The other seven comprise two programmatic situations that
retain the same answer-determining relationship after incidental names or
item identifiers change, plus five short human-comment emotion cases with
distinct text and group IDs. The two programmatic cases are material
**scenario isomorphisms**, despite their different group IDs. The human cases
show generic task/option scaffolds and sometimes similar sentiment, but this
evidence alone does not prove a duplicate source record.

## Capacity and prospective decision

The locked r1 candidate contains all **419** Score rows still eligible after
its earlier 61 whole-group near-state exclusions. Excluding only the two new
state-near groups would remove two Score rows and leave 417. A conservative
prospective rule that additionally excludes entire groups touching the eight
high-overlap median pairs, the five same-cutoff/two-reading pairs, and the two
isomorphic programmatic cases has a union of **14 selected groups** and
removes **22 Score rows**; at most **397** of the currently eligible Score rows
remain. Excluding groups touched by *every* complete-input near pair would
remove 60 selected Score rows, leaving at most **359**. Thus no new
rights-clean-v2-only, same-budget candidate can retain a **Score≥400** floor
under either conservative exclusion policy. This is an input-only upper bound,
not an optimizer or teacher-mask result.

Do not construct a nominal r2 by removing the two r1 failures and silently
allowing the other near pairs. A distinct r2 would need a prospectively
versioned data source or explicitly lower Score floor, a fixed group-atomic
selector without post-result seed search, rechecked native token exposure and
teacher-mask minima, and full-source provenance and overlap audit. In
particular, adding at least three independently cleared Score rows is only
the raw-count minimum under the high-overlap exclusion, not a guarantee that
whole-group, source, teacher or token gates can pass. Lowering the floor after
this observation would be a disclosed new experiment with weaker Score
exposure. No such r2 is admitted by this review.

The protected native evaluation projections bind complete input hashes but
omit original upstream record IDs. Distinct local source/group IDs, a
bounded lexical scan and this sample cannot close that provenance gap or
prove absence of paraphrases. Formal typed/CSS/public roles had zero exact
or bounded-near hits under the prior pinned algorithm, which is useful
evidence with that same limit. The same-family SELECT/CAL collisions also
mean those partitions test within-template behavior; they do not themselves
prove cross-source transfer. Keep r1 HOLD and GPU-hours at zero.
