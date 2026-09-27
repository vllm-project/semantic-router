# Score v8.3: CPU pilot and independent review handoff

**Status: PENDING_INDEPENDENT_BLIND_REVIEW.** The prospective
[v8.3 preregistration](score-v8p3-pilot-prereg-2026-09-27.md) was signed
before generator code and candidate creation. The frozen generator and audit
code are at `37ed617557125334132dbd17dabaaa7335de4827`. The exact code
mirror matched the local SHA-256 of the generator, audit, and tests. No GPU,
optimizer, model output, calibration fitting, or formal benchmark label was
used. CAL text was read solely for the preregistered overlap comparison.

The fresh private 32-byte seed has SHA-256
`ef2ce4457791c717ef2f6b1b6d845d0796bbaa8cb2b803e1deaf83b7f54209f2`.
The one immutable candidate contains three mechanisms, 12 TRAIN triplets
(36 rows), and six SELECT triplets (18 rows). Each triplet has one Score level
0, 1, and 2. The source cases are disjoint across roles. State lengths span
377–484 characters; these short synthetic dossiers do not test long-context
transfer.

| Artifact | SHA-256 |
| --- | --- |
| Candidate manifest | `a138ed2c564c0dd415cdd46aa526998f0bf6515087e2a8ac8d9d20facc7565be` |
| Frozen gold-free audit | `4c5447d09e5220df49d966c2dc1a33e2d1afb6041cb76e65517b24474e3da980` |
| TRAIN blind packet | `18ff1815d1c7911677c893dca80264b428efa6c175b925b40503c15b8646135b` |
| SELECT blind packet | `97e9394b61acfdda365749c11c19802ba145800f715ea4b4b408e62c48f4b5e3` |

Structured and rendered-text oracles agree on all 54 rows. The programmatic
audit checked the parent TRAIN/SELECT/CAL, earlier Score pilots through v8.2,
and 28 gold-free protected prompt inventories. Across 79 candidate-to-source
comparisons it found zero flagged exact or bounded near matches. The separate
normalized within-candidate scan found zero cross-group near pairs and zero
single-field projections that uniquely decode all three levels. These checks
are necessary but do not establish semantic independence or document realism.

The reviewer should receive **only** the two gold-free blind packets from the
separate private handoff directory, the signed preregistration, and the audit
summary. The answer keys and labeled source rows are kept in a different
private directory. A reviewer other than the generator author must solve all
54 rendered items, assess all 18 complete triplets, and seal answers and
quality judgments before any key comparison. Specifically inspect answer
support, ordinal coherence, need for the graph/time/stock evidence, decoy and
document-order shortcuts, source realism, and cross-group semantic reuse.
Any unsupported or materially templated group keeps the whole pilot on HOLD;
the author will not self-review, substitute groups, or start training.
