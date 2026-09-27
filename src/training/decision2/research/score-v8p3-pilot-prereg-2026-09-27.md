# Score v8.3: source-disjoint decision dossier pilot

**Status: prospective design, before generator code or candidate rows.** The
v8.2 pilot remains HOLD. v8.3 starts from a fresh seed and three new semantic
mechanisms; it does not edit, filter, or relabel v8.2 rows. This is a data
quality pilot only. It authorizes no GPU, training, model selection, CAL, or
formal benchmark prediction.

## Fixed candidate and oracle

Generate exactly four TRAIN and two SELECT independent triplet groups for each
of the three mechanisms below: 12 TRAIN groups/36 rows and six SELECT
groups/18 rows, **54 rows total**. Each group has one ordinal Score level 0,
1, and 2, and a unique case, target, organization, and document setting. Use
one private 32-byte seed and HMAC separation by
`decision2-score-v8.3-dossiers/1`, role, mechanism, and group. Publish only
the seed digest. A failed audit or review holds the full fixed candidate; do
not reroll a seed, discard unfavorable groups, or relax thresholds.

1. **Dependency graph readiness.** A request references a small dependency
   graph, a separate status board, and an unrelated neighboring task. A
   failed required ancestor blocks the request (0); none failed but at least
   one required ancestor is queued leaves it pending (1); all required
   ancestors complete makes it ready (2). The structured graph and statuses
   determine the oracle. Required and decoy task positions and the deficient
   ancestor vary within each triplet. Documents resemble different work
   handoff genres rather than a single repeated ledger.
2. **Timed connection feasibility.** A service bulletin gives an arrival
   interval, a station guide gives transfer time, and a boarding notice gives
   the departure and cutoff. Even the earliest feasible transfer missing
   cutoff is impossible (0); an interval straddling cutoff is uncertain (1);
   the latest feasible transfer meeting cutoff is reliable (2). Generate
   time facts first and compute the oracle in minutes, then render at least
   three varied transport document formats. Nearby services are distractors.
3. **Order fulfillment under stock uncertainty.** A pick request identifies
   target SKU and quantity; a stock and reservation report gives confirmed
   usable units; a carrier note distinguishes confirmed from tentative
   inbound units. If even the optimistic quantity is short, fulfillment is
   impossible (0); if only tentative inbound closes the gap, it is pending
   (1); if confirmed units suffice, it is ready (2). Source facts determine
   the oracle before rendering. Other SKU lines must not reveal the target
   verdict.

For every row, compare the structured oracle with a separate parser of the
rendered documents. Preserve the target's three documents or document views
with unambiguous attribution, unit/time meanings, and explicit scope. No
question may depend on identifying an arbitrary option order or record ID.
Use varied organizations, section order, notation, and prose within each
mechanism. Do not reuse the v8.1/v8.2 signed-amendment, outage, simple limit,
or scoped-exception skeletons. This synthetic pilot cannot establish
real-world long-context transfer.

## Frozen audit and review gates

- Confirm the exact 54 rows, 18 complete groups, 18 labels per level, valid
  native Score row schema, fresh role-specific IDs, and disjoint source cases.
  One field alone must not decode all three outcomes in a group; audit target
  and decoy placement, document ordering, answer cues, and lengths.
- Compare TRAIN and SELECT with each other; parent rights-clean
  TRAIN/SELECT/CAL; v7p, v8, v8.1, and v8.2 TRAIN and gold-free SELECT; and
  available gold-free protected DEV, formal, transfer, authored, and public
  prompt inventories. Open no protected labels. Record source file hashes and
  comparisons. Reject exact identity of case, ID, group, state, or full
  prompt. A bounded near-text match flagged by the existing overlap tool is
  a whole-pilot HOLD. Additionally normalize IDs/numbers and flag
  cross-group first-variant SequenceMatcher similarity at or above 0.88 or
  token 5-gram Jaccard at or above 0.60; within-group variants are intentional
  pairs and excluded only from that latter comparison. Bounded matching does
  not prove semantic independence.
- Freeze separate labeled source rows, answer keys, and **gold-free blind
  packets** before review. A different reviewer must solve all 54 rendered
  items and assess every complete group for answerability, ambiguity, ordinal
  coherence, document realism, evidence necessity, and shortcut cues. The
  author may run programmatic audits but may not self-review the blind text or
  inspect model outputs. Seal the independent answers and findings before
  opening keys. Ambiguous, unsupported, or unrealistically templated groups
  hold the entire candidate.

Even a full PASS only permits a larger, source-disjoint data and matched
control experiment preregistration. It does not admit any existing 27B
checkpoint or justify a release claim.
