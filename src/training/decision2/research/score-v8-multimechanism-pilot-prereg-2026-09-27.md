# Score v8: prospective multimechanism quality pilot

**Status before authoring: design only.** This is a replacement candidate for
the v7p corpus, whose independent review found a common abstract A/B template
in every group. No v7p row, label, renderer, or failure-based item selection
enters v8. The v8 pilot is a data-quality experiment, not a training arm or a
release evaluation.

## Frozen pilot design

Build exactly five ordinal mechanisms, three independent TRAIN groups and two
independent SELECT groups per mechanism. Every group contains one related item
for each Score level 0, 1, and 2. Thus the pilot has 15 TRAIN groups / 45 rows
and 10 SELECT groups / 30 rows. It is deliberately too small for a training
claim. Use a newly generated 32-byte private seed with HMAC-separated role,
mechanism, group, and level streams; publish only its SHA-256. Source IDs,
document IDs, and organizations differ across roles. SELECT is never used as
training input. English is the only pilot language: without independent native
language review, a translated or machine-written Chinese layer would not
support a multilingual claim.

| Mechanism | Decision rule | Presentation |
| --- | --- | --- |
| Dated state updates | Apply the latest signed event for the requested record at or before the review date; later, unsigned, and different-record events do not apply | Chronological transaction ledger and request note |
| Numeric release thresholds | Apply two inclusive measurement limits and a separate calibration requirement; failed, missing, and confirmed measurements have different outcomes | Specification sheet plus instrument log |
| Evidence sufficiency | Distinguish a contradicted claim, an unsupported claim, and a claim corroborated by independent documentary evidence | Claim, source notes, and provenance index |
| Scoped exceptions | Resolve a default restriction with a signed, currently effective exception for the exact site and activity; pending exceptions do not waive it | Policy extract and exception register |
| Long-document retrieval | Locate the controlling clause and later amendment for a named record amid unrelated paragraphs and similar records | Multi-section narrative memo, at least 1,500 characters |

Each mechanism uses its own structured facts, oracle, instructions, options,
and rendering function. The structured oracle is computed before the rendered
prompt. A separate parser of the rendered text must recover the same answer
for every pilot row. The three related levels retain the same named case and
review context; the operative evidence changes. Decoy records, document
ordering, and option text must not encode the answer. Level 0 may override an
otherwise incomplete requirement; level 1 must require actual missing or
pending evidence, not a generic default. No answer-level suffix appears in
the blind packet ID.

## Quality and separation gates

1. Validate all 75 rows against the Decision training-row contract and both
   oracles. Require exactly 25 complete triplets, five mechanism families,
   25 rows per level across both roles, and 0/1/2 balance within each role.
   Check group, source-record, and input hashes for disjointness. Record the
   number of distinct rule and document families rather than counting domain
   captions as new mechanisms.
2. Audit full prompts and contexts against the pinned parent TRAIN, SELECT,
   CAL, v7p, and every available gold-free protected DEV, formal, authored,
   transfer, and public roster. Use normalized exact and bounded near-text
   matching. Any exact or flagged near match is a group-level HOLD pending
   review. Only gold-free protected prompts may enter the audit. A similarity
   pass cannot prove semantic independence.
3. Audit answer-position, document-length, option wording, lexical cues,
   target-record placement, and short-form family classifiers. This small
   pilot can expose a shortcut but cannot certify its absence at scale.
4. Seal opaque, gold-free TRAIN and SELECT packets separately from answer
   keys. A different reviewer must solve all 75 rendered items, check each
   complete triplet for realism, ambiguity, and source necessity, and seal
   findings before key comparison. The author and mechanical oracle do not
   count as independent review. If any mechanism fails, preserve the finding
   and design a new version rather than replacing only unfavorable examples.

Only after the pilot passes review may a larger v8 data plan be preregistered.
That later plan needs source-disjoint scale-up, length/token and rights audits,
matched controls, fixed optimizer budget, zero-step parity, independent SELECT
gate, and no formal-set tuning. This preregistration authorizes **no GPU use,
training, CAL, or formal evaluation**.
