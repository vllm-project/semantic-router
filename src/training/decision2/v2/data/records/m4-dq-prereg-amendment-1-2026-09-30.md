# Amendment 1 to the PN1 / HS1 data-quality prereg — PN1-r2: construction-level fix, fresh certification sample (2026-09-30)

Committed at 2026-09-30 ~01:55 UTC+8. It comes after the §3 review of [the prereg](m4-dq-prereg-2026-09-30.md) and
the §3.6 fix run, and before any PN1-r2 row is built or sampled. §1, §2, §4 and §5 are unchanged.

## A. Why

**The §3 review failed, and so did the §3.6 fix.**

| Stage | Result |
| --- | --- |
| §3 review | 12 gold errors in 224 rows: 5.36%, exact [2.80, 9.17]. P1 fails; P2 (weighted 1.79%) and P3 pass |
| §3.6 fix | F2 found no failing cell. F3 was applied: 9 errors in 186 surviving reviewed rows, 4.84%, exact upper 8.99%. Still FAIL |

Under §3.6, PN1 `@5ad36287` stays not release-safe and the F3 output is not published.

**Why another attempt.** The assignment asks for a fix and a republish when PN1 fails. The error pattern points at
two constructions, not at the judge margin. The 12 errors have a median judge P(yes) of .096, and only 3 are above
.20.

- **All 12 are false yes.** Each has gold no, and both blind reviewers said yes. Gold-yes rows had 0 errors in 112.
- **`pn-near` in es, fr, ar, ru and ko:** 8 errors in the 40 reviewed rows (20%; exact [9.1%, 35.6%]). In ja, zh and
  de the same construction had 0 in 24.
  - The rule "no shared translation within two link hops" admits paraphrases that Tatoeba simply never linked: an
    omitted "already", tú versus usted, spelling variants of one name, and pairs that differ only in aspect.
- **Russian `pn-name` role swaps:** 2 of the 8 reviewed ru swap-no rows. When Том and the indeclinable Мэри are
  exchanged in place, Том stays nominative, so the case marking keeps the roles and the meaning. The construction
  does not work for Russian.
- **`pn-twin`:** 2 errors in 54 (one ar, one ko). No construction pattern; kept.

**One more attempt.** It uses a construction-level rule, fixed here, and is certified on a fresh sample that shares
no row with the §3 sample. The thresholds are unchanged. There is no further iteration.

## B. PN1-r2 rows

Applied to `pn1.train.jsonl` (`f6f6a531…`) by `v2.data.dq.blind_review rebuild-pn1r2`:

- **D1:** drop every `pn-near` row in es, fr, ar, ru and ko.
- **D2:** drop every Russian `pn-name` row.
- **F1:** drop the 12 rows the §3 review judged to be gold errors.
- **Balance repair,** as in §3.6: exactly 50/50 per language × group cell. Surplus rows of the larger label go in
  `sha256("dq-pn1-balance-v1:" + id)` order; reviewed rows go last.
- **No margin filter.**

Consequences, from the published counts: the natural cells of es, fr, ar, ru and ko lose all their no rows, so the
balance repair removes their hop rows as well.

- es and fr leave TRAIN.
- ar, ru and ko keep swap rows only (ru: twins only).
- About 4,364 rows are expected. The exact counts go in the record.

The dev slice is not changed, since it is never trained on. Its rows use the same constructions, so its es / fr /
ar / ru / ko near rows and its Russian name swaps may carry wrong "no" labels. This is disclosed for the readout.

## C. Certification of PN1-r2

- **Frame:** PN1-r2 TRAIN minus every row reviewed in §3.
- **Sample:** 12 rows per language × group × gold cell, or fewer if the frame has fewer (de natural has 6 per label
  left). Rows are taken in `sha256("dq-pn1-r2-review-v1:" + id)` order, at most one per group. Review ids are
  `qNNN` in `sha256("dq-pn1-r2-packet-v1:" + id)` order. About 204 rows.
- **Review:** the same packets and instructions ([`dq/reviewer-pn1.md`](dq/reviewer-pn1.md)).
  - Two fresh blind subagents, R1 and R2. R2's order is `sha256("dq-pn1-r2-r2-order-v1:" + rid)`.
  - A fresh R3 on the splits.
  - The same majority rule for gold errors.
- **Thresholds, unchanged:**
  - **P1:** error ≤ 5.0% and exact upper bound ≤ 8.0%. With 204 rows that means at most 8 errors.
  - **P2:** population-weighted error ≤ 5.0%. The weights use PN1-r2 TRAIN cell counts.
  - **P3:** a language × group cell fails if its exact lower bound exceeds 5%, which is the rationale stated in §3.5.
    In §3's 16-row cells that was "≥ 4 of 16". Here it means ≥ 5 of 24, and ≥ 3 of the 12 de natural rows. The code
    now applies this bound directly, and it gives the same verdicts on the §3 cells.
- **PASS:**
  - canonical PN1-r2 TRAIN, `v2.data.freeze`, and the A4 overlap self-check (reported);
  - `hf_headroom.sh`, then upload `m4/pn1/arms/pn1.train.jsonl`, its manifest, the self-check and the review
    receipts as a **new revision** of the private dataset;
  - a read-back SHA-256 check;
  - new hashes in the record and gist 02.
- **FAIL:** PN1 stays not release-safe, and nothing more is tried in this job.
