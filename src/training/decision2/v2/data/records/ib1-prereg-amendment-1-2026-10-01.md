# IB1 preregistration — amendment 1 (2026-10-01)

Committed after the candidate build (run `b2`, builder `17359493`) and **before any audit, scan or review** of the
candidates. Prereg `ib1-prereg-2026-10-01.md` (`1f6823fae`) is unchanged except where this amendment says so.

## A. `expqa` is empty under §2 and leaves IB1

The §2 rule takes "no" rows from claims with support `Missing` and at least one non-empty evidence text. The build
report shows that every `Missing` claim in the pinned file has no evidence text (2,216 of 2,216), so the family has no
"no" rows and the balancing step yields 0 rows. Inspecting the build report and the file schema (no labels were
tabulated against content) also showed that the evidence field of many judged claims holds only a citation URL: the
experts judged support on the linked page, which the row does not contain, so a reader of the row could not check the
label. No alternative rule (for example `Partial` / `Incomplete` as "no") is adopted. **IB1 has 16 families**; the
faithfulness target family is served by `sumedit` only. ExpertQA stays in the licence registry as read, not used.

## B. G5 for rotated families is evaluated per option count

`maud` questions have 2–6 answer options, so a family-wide position share is not meaningful. The gold-position band
(each position share within 1/k ± 0.05) is checked, and over-represented positions are trimmed in `finalize`,
separately for each option count k. A stratum with fewer than 20·k rows is reported, not gated (the band is narrower
than sampling noise there). The other rotated families have a single k. G5 gates TRAIN and DEV, as in HR2.

## C. Run identifiers

Run `b1` stopped at the build stage on a reader bug (one SNIPS file encodes emoji as UTF-8 surrogate pairs; nothing
was written). The fix is `17359493`; run `b2` holds the candidates (TRAIN 63,359 / DEV 6,655 before audits). Every
later stage runs on `b2`.
