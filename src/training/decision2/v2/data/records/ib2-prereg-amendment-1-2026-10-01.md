# IB2 preregistration — amendment 1: Glaive slots removed, constructed `fc_rel` negatives (2026-10-01)

Committed **after the first candidate build (run `c1`, builder `12e8d8c9`) and before any audit, sample or review**.
The prereg (`fff14438e`) is unchanged except where this amendment says so. Only build counts were read; no row was
rendered, scanned or shown to a reviewer.

## Why

Glaive function-calling v2 is far more redundant than its size suggests: 78,011 conversations with functions carry only
11,193 distinct normalized first requests (the five most frequent requests account for 14,055 conversations), and
first-turn refusals concentrate on a few requests (booking a flight, ordering a pizza). Under the prereg's slot rule
(each request in exactly one of the four `fc_*` families) and the cap of two rows per group, the tool-call block came
out at 4,751 candidate rows (`fc_rel` 170 after yes / no balancing, `fc_sel` 2,124, `fc_args` 1,565, `fc_ready` 892),
against the prereg's expectation of the largest block in IB2. The run `c1` candidates are superseded and not audited.

## A. Slots removed

Every `fc_*` family now considers every parsed conversation. A request's group id (`glaive:` + normalized U1) is the
same in all four families, so the DEV slice and every G0 / G2 group drop act on a request in all of them at once; the
cap of **two rows per group per family** is unchanged. A request can now appear in up to four families (different
questions about it).

## B. Constructed `fc_rel` negatives

The prereg's natural rows stay (yes: A1 calls a listed function; no: A1 refuses). In addition, every call-first
conversation contributes one **constructed no row**: the same request U1 with the function list of another conversation
(a call-first or refusal conversation, walked in hash order `ib2-fcrel-lists-v1` from a start fixed by
`hash("ib2-fcrel-start-v1:" + key)`, at most 200 steps) in which **every** function is unrelated to the called function
f and to the request:

- its name differs from f's;
- its name shares no content token with f's name or with the words of U1 (tokens as in §2.1, generic tokens removed);
- its description shares no content word (≥ 4 characters, a short stopword list removed) with f's description or with
  the words of U1.

So "no" is checkable from the schemas: the only function that did the job is absent, and nothing listed is about the
same thing. The row's `audit_metadata["ib2"]["kind"]` is `call`, `refusal` or `constructed`; yes / no balancing and the
cap of 8,000 are unchanged (the constructed rows are not separately capped). This is a generic relevance construction
from Glaive's own schemas (the coordinator's "relevance built from those schemas"), with our own wording; it does not
follow any Index benchmark's format.

## C. Run

The candidates are rebuilt as run `c2` from the same pinned raw files; every audit and review of the prereg runs on
`c2`. Expected `fc_*` rows before audits: about 20–25k.
