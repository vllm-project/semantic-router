# A7 preregistration amendment 1: construction-order option keys (2026-09-28)

Committed before any `a7-dec10-v2` row is built. It defines version
`a7-dec10-v2` = the rules of `a7-prereg-2026-09-28.md` plus rule 7d below.
Nothing else changes: sources, hashes, sub-arms, views, overlap, budget,
shortcut and freeze rules are as preregistered.

## Finding in the v1 build (blocking diagnostic)

The v1 pre-admission build (commit `38d623633b22`, node B, never published)
and its shortcut receipts showed an answer leak in how the 1.0 builders named
Choice options. Stage1–3 rows (and their Stage4 v2 replays) that use option
keys `result_<n>` number the keys in construction order and then shuffle the
options, so the key number reveals the gold:

| Sub-arm (v1 TRAIN) | Choice rows with `result_<n>` keys | Gold is the largest-but-one key | Gold is the largest key (no-match option) | Keys in display order |
| --- | ---: | ---: | ---: | ---: |
| A7i (public intents) | 7,579 | 6,728 | 851 | 92 |
| A7p (Stage4 v2 replay) | 1,892 | 615 (plus 618 smallest) | 560 | 336 |
| A7o (Stage1–3) | 14,370 | 2,783 (plus 5,641 smallest) | 2,750 | 1,104 |

In A7i the key alone identifies the gold in 7,579/7,579 such rows; the v1
option-only baseline reached 0.473 against a 0.122 cross-validated majority.
Option keys are model input (`segments` renders each candidate's key and
description), so this is a projection leak, not a property of the human
labels; under §2.9 it is blocking. Other numbered key schemes in the corpora
(`c<n>`, `item_<n>`, `answer<n>`, `category<n>`) are already in display order
and show near-uniform gold ranks; Stage4 v2 generated rows (A7g) use none of
these keys. The same leak was present in the 1.0 training data consumed by
Sol, Nox, Eos, Lux and Kai (all decoder candidate blocks include keys); this
is recorded for the coordinator, not corrected retroactively.

## Rule 7d (v2)

For a Choice row whose option keys all match `result_<n>`, the keys are
renumbered in display order (`result_0` … `result_{K-1}` by position).
Descriptions, option order and the label index are unchanged; the original
keys are kept in `audit_metadata.a7.original_keys`. Deduplication, component
linking and view resolution use the re-keyed input hash, so two 1.0 rows that
differ only in their construction-order numbering become one A7 row. Noul and
Score rows keep rule 7b (opaque keys held). The v1 build and receipts stay on
node B as the diagnostic record of this finding; no v1 file is published.
