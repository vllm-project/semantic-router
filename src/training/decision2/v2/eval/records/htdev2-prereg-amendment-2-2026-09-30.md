# HT-DEV v2 preregistration, amendment 2 (2026-09-30): pool results before the scans and the freeze

Committed after the first `pool` run (counts only, `POOL.json` on node A) and before the overlap scans, the sampling
and any model run. Thresholds, scoring, validation and the decision rule are unchanged.

1. **Reconstruction check result.** 12 of 13 tasks map 100% of their formal items by (normalised context hash, raw
   prompt hash), wiki_corpus 99.4%; label agreement is 100% on every mapped task.
2. **reddit_humor.** Its 500 formal contexts all occur in `test.tsv` column 3, but the formal SALT prompt is the
   `mappings.py` humor template without its leading `" \n\nConstraint: "` and its final newline (the snapshot's test
   file predates that template edit). The builder uses that exact prompt; the prompt-hash check verifies it. The
   formal sample is 500 of the 608 `test.tsv` rows, not the first 500 (amendment 1 item 1); exclusion is by context
   hash, so nothing else changes.
3. **persuasion and raop cannot form a class-complete parallel task.** The formal sample used every item of one gold
   class: persuasion's 217 unsuccessful replies (the SALT pool has 609 rows: its held-out part is 155 successful
   replies only) and raop's 57 "Scarcity" sentences. A class-balanced sample like the formal one is impossible, so the
   freeze rule (every task label present, ≥ 60 items) drops both. Items outside SALT's pool construction (e.g. deeper
   replies) are not used, since they would change the construct.
4. **Result: 11 tasks** (emotion, ibc, media_ideology, talklife, wiki_politeness, tempowic, flute, mrf, conv_go_awry,
   reddit_humor, wiki_corpus), at most 1,650 items. The design check is re-run on this task set for the record
   (informational, stored formal predictions only).
