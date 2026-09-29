# HT-DEV v2 preregistration, amendment 3 (2026-09-30): scan results and the per-task cap, before the freeze

Committed after the two overlap scans (verdict counts and hit locations only) and before the sampling, the freeze and
any model run. The scoring, validation and decision rule are unchanged.

1. **Scan results** (43,889 pre-scan candidates). Protected panels: 90 OVERLAP, 105 REVIEW. Training scope of
   amendment 1 item 6 (122,485 files in 668 path-prefix labels): 12,382 OVERLAP, 2,720 REVIEW. Both drop.
2. **flute and wiki_politeness are dropped.** Every held-out FLUTE row is in a prior-session CSS training pool
   (`data/css_flute_1k_v1` pool and 1k sample, `combined_6k_v1`, `balanced_human_5824_v1`), and 3,820 of 3,837 held-out
   politeness rows are in `data/css_wiki_politeness_v1` (pool and 1k sample). Neither keeps a class-complete sample.
3. **wiki_corpus stays at the conservative scope.** Its hits lie in the C1 v1.3 rescan's extraction of the data track's
   raw ConvoKit `wiki-corpus.zip` audit copy (not training data), but they are dropped anyway as preregistered; 301
   clean items (142 / 159 per class) remain. The same rescan work tree also holds an extracted copy of the SALT MRF
   directory; its hits are dropped too.
4. **Result: 9 tasks** (conv_go_awry, emotion, ibc, media_ideology, mrf, reddit_humor, talklife, tempowic,
   wiki_corpus); dropped: indian_english_dialect and tropes (a priori), persuasion and raop (amendment 2), flute and
   wiki_politeness (training scan).
5. **Per-task cap 216** (was 150), so the panel keeps the preregistered size of at most 1,950 items with fewer tasks:
   per-class quota ⌊216 / k⌋ (emotion 36, three-class tasks 72, binary tasks 108). The group cap (2), the floor (60) and
   the seeded order are unchanged; every task has enough clean candidates in every class for its quota.
