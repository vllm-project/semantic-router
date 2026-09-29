# HT-DEV v2 preregistration, amendment 1 (2026-09-30): source details before the build

Committed after the source downloads and structure checks (field names and row counts only) and before any pool is
built, scanned or sampled. No model has been run on HT-DEV v2. Thresholds, scoring, validation and the decision rule
are unchanged.

1. **reddit_humor.** `reddit_full/dev.tsv` does not exist at the RedditHumorDetection repository (the URL returns a
   404 stub), and `test.tsv` has 608 rows, of which the formal sample uses the first 500. The pool is the remaining 108
   `test.tsv` rows followed by `reddit_full/train.tsv` (19,438 rows), both read like SALT's loader (comma-separated,
   no header, text column 3, label column 1).
2. **Processed pools.** Where the SALT snapshot ships the processed pool that the formal sample was drawn from, the
   builder uses it directly instead of re-deriving it: `raop/raop.json`, `talklife/talklife.json` (joined by row index
   to `talklife.csv` for `sp_id` / `rp_id`) and `flute/flute-classification.json`.
3. **ConvoKit field names.** The older corpora (politeness, winning-args, wiki-corpus) store `user`, `root` and
   `reply-to`; the builder reads them as speaker, conversation id and reply-to, as ConvoKit does. Within a conversation
   utterances are ordered by timestamp with a stable sort (missing timestamps last, file order kept), and
   conversations in sorted id order, as pandas `groupby` does.
4. **Reconstruction check** matches a formal item by the pair (normalised context hash, raw SALT prompt hash) from the
   CSS15 gold file, so the speaker-specific wiki_corpus prompts and every task's template are checked too. The same
   check requires the builder's raw label to equal the formal gold on ≥ 95% of the matched items; a task failing
   either bound is dropped.
5. **Pre-scan subsample.** Before the overlap scans each task keeps, per gold class, the first 40 × quota candidates
   in the preregistered seeded order; sampling after the scans walks the same order and skips flagged items. This only
   bounds scan size.
6. **Training scan scope.** The C1 v1.3 rescan's node-A training manifest (123,217 files) minus the two trees that
   hold the CSS panel sources themselves and are not training data: `/data/decision20-20260926/external/LLMs_for_CSS/`
   and `/data/decision20-20260926/external/decision-models-css/`. Everything else stays in, including other external
   repositories and evaluation copies (conservative). Files are relabelled by path prefix so hits can be attributed.
   PN1 and HS1 training files that landed after that scan are added if present on node A (listed in the build record).
