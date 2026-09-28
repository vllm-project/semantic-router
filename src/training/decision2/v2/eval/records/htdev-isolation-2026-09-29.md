# HT-DEV v1: dataset-level isolation checks (2026-09-29)

Eval track, isolation worker (branch `xunzhuo/decision-2-training-eval-devpanel-iso`). Binding spec:
`htdev-prereg-2026-09-29.md` (374fa54a4) §3 items 2, 3, 5(ii), 6, 7 and
`htdev-prereg-amendment-1-2026-09-29.md` (committed in 11d55536a, before the checks ran). Everything
ran on node A: CPU steps in `decision20-train-fast:host2` with `--network none` from exact code mirrors
(0687106d8, 11d55536a, cabaec0c2, 112bab9ce); downloads on the host. Counts only. No item, hit text or
source text leaves node A.

## Result

All 14 primary sources are admitted. Two backups are rejected because they were in Decision 1.0
training: CaSiNo and HateXplain, both through Natural Instructions families. One backup,
IBM-ArgQ-9.1kPairs, could not be fetched. Every rejected source is a backup, so no primary
replacement is needed.

| Key (task) | Role | Admitted | Rows | Lexical OVERLAP / REVIEW | Strong training rows (prov. match) | Emb ≥ 0.93 / 0.85–0.93 | Flagged rows | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| nycc (humor/nycc_pairs) | primary | yes | 148,945 | 5 / 304 | 0 (0) | 0 / 0 | 309 | |
| circa (implicature/circa) | primary | yes | 34,268 | 0 / 31 | 0 (0) | 0 / 7 | 31 | SNI (4 tasks); tasksource |
| figqa (figurative/figqa) | primary | yes | 11,859 | 3 / 11 | 6 (0) | 0 / 3 | 14 | the 6 strong rows are all FLUTE (`css_flute_official_train`); see open issue 2 |
| ukpconvarg (persuasion/ukpconvarg) | primary | yes | 12,702 | 0 / 0 | 0 (0) | 0 / 0 | 0 | |
| pubhealth (misinfo/pubhealth) | primary | yes | 12,278 | 2 / 1 | 5 (0) | 0 / 0 | 3 | SNI (3 tasks); tasksource; strong rows have no provenance field |
| brighter (emotion/brighter) | primary | yes | 8,477 | 5 / 1 | 0 (0) | 0 / 0 | 6 | |
| claimstance (stance/claim_stance) | primary | yes | 4,785 | 245 / 57 | 166 (0) | 0 / 17 | 302 | shared upstream Wikipedia (MuSiQue, SQuAD 2, NQ, ROPES, HoVer rows) |
| hyperpartisan (ideology/hyperpartisan) | primary | yes | 2,544 | 0 / 2 | 0 (0) | 0 / 0 | 2 | tasksource |
| diplomacy (deception/diplomacy) | primary | yes | 15,265 | 8 / 3 | 32 (0) | 0 / 0 | 11 | SNI (1 task); strong rows: short dialogue phrases in CLINC150, ABCD, MultiWOZ |
| scruples (moral/scruples) | primary | yes | 32,766 | 0 / 0 | 0 (0) | 0 / 0 | 0 | SNI (9 tasks); tasksource; see open issue 1 |
| pavlick (formality/pavlick) | primary | yes | 21,078 | 30 / 14 | 80 (0) | 0 / 0 | 44 | generic web sentences (GoEmotions, OASST1, DBpedia, MLQE-PE, SQuAD 2) |
| empathic (empathy/empathic_reactions) | primary | yes | 1,860 | 0 / 0 | 0 (0) | 0 / 0 | 0 | |
| mhs (hate/mhs) | primary | yes | 267,774 | 24 / 10 | 2 (0) | 0 / 2 | 34 | jev-bench (316 aggregator files); tasksource card mention |
| wictsv (wordsense/wic_tsv) | primary | yes | 4,513 | 27 / 79 | 3 (0) | 0 / 12 | 106 | |
| magpie | backup | yes (lexical only) | 43,907 | 1,833 / 3,329 | 7 (0) | not run | 5,162 | British National Corpus contexts; 7 training rows |
| rumoureval | backup | yes (lexical only) | 7,994 | 0 / 0 | 0 (0) | not run | 0 | tasksource |
| xed | backup | yes (lexical only) | 14,680 | 14 / 7 | 66 (0) | not run | 21 | subtitle lines in MultiNLI, OASST1, ABCD |
| emobank | backup | yes (lexical only) | 112,146 | 193 / 10 | 1,012 (0) | not run | 203 | shared upstream MASC/OANC (806 rows `dec10:multinli_nonfiction_train`); tasksource |
| casino | backup | **no** | 1,030 | 0 / 11 | 0 (0) | not run | 11 | Decision 1.0 Kai/Lex training (NI family 353; A7 census `natural_instructions:casino`); SNI (7) |
| mfrc | backup | yes (lexical only) | 115,053 | 33 / 21 | 13 (0) | not run | 54 | Wikipedia-QA rows (HotpotQA, NQ, HoVer) |
| moralstories | backup | yes (lexical only) | 12,000 | 0 / 0 | 0 (0) | not run | 0 | tasksource |
| hatexplain | backup | **no** | 20,045 | 6 / 1 | 0 (0) | not run | 7 | Decision 1.0 Kai/Lex training (NI 1502/1503; A7 census lineage in 10 pool files); SNI (3) |
| swords | backup | yes (lexical only) | 5,093 | 0 / 0 | 0 (0) | not run | 0 | |
| ibm_argq_pairs | backup | **no** | — | — | — | — | — | not fetched: no reachable pinned archive (amendment 1 item 3) |

How to read the table:

- **Rows** are the rows of every file of every split, each with at least one string leaf of 20 or
  more characters, as `independence rows` yields them.
- **Strong training rows** are the distinct training rows matched at containment ≥ 0.8 or by an exact
  span of at least 8 tokens.
- **Prov. match** counts the strong rows whose provenance fields carry one of the source's terms.
  The rejection rule needs at least 5 such rows, and none were found. Sources with at least 5 strong
  rows but no provenance match trace to shared upstream text; they are disclosed and not treated as
  dataset presence.
- **Flagged rows** are rows at REVIEW level or worse (lexical) or at cosine ≥ 0.93 (embedding). They
  are listed in ADMISSION.json with a `leaf_sha256`, and the builder drops them from its pool.

Item 6 (JevArena-C1): no source is a C1 registry dataset. The sealed directory was never read,
listed or written.

## Names and provenance check (item 2)

- **Terms file:** `v2/eval/htdev_iso/htdev-source-terms.json`, 3–6 terms per key.
- **Reverse check:** HF ids, parent corpora and papers against `exclusions.json` (481 hard terms,
  `hf_ids`, 23 sibling corpora). There were 0 hard hits. `Empathy-Mental-Health` is a different
  dataset.
- **Scope:** `independence names` read 2,418 row files (95,262,615 rows) and 32,978 text files. The
  roots were all training labels below, the 11 track-branch code mirrors, node A's
  `/data/dev2/runs/a7` (Decision 1.0 hf-docs), the Decision 1.0 model cards
  (`runs/release/scratch/d1cards`) and the C1 aggregator root (disclosure only).
- **Hits:** 10 keys were found.
  - Aggregator root only: emobank, figqa, hyperpartisan, mhs, moralstories, pubhealth, rumoureval.
  - diplomacy: 12 files, all tokenizer vocabulary entries; cleared.
  - scruples: 1 file, the data track's NI census shortlist `m3b/arms-v3/logs/ni-census-v3.json`;
    cleared and disclosed (open issue 1).
  - hatexplain: 3 files, `TRAINING_ATTRIBUTION.md` of Decision 1.0 Kai-0.6B and Lex-0.6B plus the A7
    node-A census; confirmed.
- **CaSiNo** is confirmed by context from the same Decision 1.0 documents. The terms file missed the
  plain name, and the A7 census lists `natural_instructions:casino`.
- **Decision 1.0 NI lineages** in the A7 census: atomic, casino, google_wellformed_query,
  hatexplain, poem_sentiment, propara and wiqa. No HT-DEV primary is among them.
- **Review file:** decisions are in `v2/eval/htdev_iso/names-review.json`.

**Super-NaturalInstructions** (item 7, `flags.json`) uses allenai/natural-instructions @55a36563,
tasks/README.md, 1,613 task names, confirmed by each task's Source field:

- circa: 4 tasks;
- pubhealth (`health_fact`): 3;
- diplomacy (`diplomacy_detection`): 1;
- scruples: 9;
- casino: 7;
- hatexplain: 3;
- all other keys: 0.

**Peer aggregators** (disclosure only; 8 aggregator lists):

- tasksource-jev-typed-decisions@7470ed9e: circa, figqa, pubhealth, hyperpartisan, scruples,
  rumoureval, emobank and moralstories. It also mentions mhs on its card.
- jev-bench@18f88da8: mhs.

## Training scope (frozen)

`TRAINING-MANIFEST.json` lists every file with path, sha256, bytes and rows. `training-corpora.json`
is the overlap scanner's `c1-corpora/1` manifest of the row-format files.

**Totals:** 34 labels, 10,283 files (9,862 unique), 50.32 GB, 91.95M rows, 10,065 corpus files.

**Private HF training dataset:**

- `hf-head`: full snapshot at HEAD `75e557f1`, which was still HEAD when the freeze ran. 690 files,
  3.26 GB, 10.47M rows. The 691st file is `.gitattributes`.
- `hf-hist`: the 8 commits after `30e0a1f7` list 32 file versions that HEAD no longer has; 25 of
  them are on disk (387 MB). The gap of 7 is not yet explained (open issue 6).
- Older revisions in place: `rev-5c0255ed`, `rev-39a120ca`, `rev-ed87a03a`, `rev-d8eae3e4`,
  `rev-3a99bf1c` and `rev-12912429`.
- Recheck-2 roots: `r2-hf-delta`, `r2-extra-derived` (node-A 06b m5 and node-B 27b
  replay-v2-strict), `r2-nodeB` (m3b-lux w2), `r2-local` and `r2-local-code`.
- The C1 manifest's `train-root`, `train-v2`, `train-a7` and `train-m2`, merged with unchanged
  hashes.

**Local copies**, frozen at the start of the run (see `OPERATIONS.log`; files older than 2 minutes,
triton caches skipped):

| Label | Files | Bytes |
|---|---:|---:|
| m3a2 | 82 | 1.11 GB |
| m3b (incl. `arms-v3/r1`, `gap/c2`) | 1,727 | 17.98 GB |
| m3a | 178 | 1.27 GB |
| teachers-v2 | 6 | 0.42 GB |
| arms-v1 | 107 | 0.31 GB |
| arms-v2 | 309 | 4.38 GB |
| from-nodeB | 28 | 0.02 GB |
| hf-upload-* | 572 | 1.26 GB |
| A7 private runs | 1 | — |
| `runs/a7` | 124 | 0.15 GB |
| 06b m4 | 27 | 1.38 GB |
| 06b m5 | 20 | 0.29 GB |
| 06b m6 | 59 | 2.41 GB |
| 9b m2 | 4 | 0.03 GB |
| 9b m3 | 16 | 0.99 GB |
| dec m2 | 21 | 0.58 GB |
| `local-code` (m3b code mirrors) | 2,604 | — |

**Node B:** no new stream. The recipe's node-B files are id/probability files published on HF and
included through `hf-head`. The recheck-2 node-B copies are included.

## Source-level lexical overlap (item 3)

- **Protected set:** 911,062 rows from 23 sources, every file of every split, with sha256
  `0f73906c…`. 104,395 rows have no shingles, and 11 texts are unscreenable.
- **Scan:** `overlap scan` over all 34 labels, 12,050 file tasks, with `--workers 48
  --exact-min-tokens 8` and default parameters. All 10,065 manifest files were verified.
- **Resources:** 246.7 s wall, peak PSS 15.3 GB.
- **Verdicts:** 904,742 CLEAN, 2,428 OVERLAP and 3,892 REVIEW.
- **OVERLAP reasons:** 1,849 exact-only (988 at 12–19 tokens, 861 at 20 or more), 321
  containment-only, 258 both, and 295 short-exact REVIEW.

## Embedding scan (item 5(ii))

- **Model:** Qwen3-Embedding-0.6B @97b0c614, downloaded to node A's HF cache; the revision was
  verified.
- **Settings:** 1,500 / 1,000-character windows, batch 256, thresholds 0.93 / 0.85 and seed
  `decision2-embed-scan-v1`.
- **Candidates:** the 14 primaries, deduplicated by normalised row text, giving 427,096 windows.
- **Protected:** training rows whose provenance matches `embed-families.json`. The pre-deduplication
  matches were GoEmotions 430,737, OASST1 3,398,786, ArgQ-30k 232,140, FLUTE 3,559, `targeted`
  56,990 and `stance` 6,561. After deduplication there were 26,979 rows and 37,047 windows.
- **Result:** 0 rows at cosine ≥ 0.93 for every primary. The 0.85–0.93 band had 41 rows: circa 7,
  claimstance 17, figqa 3, mhs 2 and wictsv 12.

## Compute

- **GPU:** one shared-lease job on node A GPU5, which had 0.3 GB of 274 GB in use at launch. It ran
  21:43:10–21:55:35 UTC, 745 s, which is **0.21 GPU-hours**. The lease was
  `gpu5.lock/owner.eval-htdev`, closed with `end_utc`; no other job was touched.
- **CPU:** source fetch about 3 min (7.9 GB NYCC), HF freeze about 1 min, local freeze about 2 min,
  prepare 25 s, rows 15 s, manifest 3 s (48 workers, page cache), names 12 min (single process),
  overlap 247 s and admission 16 s.

## Hashes (node A, `/data/dev2/private/htdev/iso/`)

| File | sha256 |
|---|---|
| `ADMISSION.json` (mode 600) | `deff36a82154ced2d2ba51137e8fa44f8faac574435bafbec27dcf30e9029227` |
| `TRAINING-MANIFEST.json` | `cfb156d540e2bde2a45d387a877a5e8bd6f1b3edac2aaf8d0ce8a585f2a0b703` |
| `training-corpora.json` | `cbf8a7314af2b1750c94af44057564a50822fdef9327aaac2499c98a84dd6e21` |
| `SOURCES-MANIFEST.json` | `1a3dadbd6b68e9bdb15c409c4bd2ab9453a3fbe3e03e33b666c0984d5434a239` |
| `work/names.json` | `f87811333635fdd98721fb0ac672dadbd07f5f6d43809205f26fe59868acce8c` |
| `work/overlap-receipt.json` | `d97992064015ad3f15cda8056e84f0495b536e86379355b4aabc70bf4a8997c0` |
| `work/overlap-hits.jsonl` | `e77acc507b7e39b25942aabca82640e327f03a336ef1e96768844e1ab9b40357` |
| `work/embed/pi/manifest.json` | `ad9b11d9d02f7cab2703bca90c6b4d20ecde28c4fbb10f0f0a4b500c5e1c34cf` |

## Builder hand-off

- Pool sources come from `sources/<key>/`, whose files are listed with sha256 in
  `SOURCES-MANIFEST.json`.
- Drop every row whose `leaf_sha256` appears in `ADMISSION.json` `flagged_rows`.
- Then run the item-level scans in `ITEM-SCAN-RECIPE.md`, which is also committed as
  `v2/eval/htdev_iso/ITEM-SCAN-RECIPE.md`:
  - (a) `overlap scan --manifest $ISO/training-corpora.json` for all training labels;
  - (b) `--corpus panel-<name>=/data/dev2/private/panels/goldfree/<name>.prompts.jsonl` for
    typed-final, css15, public231, typed-dev, css-pilot and mlx-diag, named explicitly and never
    sealed-c1.
- Drop items with verdict OVERLAP or REVIEW, and items with 0 shingles.
- Embedding at item level (§3.5(i)): use PI-v4 plus `$ISO/work/embed/pi/manifest.json`.

## Open issues

1. **arms-v3 NI shortlist.** The data track's NI census for arms-v3 (in flight) shortlists 2 circa
   and 2 scruples SNI tasks. No such rows exist at freeze. If arms-v3 adds NI circa, scruples,
   healthfact or diplomacy tasks, those HT-DEV tasks stop being isolated for any model trained on
   it. The coordinator should either have the data track exclude these tasks or re-run this check
   before the build.
2. **Fig-QA and FLUTE.** Fig-QA's 6 strong matches are all FLUTE training rows. FLUTE is also CSS15
   `flute`. The FLUTE card does not name Fig-QA as a source, so Fig-QA stays admitted under the
   rule, and the 14 matching rows are flagged. If FLUTE's metaphor subset is shown to derive from
   Fig-QA, replace Fig-QA with MAGPIE.
3. **Training families with no provenance match.** The embedding protected set got 0 TweetEval,
   SocialIQA and SELECT/CAL rows by provenance. Those pools use other field names, or are not on
   node A as row files; the A7 census shows `socialiqa` lineage. The lexical scan covers them fully.
   A lineage-based selection would widen the embedding check.
4. **06b mirror.** The node-A mirror of `-06b@1007d180e` does not match its commit. The branch code
   is covered only through the integration branch mirror (`e0ac90ef8`).
5. **Decision 1.0.** Only the A7 inputs, hf-docs and cards are on node A. The Kai research-mix rows
   themselves are not, so Decision 1.0 NI presence rests on the attribution and census documents.
6. **`hf-hist` gap.** Of the 32 historical versions, 7 are not on disk. They were probably
   overwritten by the same path in the same commit directory or skipped as dot-files. Before the
   build, check them against `hf-freeze.json`.
