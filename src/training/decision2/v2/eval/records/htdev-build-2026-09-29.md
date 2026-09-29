# HT-DEV v1: build, item scans, freeze and registration (2026-09-29)

Eval track, build worker (branch `xunzhuo/decision-2-training-eval-devpanel`). Binding spec:
`htdev-prereg-2026-09-29.md`, `htdev-prereg-amendment-1-2026-09-29.md`,
`htdev-prereg-amendment-2-2026-09-29.md` and `htdev-isolation-2026-09-29.md`. Everything ran on node A from exact
code mirrors in `decision20-train-fast:host2` with `--network none`. Downloads and the Hub upload ran on the host.
This record gives counts, ids and hashes only. No item, gold or source text left node A.

## Result

HT-DEV v1 is frozen and registered as a **development** panel `ht-dev`. It has 3,240 items in 13 tasks:

| Type | Items | Tasks |
| --- | ---: | ---: |
| Choice | 1,240 | 5 |
| Noul | 1,500 | 6 |
| Score | 500 | 2 |

| File | sha256 |
| --- | --- |
| `goldfree/ht-dev.prompts.jsonl` (0644) | `30b0bd3569da9dd183f142e606ba6dcb598de7e4d3d9c168f87f91852178fd65` |
| `gold/ht-dev.gold.jsonl` (0600) | `c1f138918f65edf1dc06143a876333f4380f03093c44a1218f20960d6cd49f1e` |
| `MANIFEST.json` (run dir) | `16aca8c3e00150e41e5777b74e63e6b06dc94f5a5ec3958d35aa156ba6732095` |
| `LEAK-AUDIT.json` (run dir) | `90035929500f89eafafa45cce669f0fcdd20b915ad7658a83bd5543ec3cbc0ad` |

- **Registry:** the panel is registered in `panels.py` DEVELOPMENT with `originals` = 3240 (`679d3636c`, with a
  test).
- **Build code:** `a580d4697` (pool, flagged rows, scan drops and select).
- **Run directory:** `/data/dev2/runs/eval/htdev/build-v1/` (700; files 600).
- **Backup:** private Hub dataset `llm-semantic-router/decision-2.0-eval-artifacts` under `htdev/v1/`, commit
  `ccd6d8bac6b8cf0e69a9a27d99d48e9326063d5c` (§7).

## 1. Fig-QA lineage (amendment 2 item 5)

FLUTE (Chakrabarty, Saakyan, Ghosh and Muresan 2022, [arXiv 2205.12404](https://arxiv.org/abs/2205.12404)) derives
from Fig-QA:

- **§2.1.2 (simile contradiction pairs):** "We further selected 108 challenging <simile, entailment, contradiction>
  instances from Liu et al. (2022b)".
- **Appendix A.2.2:** "To select challenging instances from FigQA dataset Liu et al. (2022b)…"
- **Other sources:** FLUTE's metaphors come from Chakrabarty 2021, Srivastava 2022 and Stowe 2022. The
  [HF card](https://huggingface.co/datasets/ColumbiaNLP/FLUTE) lists no sources.

FLUTE is in training (`css_flute_official_train`) and is a CSS15 task, so Fig-QA fails the dataset-level rule.
**MAGPIE** (figurative backup 1) replaces it.

MAGPIE's dataset-level embedding scan covered 39,245 items against the ISO protected set: 0 items at ≥ 0.93 and 0 in
the 0.85–0.93 band. Pavlick is dropped under amendment 2 item 1.

## 2. Pool and flagged rows

**Fetch:** every pinned file verified, and the revisions equal the ISO pins. The "7 missing" historical files
(amendment 2 item 7) are all older `.gitattributes` versions. All 32 are on disk; the scanner skips dot-files, and
they hold no training rows. Nothing needed fetching.

**Pool:** 12 primaries (excluding Fig-QA and Pavlick) plus MAGPIE, 206,183 items. Pool sha256
`50e86d8026ed7307b4d4a669cd1df0434725d5b4aee07bfd96ddb5a61b1634e9`, code `a580d4697`.

**Flagged-row removal (amendment 2 item 6).** The ISO `ADMISSION.json` has sha256 `deff36a8…`. Its
`training_manifest_sha256` and `training_corpora_sha256` match the ISO files.

The recipe's literal "any shared leaf" rule drops shared contexts wholesale: 28,148 items, including Circa and
claim_stance topic sentences. The strict "all leaves" rule catches only 135. The build uses **rarest**: a flagged row
is matched through its leaves with the lowest pool item count. All three counts are in `FLAGGED.json`.

Rarest drops **5,551** items (ids sha256 `d36587fd…`):

| Task | Items dropped |
| --- | ---: |
| MAGPIE | 5,206 |
| claim_stance | 160 |
| NYCC | 72 |
| Circa | 45 |
| Diplomacy | 42 |
| WiC-TSV | 9 |
| MHS | 8 |
| BRIGHTER | 5 |
| PubHealth | 3 |
| Hyperpartisan | 1 |

## 3. Item-level scans (200,632 items after flagged rows)

### Lexical scan (`ITEM-SCAN-RECIPE.md`, `v2.eval.sealed.overlap`, CPU)

| Scanned against | OVERLAP | REVIEW |
| --- | ---: | ---: |
| Full training manifest (136 s) | 4 | 84 |
| Six goldfree prompt files plus SELECT/CAL (`pi-v3/rights_clean_{select,cal}.jsonl`) | 6 | 3 |

The six goldfree prompt files: `typed-final`, `css15`, `public231`, `typed-dev`, `css-pilot` and `mlx-diag`.

### Embedding scan

- **Model:** Qwen3-Embedding-0.6B @97b0c614.
- **Protected set:** 573,303 windows, the PI-v4 inventory (87 roles) plus the ISO training-social set (37,047
  windows).
- **Candidates:** 217,226 windows.
- **Hardware and lease:** one job on GPU5 (22:29:33–22:55:24Z, 1,551 s). Lease entry `gpu5.lock/owner.eval-htdev`
  was written before the start and appended at the end (exit 0). Only `ROCR_VISIBLE_DEVICES` was pinned. No other
  job was disturbed.

Results:

- **At ≥ 0.93:** 0 items.
- **In the 0.85–0.93 band (counted, not dropped):** 148 items.

| Task | Band items |
| --- | ---: |
| Circa | 86 |
| MHS | 23 |
| Diplomacy | 19 |
| claim_stance | 11 |
| Empathic | 3 |
| Hyperpartisan | 3 (best match: css15) |
| PubHealth | 3 |

### Drops

The rule drops lexical OVERLAP or REVIEW, embedding ≥ 0.93, and items with no scannable shingle (the scanner cannot
clear them). That drops **3,262** items (ids sha256 `c184e611…`):

| Task | Dropped | Breakdown |
| --- | ---: | --- |
| MHS | 1,595 | 1,592 no-shingle, 3 panel OVERLAP |
| MAGPIE | 975 | 949 no-shingle, 23 training REVIEW, 2 panel OVERLAP, 1 panel REVIEW |
| BRIGHTER | 493 | 492 no-shingle, 1 panel OVERLAP |
| PubHealth | 54 | 48 no-shingle, 6 REVIEW |
| WiC-TSV | 48 | 42 no-shingle, 6 REVIEW |
| Diplomacy | 43 | 42 no-shingle, 1 training OVERLAP |
| NYCC | 39 | 3 OVERLAP, 36 REVIEW |
| Circa | 15 | 13 training REVIEW, 2 panel REVIEW |

## 4. Selection (amendment 2 items 2–4, 8, 9)

No task needed a backup except figurative. No length gate fired.

| Task | Role | Type | Items | Split stage | Balance | Groups / clusters | Length-only macro-F1 |
| --- | --- | --- | ---: | --- | --- | --- | ---: |
| humor/nycc_pairs | primary | Choice | 250 | test | gold | 250 | 0.524 |
| implicature/circa | primary | Choice | 250 | test+validation+train | gold | 250 | 0.257 |
| figurative/magpie | backup1 | Noul | 250 | test | gold | 250 | 0.504 |
| persuasion/ukpconvarg | primary | Choice | 250 | test | gold length rank | 31 | 0.498 |
| misinfo/pubhealth | primary | Choice | 240 | test+validation | length | 240 | 0.231 |
| emotion/brighter | primary | Choice | 250 | test | gold | 250 | 0.139 |
| stance/claim_stance | primary | Noul | 250 | test | gold | 30 topics | 0.440 |
| ideology/hyperpartisan | primary | Noul | 250 | test | gold | 177 | 0.582 |
| deception/diplomacy | primary | Noul | 250 | test | gold | 40 dialogues / **2 games** | 0.593 |
| moral/scruples | primary | Noul | 250 | test | gold | 250 | 0.489 |
| empathy/empathic_reactions | primary | Score | 250 | test | gold | 250 | 0.151 |
| hate/mhs | primary | Score | 250 | test | gold | 250 | 0.369 |
| wordsense/wic_tsv | primary | Noul | 250 | test+validation | gold | 250 | 0.554 |

- **Diplomacy:** at most 10 items per dialogue. The bootstrap cluster is the game (`cluster_id`), and the test split
  holds only 2 games.
- **PubHealth:** 240 items (80 per class).

## 5. Option-order fix and rebuild

The first selection (code `a9ad4fd56`) showed one option order per Choice task. The converters build each question
with the salted per-item `display_order`, but `jsonl_bytes` wrote the pool and panel files with `sort_keys=True`. That
re-sorted every `criteria` dict alphabetically, and state fields too.

For the fixed-option tasks (Circa, BRIGHTER, PubHealth) this broke amendment 2 item 8. It also meant the leak audit
had excluded them as fixed option sets. The pairwise tasks (NYCC, UKPConvArg) were unaffected: their A/B keys are
positional, and the texts were already shuffled.

**Fix (`a580d4697`):** JSON lines keep insertion order, with a test that the written criteria order equals the
converter's display order and varies across items.

**Rebuild at `a580d4697`:**

- **Pool:** all 206,183 rows equal the scanned pool in canonical (key-sorted) form.
- **Exclusions:** the flagged-id set is equal (5,551), and the scan-drop id file is byte-identical. The existing
  lexical hits and embedding results were reused. Both are key-order-free: shingles come from string leaves, and the
  embedding text is `json.dumps(sort_keys=True)`.
- **Selection:** the same 3,240 ids, with prompts and gold equal in canonical form.
- **Option orders now:** Circa 24 distinct, PubHealth 6, BRIGHTER 209.
- **Old run:** moved to `void-build-v1-sortkeys/`. Its item files were deleted; its MANIFEST and LEAK-AUDIT are kept.

## 6. Checks

- **Leak audit** (`v2.eval.leak_audit audit --panel ht-dev --files …`, code `a580d4697`): 3,240 questions.
  `verdict_option_surface` CLEAN and `verdict_with_state_surface` CLEAN. `id_mentions_gold` 0. Every per-task and
  pooled cue is CLEAN, including position, key and description cues for Circa, BRIGHTER and PubHealth.
- **Scorer sanity** (`htdev.score` seal and score):
  - Gold as predictions: macro-F1 1.0 for all 13 tasks, H_dev 1.0, 3,240 valid.
  - All-invalid: 0.0 for all 13 tasks, H_dev 0.0, 0 valid.
- **Freeze:** install was write-new (targets absent before, named explicitly), and `cmp` matched against the run dir.
- **Registry and runner (node A, CPU only, mirror `679d3636c`):**
  - `panels verify --panel ht-dev` returns both frozen hashes.
  - `same_panel collect --panels ht-dev` ran with the `run_same_panel.sh` volumes (mirror ro, goldfree ro, model dir
    ro, run dir rw) but without GPU devices. A parse-only adapter spec (`python3 -m json.tool --json-lines --compact`)
    was used. The frozen-hash check passed, and it parsed 3,240 rows (expected 3,240), exit 0. The parse output was
    deleted.
  - Prompt structure: every row is `{id, state, questions.decision}` with type choice/noul/score. `input_digest`
    computes, and the 3,240 ids are unique.
  - `run_same_panel.sh` itself always takes a GPU lease, so it passes `--panels ht-dev` through unchanged but was not
    launched.

## 7. Hub backup

- **Storage before upload:** the org's private total was **91.43 GB** of the 100 GB cap, measured with the release
  storage probe. That is higher than the 67 GB noted in COORDINATION; the largest repos are `dev2-dec-staging` 33.69
  and `dev2-9b-staging` 31.78. The upload was about 5.7 MB, and nothing was deleted.
- **Upload:** `htdev/v1/` holds `ht-dev.prompts.jsonl`, `ht-dev.gold.jsonl`, `MANIFEST.json` and `HASHES.json`.
  `HASHES.json` has the file sha256s, the ISO `ADMISSION.json` (`deff36a8…`) and `TRAINING-MANIFEST.json`
  (`cfb156d5…`) hashes, and the build and registry commits.
- **Commit:** `ccd6d8bac6b8cf0e69a9a27d99d48e9326063d5c`, verified by downloading at that revision and matching
  sha256.

## 8. GPU time

| Stage | GPU-h |
| --- | ---: |
| Isolation embedding scans | 0.21 |
| Build item embedding scan (GPU5) | 0.431 |
| **Total** | **0.64** |

## 9. Open issues

1. **Diplomacy bootstrap:** the test split holds only 2 games, so its cluster bootstrap is nearly degenerate. Read
   the Diplomacy interval as unreliable.
2. **Flagged-row rule:** the build used the rarest rule instead of the recipe's "any" rule (§2). All three counts
   are recorded.
3. **Embedding scan:** one merged pass (PI-v4 plus training-social) replaced two runs. Per-set band counts are
   attributed by the best-match role.
4. **Unscanned backups:** backups other than MAGPIE were not item-scanned, because none was selected.
5. **High length-only baselines:** Diplomacy 0.593, Hyperpartisan 0.582 and WiC-TSV 0.554 against chance 0.5. They
   are disclosed, not gated; the gate applies to Choice only (amendment 2 item 9).
6. **Hub storage:** 8.6 GB of headroom remains.
