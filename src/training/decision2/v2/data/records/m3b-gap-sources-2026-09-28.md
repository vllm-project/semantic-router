# M3b item 5 — new data sources, gap arms H7 and H8 (2026-09-28)

**Status: final. H7 and H8 pass gates 1–8 of amendment 2 and are published** (TRAIN + AHO) in the
private HF dataset `llm-semantic-router/decision-2.0-training-data` at revision
**`09f73967bc21b2b1e27160397272b7f66a1ef3af`**, folder `m3/arms/`. Gate 10 (same-source) is
recorded as PASS by the same-source audit (`m3b-same-source-2026-09-28.md` §4d, `3824d2fda`).
Gate 9 (node-B rebuild) is pending: another agent runs it, and this record did not wait for it.

- **Build:** exact mirror of `39bc74ec507c411c4720011860e411b727721940` (tree
  `c29aff6e62455b91873f2e78f2bfcc18e10b5637`).
- **Audits, finalize, stats, freeze:** exact mirror of
  `d14dff8e8c6e4f5e796b097ec4ce9cc9c193d3b9` (tree `b133490abe4c9307bb0973cdbbad4d7442e6debe`).
  Its build code is identical to `39bc74ec5`, and a second build from it was byte-identical.
- **Node A:** CPU only.
- **GPU:** only the embedding scan, on node B GPU7, 0.21 GPU-hours (section 3).
- No C1-registry dataset (any split) and no sealed C1 directory was read.

The draft builds b1–b4 (uncommitted working trees on top of `8eb5d42cd`) are disclosed in
amendment 2 §1. Nothing from them was published or trained on.

Rules otherwise as `data-arms-v2-prereg-2026-09-28.md` with amendments 1–2 (group-level slicing
AHO `sha256(group_id) % 10 == 0`, SHO `sha256("sho-v2:" + group_id) % 50 == 0` among the rest,
positional option keys, whole-group caps in seed-hash order, 8,192 native-token budget, gates at
majority + 5 points on ≥ 30-row cells, whole-group lexical quarantine).

## 1. Licence and availability screen

| Source | Pin | Licence (dataset / text) | Redistribution | Decision |
| --- | --- | --- | --- | --- |
| HoVer (TRAIN claims, v1.1) | GitHub `hover-nlp/hover@39b84697` `hover_train_release_v1.1.json` `1f1cd57a…`; paragraphs from HoVer's HotpotQA-processed Wikipedia `wiki_wo_links.db` `c37ee397…` (nlp.cs.unc.edu); distractor titles `train_tfidf_doc_retrieval_results.json` `45e03a5e…`; HF card `hover-nlp/hover@c0e43052` | CC BY-SA 4.0 (website and card; code MIT) / Wikipedia CC BY-SA | attribution, share-alike | **admitted** (H7) |
| HotpotQA (parent questions only) | HF `hotpotqa/hotpot_qa@1908d6af` distractor TRAIN (v2 pin) + `validation-00000-of-00001.parquet` `c20b638c…` | CC BY-SA 4.0 | — (never rendered; keys HoVer groups) | used for group keys |
| Natural Questions (TRAIN) | HF `google-research-datasets/natural_questions@e8103d56`, `default/train-00000..00063-of-00287.parquet` (64 of 287 shards, every SHA-256 equal to the Hub LFS id); extraction `nq-train.jsonl` `ad4140ed…` | CC BY-SA 3.0 (card, Licensing Information; code Apache-2.0) | attribution, share-alike | **admitted** (H7) |
| TyDi QA primary (beyond H5) | HF `google-research-datasets/tydiqa@da78f23f` (v2 pin) | Apache-2.0 / Wikipedia CC BY-SA | attribution, share-alike | **admitted** (H8) |
| MIRACL es, fa, fr, hi, zh (TRAIN) | topics/qrels `miracl/miracl@5be20db9` (v2 pin); passages `miracl/miracl-corpus@d921ec7e` (the five languages only) | Apache-2.0 / Wikipedia CC BY-SA | attribution, share-alike | **admitted** (H8) |
| JCommonsenseQA v1.3 (TRAIN) | JGLUE `yahoojapan/JGLUE@6f071c09` (v1 pin); v1 A5 items excluded | CC BY-SA 4.0 | attribution, share-alike | **admitted** (H8) |
| SentiMix Spanglish (TRAIN) | Zenodo `10.5281/zenodo.3974927` `Semeval_2020_task9_data.zip` `69509f17…` (the file A7 pinned), member `Spanglish/Spanglish_train.conll` `b78294c5…` | CC BY 4.0 (record licence field) / tweets | attribution | **admitted** (H8) |
| MIRACL-English | — | Apache-2.0 | — | **excluded**: parent of HAGRID, a Decision Bench v4 source (v2 prereg §1.2) |
| Super-NaturalInstructions | GitHub `allenai/natural-instructions@55a36563` (metadata census only) | repo Apache-2.0; per-task instance licences: 513 of 1,613 tasks "Unknown", ~130 NC or bespoke (CC BY-NC(-SA), PAWS, Amazon, MSR, CCG, "© Original Authors") | per task | **deferred**: ~720 tasks look permissive with non-excluded sources by a name screen, but most are translation or generation, many re-package excluded sets (XNLI, PAWS-X, MASSIVE, SNLI/SQuAD/IMDB, CSS parents); a per-task decision-type and source review was not done. `tasksource/*` mirrors are C1 entries and must not be used |
| TyDi QA GoldP | — | Apache-2.0 | — | not used: subset of the primary task's questions (same groups), mostly single-sentence passages (v2 amendment 1) |
| NQ simplified (GCS) | `storage.googleapis.com/natural_questions/...` | CC BY-SA 3.0 | — | not used: HTTP 403 without credentials; the HF mirror is pinned instead |
| MASSIVE, PAWS-X, XNLI; ConditionalQA; NC / research-only sets | — | — | — | excluded (mlx-diag sources; brief) |

**C1 check (gate 2).** Every `dataset_id` of `v2/eval/records/sealed-c1-source-registry-2026-09-28.json`
(47 entries, all grades including rejected) was compared by name and upstream with every source and
parent above (HoVer, HotpotQA, Wikipedia dumps, Natural Questions, TyDi QA, MIRACL, HAGRID, JGLUE,
SentiMix/SemEval-2020, Zenodo, LinCE, Natural Instructions). The comparison was re-run for the
final build and there is no match anywhere in the registry text. The only generic matches are
entry 42 ("Arena-90K / SHP / SQuAD / SNLI / IMDB derivatives; IndicQE-APE; tasksource/*") and
entry 43 (peer JEV aggregators); none of them is read. Entry 6 is a different dataset: Taglish
product-review polarity, not SentiMix.

## 2. Constructions (`v2.data.m3.src_gap`)

Long states are sized in native tokens of the state text under the pinned Qwen3.5-0.8B-Base
tokenizer loaded as `row_tokens` loads it: each item draws a target from (2,300, 3,200, 4,400,
6,000) tokens by SHA-256; no state passes 7,400 tokens. The instruction scaffold is English with the
native question or claim (v2 prereg §1.5). Noul options are `No`/`Yes`.

- **H7 `hover_answerable` (Noul, page-scale C3).** SUPPORTED claims with
  `sha256("hover-alloc:" + uid) % 100 < 50`. Gold = the distinct supporting titles' paragraphs;
  distractors = the claim's TF-IDF top-100 retrieved titles minus supporting titles, distinct
  paragraphs, seeded order, added until the target. Complete twin = all gold + base minus its last
  distractor + one unused distractor; removed twin = gold minus one + base + the same extra
  distractor (equal paragraph counts, one edit each, seeded shuffle). "Do these paragraphs contain
  enough information to verify this claim? Claim: …".
- **H7 `hover_coverage` (Score, page-scale C4).** All other claims (SUPPORTED and NOT_SUPPORTED):
  L = n + 1 rows (n = 2–4 supporting titles), row g holds exactly g supporting paragraphs padded
  with retrieved distractors to one paragraph count per group; levels "The paragraphs state g of the
  n facts needed to check the claim."; question, instructions and options identical in the group.
- **H7 `nq_window_removal` and H8 `tydi_window_removal_<lang>` (Noul, page-window removal).** A
  window of consecutive page paragraphs (NQ: top-level `<p>` candidates as visible text; TyDi:
  non-empty passage candidates) around the annotated answer paragraph, grown left with a per-item
  probability until the token target; complete twin drops the non-answer paragraph closest in
  length to the answer paragraph (never the page's first), removed twin drops the answer paragraph.
  Guards (drop the item): answer paragraph is the page's first (in NQ 51% of long answers are the
  lead, which would let "does the excerpt start with the lead" carry the label); an answer longer
  than 8 words, shorter than `usable_answer`, naming the page subject or stated by the title; the
  removed twin still states an answer (`src_qa.states`, containment for CJK/Thai/Brahmic scripts);
  windows under 4 paragraphs or 2,200 tokens. NQ: one annotation, long answer a top-level `<p>`,
  short answers inside it, yes/no NONE. TyDi: the first minimal answer with yes/no NONE.
  "Does this page excerpt state the answer to this question? Question: …".
- **H8 `miracl_relevance_<lang>` (Noul, C2 twins)** — one judged-relevant and one
  judged-non-relevant passage (title + text), and **`miracl_pool_<lang>` (Noul)** — one relevant
  passage among up to 9 judged-non-relevant ones vs the same pool with the relevant passage swapped
  for an unused non-relevant one (at least 3 judged non-relevant passages required). Queries split between the two by `sha256("miracl-alloc:" + id) % 2`.
- **H8 `jcqa` (Choice, C8).** JCommonsenseQA TRAIN minus the 1,000 v1 A5 items (by `q_id`) and 21
  items sharing a v1 question; five options, English scaffold, seed rotation with positional keys.
- **H8 `sentimix_spanglish` (Score L = 3).** CoNLL blocks → tweet text, A7s sentiment template
  (`Message: …`, negative / neutral / positive), language `es-en`, conflicting and repeated texts
  dropped, ≤ 1.2 × the rarest level.

**Groups and isolation.** HoVer groups are the v2 `multihop` groups of the HotpotQA parent question
(TRAIN or validation), TyDi and MIRACL use `tydi-miracl` (normalized question), NQ `nq`, JCQA `jcqa`,
SentiMix `sentimix-spanglish`. Before the cap, every group whose id or any input hash occurs in the 69
existing partitions (`existing.json` `d54b85ce…`: v2 final TRAIN/AHO/SHO of all nine arms, v1 TRAIN/AHO
of all eight arms, A0 TRAIN, SELECT700, CAL700, CAL698, A7 TRAIN/AHO of all eleven sub-arms; 367,429
groups, 645,817 inputs) is dropped: HoVer 192 + 291 groups (shared HotpotQA questions), TyDi 351–1,400
groups per language (questions used by H5/E11); nothing else.

**Caps (rows, TRAIN+AHO+SHO, before quarantine).** HoVer twins 4,400, HoVer coverage 3,300, NQ
2,600; TyDi windows 200 per language; MIRACL relevance 1,000 and pool 600 per language; JCQA 9,000;
SentiMix 8,000.

## 3. Runs and audits (`v2/data/m3/gap_nodeA.sh`)

### 3.1 Committed-mirror runs on node A

- **Mirrors.**
  - `/data/dev2/src/<sha>-src_training_decision2` for `39bc74ec5` and `d14dff8e8`, both extracted
    from `git archive`.
  - The file-hash digests equal `git archive`: `14cfbef6…` and `f8b4e1cd…`.
  - Node B's mirror of `39bc74ec5` is equal too.
  - Unit tests pass on the host and in the image `f83b1d10…`.
- **Run roots.**
  - `/data/dev2/runs/data/m3b/gap/c1/` (`39bc74ec5`): NQ extraction, PI-v4, A7k check, builds and
    the first audits.
  - `/data/dev2/runs/data/m3b/gap/c2/` (`d14dff8e8`): builds, audits, finalize, stats and freeze.
    Its `pi/` links to c1's, so the recorded PI-v4 hash stays the one in use.
- **Existing partitions.**
  - `existing.json` is the draft's list, byte for byte (`d54b85ce…`, 69 partitions).
  - Every file was hash-verified: each SHA-256 equals the value recorded by the draft build.
  - 60 of them also appear in git records and specs; the 9 v2 SHO slices are in the HF `m2/arms/*/sho.dedup.json` at `ba848147`.
- **Reproduction.**
  - `extract` re-created `nq-train.jsonl` byte-identical to the sources copy (`ad4140ed…`, same
    report).
  - Both builds (c1 and c2) are byte-identical to draft b4 in all six slices:
    - `h7.{train,aho,sho}`: `165e9681…`, `e10d6ffc…`, `1c4afae3…`;
    - `h8.{train,aho,sho}`: `fc5adc2e…`, `372027e7…`, `c1983981…`.
  - `build.json` differs from b4 only in its `rules` and `status` strings: `2a7efd28…` (H7) and
    `c45bd48a…` (H8).
  - The budget step is identical: 0 groups over 8,192 native tokens in either arm.

### 3.2 Code changes (review of `src_gap.py` against sections 1–2)

The constructions, guards, group keys, AHO/SHO slicing, caps, isolation drop, budget and tokenizer
loading match this record. No content-rule bug was found and no row changed. Changes:

1. **`39bc74ec5`**
   - `build.json` no longer says "draft (uncommitted code)". This changes only `build.json`.
   - `src_gap protected` checks the base manifest and every added origin against expected SHA-256
     values before writing, and lists the report-only roles in its receipt.
   - New `pair-check` (A7k) and `stats` subcommands.
   - `gap_nodeA.sh`:
     - new `protected`, `a7k`, `stats` and `freeze` steps;
     - `NQ_EXTRACT`, because `extract` refused to overwrite and so could not be re-run;
     - `finalize` now fails when a cell has no shortcut receipt (a crashed shortcut run would have
       passed silently);
     - optional `EMBED_RECEIPT`; gate outputs are written under `final/`;
     - the eleven `a7_train_*` roles added to `REPORT_ONLY`.
2. **`d14dff8e8`**
   - Quarantine uses the union of the full PI-v4 scan and a scan of PI-v4's 58 quarantining roles
     only (`src_gap quarantining`, section 3.3). This removed rows (section 4).
   - The freeze step also writes per-slice TRAIN/AHO token files.

### 3.3 Gates

1. **Rights: PASS.**
   - `license-registry-m3b.json` (`a598405d…` as published).
   - The freeze manifests carry complete licence tables for all six sources.
2. **C1 registry: PASS** (section 1).
3. **Isolation: PASS.**
   - `v2.data.freeze isolation` of each arm's final slices against the 69 existing partitions
     (every v2/v1/A7 partition, A0, SELECT700, CAL700, CAL698): 0 failures.
4. **Lexical overlap against PI-v4** (whole-group quarantine).
   - **PI-v4 manifest `24b060da2863a964da1ff2c40591943fff9d27e3ce19b28e2a9bd0918fdb47dc`** (87 roles).
     It was built on node A by `gap_nodeA.sh protected` before the first PI-v4 scan:
     - PI-v3 unchanged: `fc09b2bd…`, 56 roles, verified;
     - the eleven A7 AHO slices, quarantining: A7g, A7h, A7i, A7m, A7o, A7p (`a7-dec10-v3`),
       A7k, A7q, A7s, A7x (`a7-enc10-v4`) and A7r (`a7-rec10-v2`), origins equal to the A7
       registries;
     - A7 has no sealed slice: none in its HF registry and none under node B's A7 run
       directories;
     - the eleven A7 TRAIN files and the nine v2 AHO slices, report-only, all origins verified.
   - Receipt `10b804c1…`: 29 report-only roles (the 9 PI-v3 TRAIN roles, 9 `v2_aho_*`,
     11 `a7_train_*`). The quarantining subset is `48fd2537e63b1069…` (58 roles).
   - **Finding.** Report-only rows in the same scan hide matches with quarantining roles. They
     raise gram posting counts: near-duplicate candidates come only from the 6 rarest grams with
     at most 128 postings. They also raise protected-row counts for n-gram boilerplate (5 or more
     protected rows, first 5 refs per text).
     - Evidence: the full PI-v4 scan quarantined fewer groups than the draft's smaller manifest.
       H7 kept 5 TRAIN and 2 AHO groups, and H8 1 AHO group, that the draft had flagged against
       A7q/A7m AHO or Decision Bench v4. Their hits moved to A7q TRAIN, v1 A3 TRAIN or v2 AHO, or
       became boilerplate.
     - The quarantining-roles scan flags 1,207 H7 groups against 1,008 from the full scan, and 59
       H8 groups against 33. In both arms it contains every full-scan hit.
   - Quarantine is therefore the union of both receipts. This is stricter than the full scan
     alone and relaxes nothing.
   - Final removals, groups (a group can count under several roles):
     - H7 TRAIN 1,055 (9,076 → 4,292 rows): v1 A3 AHO 808 (MuSiQue Wikipedia paragraphs), CSS15
       281, Decision Bench v4 86, A7q AHO 63, A7m AHO 20, A7h AHO 14, `mlx-diag` 5, CSS pilot 4,
       v1 A1 AHO 3, ml-parallel-dev 3, JevBench-231 1. HoVer lost 4,224 rows, NQ 560.
     - H7 AHO 128 (→ 460 rows); H7 SHO 24 (→ 41).
     - H8 TRAIN 55 (21,488 → 21,378): v1 A3 AHO 33, CSS15 17, Decision Bench v4 10, A7q AHO 5,
       A7h AHO 3, A7m AHO 3, CSS pilot 1, `mlx-diag` 1.
     - H8 AHO 4 (→ 2,427); H8 SHO 0.
   - Compared with the draft: every draft-quarantined group is still removed. Additionally
     removed: H7 168 TRAIN, 20 AHO and 4 SHO groups; H8 24 TRAIN and 1 AHO group.
5. **Embedding scan: PASS, 0 quarantined.**
   - Setup: node B GPU7; Qwen3-Embedding-0.6B@`97b0c614`; 1,500/1,000-character windows (480/320
     CJK); batch 256; runtime image `ce895822…`; commit `39bc74ec5` (`embed_scan.py` is unchanged
     in `d14dff8e8`).
   - Candidates: the six final files, copied node A → local pipe → node B, gzip, SHA-256 equal
     at both ends, mode 0600 under `/data/dev2/private/data/m3b-gap/c1/final/`.
   - Protected: manifest `deb4b7e9fd9ef5dd…` = PI-v3's embedding manifest (`d126bb27…`, 47
     roles) plus the eleven A7 AHO slices, hash-checked on node B.
   - Windows: 115,663 candidate, 122,921 protected.
   - 0 groups ≥ 0.93.
   - Review band [0.85, 0.93): 9 groups, all in H8 (TRAIN 6, AHO 2, SHO 1), all against A7 AHO
     (A7x 6, A7s 2, A7q 1); H7 has none.
   - All 9 pairs (fewer than 20 exist) were reviewed: 9 topical neighbours with no shared text,
     short Spanglish tweets against MASSIVE requests, Hinglish tweets and an OASST exchange.
     0 shared text.
   - Wall 756 s (17:00:52–17:13:28 UTC) × 1 GPU = **0.21 GPU-hours**. GPU7 was shared with the
     running own-Lux teacher job (98–100% utilization).
   - Lease: `owner.embed` was written before start and set to `released` at the end; `owner` and
     `owner.eval` were not touched.
   - Receipts: public `46764f44…`, run `e49d2a15…` (path-stripped copies on HF). The private
     receipt stays on node B.
6. **Shortcut gates: PASS, all 24 cells** (3 H7, 21 H8; TyDi ko and sw have fewer than 30 rows).
   - Twin families sit exactly at the majority in state-removed and option-only. JCQA is below it
     (0.117 / 0.128 against 0.203). SentiMix is at most +0.3 points above it (0.337 / 0.328
     against 0.334).
   - State-length logistic baseline (reported, not a gate): H7 ≤ +0.9 points; H8 ≤ +4.2 except
     MIRACL relevance fr at +5.0. Relevance twins pair different passages, so lengths are not
     matched there; the draft had +4.9.
7. **Held-out dedup: 0** groups removed from AHO or SHO.
8. **Canonical freeze: PASS.**
   - `v2.data.freeze freeze` in the pinned image; content hash = file hash for all four
     published files.
   - Manifests, published path-stripped copies: H7 `28dd76a8…`/`1ac8a695…`, H8
     `cab69890…`/`584ceb65…` (TRAIN/AHO).
9. **Node-B rebuild: pending** (another agent). Its interim report says the six build slices and
   the NQ extraction came out byte-identical on node B from `39bc74ec5`.
10. **Same-source: PASS** per `m3b-same-source-2026-09-28.md` §4d (`3824d2fda`). No H7/H8 source
    is SAME DATASET or SAME PARENT CORPUS with a CSS pilot or CSS15 task.

**A7k.** 0 A7k pairs, TRAIN 2,190 or AHO 237, occur in the H6 AHO (`e7223148…`) or H6 SHO
(`7ae4aa75…`) slices, which hold A6h2's 454 KLUE-STS/JSTS pairs. The check was by `input_sha256`
and by normalized sentence pair in either order (`src_gap pair-check`, receipt `a7k-pairs.json`).
Even single-sentence sharing is 0. M3b rebuilt no H6/A6h2 held-out or sealed slice.

## 4. Results

Native tokens are Qwen3.5-0.8B-Base native `encode` of the final files; "long" means tokens in rows
of ≥ 2,000 native tokens. SHO slices are given by rows, groups and hash only.

| Arm | Slice | Rows | Groups | Choice / Noul / Score | Native tokens | Long share | Max | SHA-256 |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| H7 | train | 4,292 | 1,713 | 0 / 3,748 / 544 | 16,122,520 | 99.9% | 7,563 | `7c4133b05664de492bb80b40df5eef0f028257f10327a492564bea5f62813a32` |
| H7 | aho | 460 | 189 | 0 / 400 / 60 | 1,767,474 | 99.8% | 6,920 | `0eaf41c0bfdbcd0c2a262a93804a16d8ba2698a8d621e17e83e554f50748a2b1` |
| H7 | sho | 41 | 20 | — | — | — | — | `3906f68abd4a945443986d58640c25e36d71c7fcbc50d3621a66e3fde6785122` |
| H8 | train | 21,378 | 17,062 | 6,964 / 8,428 / 5,986 | 12,379,042 | 54.5% | 7,080 | `1f19e5ab84e6b80182e99cd8e2b7efc79ec17275048e6daf7987afc335e7d361` |
| H8 | aho | 2,427 | 1,928 | 810 / 968 / 649 | 1,438,676 | 54.5% | 6,867 | `3c1ba01ab3bb9244201f741c428f0e75324a422d5744783016f002154cacd627` |
| H8 | sho | 439 | 354 | — | — | — | — | `c1983981284671e8dd24dde43e4fba2ac2086457dc56a2a266f6c7e6768c2efa` |

**H7 TRAIN by family.**

- `hover_answerable`: 1,982 rows / 764 groups / 7,891,050 tokens.
- `hover_coverage`: 544 / 79 / 2,244,558. Score L3 114 × 3, L4 43 × 4, L5 6 × 5; exactly balanced.
- `nq_window_removal`: 1,766 / 883 / 5,986,912.
- English only. Noul 1,874 / 1,874. Every row is above 1,024 Kai tokens (T1a-ineligible).

**H8 TRAIN by family.**

- `jcqa`: 6,964 rows (ja); gold positions 1,331–1,443 per position.
- `sentimix_spanglish`: 5,986 (es-en); levels 1,765 / 2,125 / 2,096.
- `miracl_relevance_<lang>`: 864–894 per language; 1.29M tokens.
- `miracl_pool_<lang>`: 518–540 per language; 4.26M tokens. Long share: hi 93%, fa 35%, the
  others ≤ 7%.
- `tydi_window_removal_<lang>`: ar 180, te 180, ja 178, ru 176, id 174, fi 166, th 166, en 112,
  bn 50, sw 6, ko 4; 4.92M tokens; long share ≥ 97% except ko and sw.
- 17 languages: ja 7,142; es-en 5,986; fa 1,426; hi 1,412; fr 1,410; zh 1,402; es 1,386; ar 180;
  te 180; ru 176; id 174; fi 166; th 166; en 112; bn 50; sw 6; ko 4.
- Noul exactly 50/50. 3,089 rows are above 1,024 Kai tokens.

**Changes against the draft (r4):**

- H7: TRAIN −639 rows, AHO −48, SHO −9.
- H8: TRAIN −48, AHO −2.
- All of these come from the quarantining-roles scan (section 3.3, gate 4).

## 5. Targets

- **Long evidence.**
  - New TRAIN: 28,501,562 tokens, of which 22,856,958 (80.2%) are in rows of ≥ 2,000 tokens.
    H7 alone adds 16.1M long tokens.
  - With the frozen v2 arms (141.47M tokens, 15.2% long ≈ 21.5M) the share becomes
    ≈ 44.4M / 170.0M = **26.1%**. The floor of 20% is met; the stretch of 30% is not.
- **Score:** 6,530 new Score TRAIN rows, 25.4% of the 25,670 new rows (L3–L5).
- **Multilingual:** H8 adds ja, es-en, fa, fr, hi, zh and es with ≥ 1,000 rows each. Code-mixed
  es-en and Persian are new languages for the v2 human arms.
- **Sources:** three upstream releases are new to the v2 human arms: HoVer, Natural Questions, and
  non-English MIRACL with its corpus (pinned by v2 but never built). There are also new items from
  TyDi QA and JCommonsenseQA. SentiMix Spanglish comes from the Zenodo release that A7s already
  uses for Hinglish, so it is not a new release.

## 6. Decisions and deviations

1. **v2 AHO slices are report-only** (amendment 2 §3). For models trained with H7/H8, the H1, H3,
   H5, H6 and E11 AHO readouts are disclosed as passage-familiar. The private receipts name the
   groups, so a stricter policy can be re-applied without a rebuild.
2. **Quarantine from the union of two scans** (section 3.3, gate 4). This is stricter than the
   amendment's single PI-v4 scan and does not relax it. Both receipts are published.
3. **Caps raised after the first draft audit, and the budget counter fixed from b3:** amendment 2
   §1. The content rules did not change.
4. **Shortfalls.** TyDi windows for ko (4 rows), sw (6) and bn (50): v2 H5 already used nearly every
   usable question. HoVer coverage keeps 79 TRAIN groups: a HotpotQA parent carries about ten
   coverage rows, and whole-group quarantine hits large groups more often.
5. **Publication before the gate-9 record.** This follows the brief, which says not to wait for
   gates 9 and 10. The arms must be withdrawn if the node-B rebuild disagrees.
6. **SHO.** Both final SHO slices were moved to `/data/dev2/private/sealed/m3b/{h7,h8}.sho.jsonl`
   on node A (mode 0600) and never uploaded. Node B's copies were deleted after the embedding scan.
   The intermediate build and audit copies stay in the private run directories.

## 7. Outputs

- **HF** (revision `09f73967bc21b2b1e27160397272b7f66a1ef3af`, parent `530b0bce…`; private before
  and after).
  - Folder `m3/arms/`: 70 files. `registry.json` (`91747a05…`) lists the other 69 with SHA-256.
  - All 70 files were re-downloaded at the revision; every SHA-256 equals the upload and the
    registry.
  - Per arm under `H7/` and `H8/`:
    - `train.jsonl`, `aho.jsonl`, and `{train,aho}.tokens.jsonl` (H7 `5c48a2c6…`/`799c2eb0…`,
      H8 `7c10a127…`/`512f20fd…`);
    - `{train,aho}.manifest.json`, `build.json`, `stats.json` (H7 `56ef6549…`, H8 `83b83952…`);
    - `{train,aho,sho}.quarantine.json`, `{train,aho,sho}.gates.json`, `{aho,sho}.dedup.json`.
  - `audits/<ARM>/` holds `overlap.public.json`, `overlap-quarantining.public.json`,
    `length-baseline.json` and `shortcut/<cell>.json`. `audits/` also holds `embed.public.json`,
    `embed-run.json` and `a7k-pairs.json`.
  - `protected-inventory/` holds the PI-v4 receipts. The folder also has `README.md`
    (`hf-dataset-m3-arms-readme.md`) and `license-registry-m3b.json`.
- **Node A final files** (run root `/data/dev2/runs/data/m3b/gap/c2/`):
  - TRAIN `final/h7.train.jsonl` and `final/h8.train.jsonl`.
  - TRAIN token files `final/h7.train.tokens.jsonl` (`5c48a2c6…`) and `final/h8.train.tokens.jsonl`
    (`7c10a127…`), one `{id, native, kai}` line per TRAIN row.
  - The all-slice token files `final/{h7,h8}.tokens.jsonl` (`9d7d1224…`, `e3a3d448…`) also list the
    SHO ids; use the TRAIN files for recipes.
  - AHO `final/{h7,h8}.aho.jsonl`; stats, gates, dedup, isolation and freeze manifests are also
    under `final/`.
  - Build `build/`; audits `audits/{h7,h8}/`; embedding receipts `embed/`; upload folder
    `hf-upload/`; readback `readback/`.
- **Node A, c1** (`/data/dev2/runs/data/m3b/gap/c1/`): PI-v4 `pi/` (manifest, receipt, the 31
  projected role files, and `manifest.quarantining.json` with its receipt); `a7k-pairs.json`;
  `nq-extract-64/`; `existing.json`.
- **Node B:** embedding workspace `/data/dev2/private/data/m3b-gap/c1/` (manifest, final TRAIN and
  AHO copies, receipts). The driver is `/data/dev2/logs/data/embed-m3b-gap.sh` (`a435a2b9…`) and its
  log `embed-m3b-gap.log`.
- **Draft outputs** (superseded, never published): `/data/dev2/runs/data/m3b/gap/{arms,audits,r4,pi}`.

## 8. Open items

- Gate 9 record (node-B rebuild agent).
- **v2 arms re-screen.** The v2 arms were quarantined with PI-v3's nine report-only TRAIN roles in
  the same scan, so the effect in gate 4 may have hidden hits there too. A report-only rescreen of
  the v2 arms against PI-v3's 47 quarantining roles would size it. The A7 rescreens already used the
  47 roles.
- XL recipe revision r2 (amendment 2 §4), from the two node-A TRAIN files and their token files.
  It is not built here.
- Super-NaturalInstructions per-task review (instance licence, source, decision type), if wanted.
- More NQ shards are available (223 unused) if more NQ volume is needed.
