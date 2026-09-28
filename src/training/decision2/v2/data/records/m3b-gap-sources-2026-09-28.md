# M3b item 5 — new data sources, gap arms H7 and H8: record draft (2026-09-28)

**Status: draft (uncommitted code).** Built on node A (CPU only) from a scratch mirror of the
uncommitted working tree on top of `8eb5d42cd` (mirror `b4`: `v2/data/m3/src_gap.py`
`dce5d491…`, `v2/data/m3/gap_nodeA.sh` `dabba075…`; the working-tree `src_gap.py` differs from
it only in two docstring lines about `--tokenizer`). The coordinator re-runs `gap_nodeA.sh` from a
committed exact mirror before anything is published. No GPU was used,
nothing was uploaded, no C1-registry dataset and no sealed directory was read.

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

**C1 check.** Every `dataset_id` of `v2/eval/records/sealed-c1-source-registry-2026-09-28.json`
(47 entries, all grades including rejected) was compared by name and upstream with every source and
parent above (HoVer, HotpotQA, Wikipedia dumps, Natural Questions, TyDi QA, MIRACL, HAGRID, JGLUE,
SentiMix/SemEval-2020, Zenodo, LinCE, Natural Instructions): no match anywhere in the registry text.
The only generic matches are entry 42 ("Arena-90K / SHP / SQuAD / SNLI / IMDB derivatives;
IndicQE-APE; tasksource/*") and entry 43 (peer JEV aggregators); none of them is read.

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

## 3. Audits (node A, `gap_nodeA.sh audit|finalize`)

1. Per-row tokens: `v2.data.m2.row_tokens` in the pinned image (native `qwen3.5-0.8b-base@dc7cdfe2`,
   raw `kai-0.6b@7185f514`, spec `arms-v1/tokenizers.json`); 8,192 budget by whole groups
   (`src_gap budget`): **0 groups over** in both arms.
2. Lexical overlap (`v2.data.overlap`, four methods) of TRAIN+AHO+SHO against a new manifest
   `pi/manifest.json` `1e2e13b6…` = PI-v3 (`fc09b2bd…`, 56 roles) + the eleven A7 AHO slices (hashes
   equal to the A7 v3 registry) + the nine v2 AHO slices; whole-group quarantine
   (`apply_quarantine`) with the PI-v3 TRAIN roles and the **v2 AHO roles report-only** (see §6).
3. Shortcut gates: `m2.audit_cells` cells (source × family, ≤ 20,000 rows), `v2.data.shortcut`
   (state-removed, option-only; majority + 5 points), and a state-length logistic baseline per cell
   (`src_gap length-baseline`, group-disjoint 5-fold). Failing cells would be dropped from every slice.
4. Held-out dedup (`m2.dedup_heldout`), final tokens, isolation (`v2.data.freeze isolation`, the two
   arms' slices against all 69 existing partitions).

## 4. Results

Native tokens are Qwen3.5-0.8B-Base native `encode` of the final files. "Long" = tokens in rows of
≥ 2,000 native tokens.

| Arm | Slice | Rows | Groups | Choice / Noul / Score | Native tokens | Long share | Max | SHA-256 |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| H7 | train | 4,931 | 1,881 | 0 / 4,104 / 827 | 18,761,859 | 99.9% | 7,563 | `997aa1ea743c6d731d41ec7d6ea3b802bea2ab8fe26e572f57c088ed7bf031bb` |
| H7 | aho | 508 | 209 | 0 / 442 / 66 | 1,943,278 | 99.8% | 6,920 | `92e0ddd38ee90d38008f454487885b5fa2303c42e61524223e89108cc0ef2187` |
| H7 | sho | 50 | 24 | 0 / 44 / 6 | 178,734 | 100% | 6,313 | `f5a3e710db4fa23cd52e2fcf0da335628a51f3ba7741a38e71e049efde089a46` |
| H8 | train | 21,426 | 17,086 | 6,964 / 8,476 / 5,986 | 12,539,410 | 55.1% | 7,080 | `91af12b9c21ef6a03151aec4ab1f02bf0e1bc45926a9cc2c5653225a73df4a82` |
| H8 | aho | 2,429 | 1,929 | 810 / 970 / 649 | 1,443,238 | 54.6% | 6,867 | `d397f6f9b72c1fcd84b1e511e4027702878995464bc54650b4bda80c017984ea` |
| H8 | sho | 439 | 354 | 143 / 162 / 134 | 198,677 | 47.7% | 6,359 | `c1983981284671e8dd24dde43e4fba2ac2086457dc56a2a266f6c7e6768c2efa` |

**H7 TRAIN by family:** `hover_answerable` 2,246 rows / 860 groups / 8,972,713 tokens;
`hover_coverage` 827 / 108 / 3,454,319 (Score L3 148 × 3, L4 72 × 4, L5 19 × 5, exactly balanced);
`nq_window_removal` 1,858 / 929 / 6,334,827. English only; every row is above 1,024 Kai tokens
(T1a-ineligible).

**H8 TRAIN by family:** `jcqa` 6,964 (ja; gold positions 1,331–1,443 per position);
`sentimix_spanglish` 5,986 (es-en; levels 1,765 / 2,125 / 2,096); `miracl_relevance_<lang>` 864–894
per language (es, fa, fr, hi, zh; 1.29M tokens); `miracl_pool_<lang>` 518–540 per language (4.26M
tokens; hi 93% long, fa 35%, others ≤ 7%); `tydi_window_removal_<lang>` 138–182 rows for ar, en, fi,
id, ja, ru, te, th, and bn 50, sw 6, ko 4 (5.08M tokens; ≥ 97% long except ko and sw). 17 languages (ja 7,142;
es-en 5,986; fa 1,426; fr 1,412; hi 1,412; zh 1,402; es 1,388; id 182; ar 180; te 180; ru 178;
th 172; fi 168; en 138; bn 50; sw 6; ko 4). Noul exactly 50/50. 3,135 rows above 1,024 Kai tokens.

**Quarantine (whole groups, non-report-only roles).** H7: TRAIN 887 of 2,768 groups (9,076 → 4,931
rows), AHO 108 (1,085 → 508), SHO 20 (139 → 50); by role v1 A3 AHO 601 (MuSiQue Wikipedia
paragraphs), CSS15 268, Decision Bench v4 80, A7 AHO 87 (A7q 56, A7m 19, A7h 12), CSS pilot 4,
mlx-diag 3, ml-parallel-dev 3, JevBench-231 1, v1 A1 AHO 1; HoVer 3,677 rows, NQ 468 rows. H8: TRAIN
31 groups (CSS15 10, v1 A3 AHO 11, Decision Bench v4 8, A7 AHO 6, CSS pilot 1, mlx-diag 1), AHO 3,
SHO 0. Report-only hits: v2 AHO H3 2,386 / H6 1,395 / E11 448 / H1 251 / H5 40 groups (H7) and
v2 AHO H5 351 / H3 64 / E11 24 / H6 15 groups (H8); PI-v3 TRAIN roles.

**Shortcut gates: all 24 cells PASS** (3 in H7, 21 in H8; TyDi ko and sw have < 30 rows). Noul and
Score twins sit exactly at the majority in state-removed and option-only (their question, options
and instructions are shared inside each group); JCQA state-removed 0.117 and option-only 0.128 vs
majority 0.203; SentiMix 0.337 / 0.328 vs 0.334. **State-length baseline** within +5 points
everywhere (H7 ≤ +1.5; largest in H8: MIRACL relevance fr +4.9, zh +4.2, es +3.8 — relevance
twins pair different passages, so lengths are not matched there). No family dropped.

**Isolation** of the six final files against the 69 existing partitions: **PASS** (no shared id,
group id or input hash). Held-out dedup removed nothing. Rebuilds from the same sources reproduced
H8 byte-for-byte across mirrors b1/b2 and b3/b4, and H7 across b1/b2.

## 5. Targets

- **Long evidence.** New TRAIN tokens 31,301,269, of which 25,651,376 (81.9%) are in rows of
  ≥ 2,000 tokens. With the frozen v2 arms (141.47M tokens, 15.2% long ≈ 21.5M) the share becomes
  ≈ 47.2M / 172.8M = **27.3%** (floor 20% met; stretch 30% not). H7 alone adds 18.7M long tokens.
- **Score:** 6,813 new Score TRAIN rows (25.8% of the new rows; L3–L5).
- **Multilingual:** H8 adds ja, es-en, fa, fr, hi, zh, es ≥ 1,000 rows each; code-mixed es-en and
  Persian are new languages for the v2 human arms.
- **Sources:** three upstream releases new to the v2 human arms (HoVer, Natural Questions, MIRACL
  non-English with its corpus, which v2 pinned but never built) plus new items from TyDi QA and
  JCommonsenseQA; SentiMix Spanglish comes from the Zenodo release A7s already uses for Hinglish, so
  it does not count as a new release.

## 6. Decisions and deviations to confirm

1. **v2 AHO slices are report-only.** The brief protects PI-v3 + A7 AHO; v2 AHO slices were added to
   the manifest as report-only roles. Quarantining them too would remove most H7 groups (HoVer and
   NQ share Wikipedia paragraphs with H3/H6/E11 AHO, as v2 TRAIN already does with its own AHO).
   The private receipts name the groups, so either policy can be re-applied without rebuilding.
2. **Caps raised after the first audit.** A first audit (mirror b2) showed ~50% of HoVer and ~23% of
   NQ rows quarantined; the H7 caps were raised from 2,000 / 1,500 / 1,800 and NQ extended from 32 to
   64 shards before the final build (b4). Content rules did not change; the change is disclosed like an
   amendment.
3. **Budget counter.** The first builds sized states with the raw `tokenizer.json`, which splits Thai,
   Telugu and Bengali marks differently from the `AutoTokenizer` path of `row_tokens` (34 Thai rows
   went over 8,192); from b3 the builder loads the tokenizer the same way (`freeze.load_tokenizer`).
4. **Shortfalls.** TyDi windows for ko (4 rows), sw (6) and bn (50): v2 H5 already used nearly every
   usable question; HoVer coverage keeps 108 TRAIN groups because a HotpotQA parent carries about ten
   coverage rows and whole-group quarantine hits large groups more often.
5. **SHO location.** SHO files are under the run directory (`r4/final/*.sho.jsonl`, mode 0600) as the
   brief asked; they belong under the sealed location before publication and must not be uploaded.

## 7. Node-A outputs (run root `/data/dev2/runs/data/m3b/gap/`)

- Final rows: `r4/final/{h7,h8}.{train,aho,sho}.jsonl`; tokens `r4/final/{h7,h8}.tokens.jsonl`
  (`433a67d9…`, `8bec2b3d…`; one `{id, native, kai}` line per row of all three slices); stats
  `r4/final/{h7,h8}.stats.json`; isolation `r4/final/{h7,h8}.isolation.json`; dedup
  `r4/final/{h7,h8}.{aho,sho}.dedup.json`.
- Build: `arms/build4/{h7,h8}.{train,aho,sho}.jsonl` and `{h7,h8}.build.json` (`e31b8c36…`,
  `647ccaed…`); superseded builds `arms/superseded-b1`, `arms/build` (b2), `arms/build3`.
- Gate receipts: `r4/audits/{h7,h8}/cells/*.jsonl.shortcut.json`, `length-baseline.json`
  (`b33dfb5b…`, `b0a52d2f…`), `overlap.{public,private}.json` (public `51388762…`, `e6bfee93…`),
  `*.budget.json`, `*.quarantine.json`, `*.gates.json`.
- Protected manifest `pi/manifest.json` (+ `pi/receipt.json`); isolation list `existing.json`.
- Sources: `/data/dev2/private/sources/m3b/{hover,nq,nq-extract-64,miracl-corpus,sentimix,hotpotqa-validation,natural-instructions}`
  with `SHA256SUMS` / extraction receipts.

**GPU embedding scan still needed** (Qwen3-Embedding-0.6B@`97b0c614`, v2 thresholds, against the
PI-v3 embedding manifest plus the A7 AHO slices): `r4/final/h7.train.jsonl`, `h7.aho.jsonl`,
`h7.sho.jsonl`, `h8.train.jsonl`, `h8.aho.jsonl`, `h8.sho.jsonl`. Until then both arms are
development-only.

## 8. Open items

- Re-run `gap_nodeA.sh` (extract, build h7/h8, audit, finalize) from a committed exact mirror; the
  second-node byte-identical rebuild; freeze manifests with `license-registry-m3b.json`.
- Coordinator decision on v2 AHO quarantine (§6.1) and on whether H7/H8 enter the XL recipes.
- Super-NaturalInstructions per-task review (instance licence, source, decision type) if wanted.
- More NQ shards are available (223 unused) if more NQ volume is needed.
