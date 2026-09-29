# PN1 / HS1 data-quality closure and the qualification-pool relabel: results (research & data, 2026-09-30)

Rules: [prereg](m4-dq-prereg-2026-09-30.md) (`429173dfc`, committed before any sampling) and
[amendment 1](m4-dq-prereg-amendment-1-2026-09-30.md) (`46702e2fc`, committed before PN1-r2 was built or sampled).
Tooling: [`v2/data/dq/`](../dq/). Public receipts: [`dq/receipts/`](dq/receipts/). Assignment: coordinator notes
2026-09-30 00:25, 2026-09-29 22:50 and 16:40.

## Outcome

| Item | Result |
| --- | --- |
| PN1 (a) house embedding scan | **PASS.** 0 groups at cosine ≥ 0.93 and 0 in [0.85, 0.93) |
| PN1 (b) isolation against H7 / H8 | **PASS.** 0 shared ids, groups or input hashes. The short-text scan found 3 near hits, all against H8 TRAIN (disclosed) |
| PN1 (c) blind label review, 224 TRAIN rows | **FAIL.** 12 errors, 5.36% [2.80, 9.17]; the preregistered fix also failed (§3.6). **Amendment 1 (PN1-r2) PASSES:** 4 errors in 204 fresh rows, 1.96% [0.54, 4.94] |
| HS1 spot-check, 3 × 51 TRAIN rows | 153 / 153 blind answers equal the gold. **One template defect** (a duplicated word in F2) was fixed, rebuilt, re-audited and republished |
| Qualification-pool relabel | Done. The manifest was relabelled (89 files moved), and a mixture guard with tests now covers every builder |
| **PN1-r2 and HS1** | **Cleared for released models** at the revisions below |

**Revisions of the private dataset** `llm-semantic-router/decision-2.0-training-data`:

| Revision | Content | Read-back |
| --- | --- | --- |
| **`c41d65d4a5c9d2cbfde7c098d7d943838b32fd12`** | PN1-r2 | 40 / 40 files |
| **`27b1d2f130292268b43a618584bebab5d4e4a6b5`** (current head) | HS1 TRAIN fix; also holds PN1-r2 | 19 / 19 files |

The dataset was private before and after each commit. Storage is 61.36 / 100 GB.

| File | SHA-256 | Rows |
| --- | --- | --- |
| `m4/pn1/arms/pn1.train.jsonl` (PN1-r2) | `c1cec06bd5f1caadb7631338796054aae120cbf8b17dcb0bbf4c694470c52cfb` | 4,364 |
| `m4/hs1/train.jsonl` (build `46702e2fce2b`) | `0dfaa6ebff1b614e5f1c432f2c7432c3fd217f7c344d402c7a7b37fe507de5e7` | 19,968 |
| `m4/pn1/dev/*` and `m4/hs1/dev.jsonl` | unchanged (`c3b68ac1…`, `2e9ee9ab…`) | |

- The superseded TRAIN files stay at their revisions: PN1 `f6f6a531…` at `5ad36287`, HS1 `c90ef316…` at `171e6f0c`.
- **Models trained on either superseded file:**
  - Those trained on PN1 `@5ad36287` carry the §3 label noise: about 1.8% of rows, all false "no".
  - Those trained on HS1 `@171e6f0c` saw the duplicated word in 150 rows; no gold is affected.
  - Retraining is not required. Release under PN1 should use PN1-r2 or disclose the §3 error rate.

## 1. PN1 (a): house embedding scan

- **Setup:** `v2.data.embed_scan`, Qwen3-Embedding-0.6B@`97b0c614`, run on node A GPU1 under a shared lease.
- **Protected set:** the node-B house manifest `deb4b7e9…` (58 roles), resolved on node A to files with the same
  SHA-256 (46 at the same path, 12 remapped), plus HT-DEV, Score5-DEV, Score5-typed-DEV and `hs1-dev`. That is 62
  roles and manifest `122bb747…`.
- **Result:**
  - 134,797 protected windows and 5,954 candidate windows (4,129 TRAIN groups and 1,735 dev groups).
  - 0 groups at ≥ 0.93, and 0 in the review band, so there was nothing to read.
- **Disclosure:** 908 of the 6,862 rows are below the scanner's 40-character minimum (678 TRAIN, 230 dev; mostly
  short zh / ja / ko pairs). For those rows, the lexical scan and the PN1 short-text scan are the only screens.

## 2. PN1 (b): isolation against the H7 / H8 slices

- **Freeze isolation:** `v2.data.freeze isolation`, PN1 TRAIN and dev against the H7 / H8 train / aho / sho slices
  (hashes as published). **PASS**: 0 intersections on every field. PN1-r2 is a subset of PN1, and its TRAIN against
  dev is also PASS.
- **Short-text scan** (`pn1_scan`) of every PN1 sentence against the six slices: 3 near hits (character 4-gram
  containment ≥ 0.8), all against H8 TRAIN, none against AHO or SHO.
  - One TRAIN row, an Arabic `pn-hop` pair. It is not in PN1-r2.
  - One dev row, a French `pn-name` pair (both sentences). It is disclosed as dev familiarity for models trained on
    H8.

## 3. PN1 (c): blind label review

**Design** (prereg §3):

- 8 TRAIN rows per language × construction group × gold cell (28 cells), taken in salted-hash order, at most one row
  per group.
- Blind packets carry only the instructions, the state and No / Yes.
- Two independent reviewer subagents answer every item. A third answers the items where they split.
- A gold error is a majority disagreement with the gold.

**Round 1** (first TRAIN, 224 rows):

| Measure | Value (95% CI) |
| --- | --- |
| Label error (pooled) | **12 / 224 = 5.36%**, exact [2.80, 9.17] |
| Population-weighted error | 1.79%, bootstrap [0.62, 3.11] |
| Agreement with gold: R1 / R2 / majority | .946 / .942 / .946 |
| R1 × R2 | 223 / 224 agree; κ .991 [.973, 1.0] |
| Errors | all 12 false "no" (gold no, reviewers yes); gold yes 0 / 112 |
| By construction | near 8 / 64, name 2 / 42, twin 2 / 54, hop 0 / 64 |
| By language | ru 4, ar 3, es 2, ko 2, fr 1, and ja / zh / de 0 (each of 32 rows, es / fr of 16) |

- **Verdict:** P1 fails (error > 5% and upper bound > 8%). P2 and P3 pass.
- **Fix rule, §3.6:** F2 found no failing cell. F3 (judge margin) was applied: 9 errors in 186 surviving reviewed
  rows, 4.84% with an exact upper bound of 8.99%. Still **FAIL**, so the F3 output was not published.
  - The margin was the wrong lever: the 12 errors had a median judge P(yes) of .096.
- **What went wrong:**
  - **`pn-near` in es, fr, ar, ru and ko** had 8 errors in 40 rows (20%). "No shared translation within two hops"
    does not rule out paraphrases that Tatoeba simply never linked: an omitted "already", tú versus usted, spelling
    variants of one name, aspect-only differences.
  - **Russian `pn-name` role swaps:** exchanging Том and the indeclinable Мэри in place keeps the case marking, and
    with it the roles.

**Amendment 1 (PN1-r2):**

- **Rule:** drop `pn-near` in es, fr, ar, ru and ko (147 rows), Russian `pn-name` (196) and the 2 remaining reviewed
  errors, then rebalance (179 rows).
- **Result:** 4,364 rows, of which ja 2,236, zh 948, de 822, ru 148, ar 108 and ko 102.
  - es and fr leave TRAIN; ar, ru and ko keep swap rows only.
  - Labels are 50/50 in every language × construction group.
  - 491,023 native tokens (Qwen3.5-0.8B) and 4,290 groups.
- **Checks:** `v2.data.freeze` content hash equals the file hash. The A4 overlap learner is at most .485 in every
  language, against a limit of .55.

**Round 2** (PN1-r2, 204 fresh rows disjoint from round 1; 12 per cell, de natural 6 per label):

| Measure | Value (95% CI) |
| --- | --- |
| Label error (pooled) | **4 / 204 = 1.96%**, exact [0.54, 4.94] |
| Population-weighted error | **0.73%**, bootstrap [0.10, 1.70] |
| Agreement with gold: R1 / R2 / majority | .985 / .980 / .980 |
| R1 × R2 | 203 / 204 agree; κ .990 [.971, 1.0] |
| Errors | 4 false "no", all generated `pn-twin` no-edits (zh 1, ko 3). natural 0 / 60, name 0 / 49, near 0 / 30, hop 0 / 30 |

- **Verdict:** P1, P2 and P3 **PASS**.
- **Disclosure:**
  - Korean twin no-edits have the highest remaining error rate: 3 of the 12 ko swap-no rows reviewed.
  - Both reviewers called the edited sentence ungrammatical in 6.9% of swap rows (10 / 144, exact [3.4, 12.4]).
  - The reviewers are the same model family (AI adjudication, not human annotation), so their errors may
    correlate.

## 4. HS1 spot-check

**Design:** 17 TRAIN rows per family × task type, one per world; 153 rows in blind packets. One reviewer subagent
covered F1 and F3, and another covered F2. A fresh adjudicator subagent, blind to which answer was the gold, handled
every disagreement or flag.

| Family | Answers equal to gold (Wilson 95%) | Flags | Confirmed defects (exact 95%) |
| --- | --- | --- | --- |
| F1 quote check | 51 / 51 [.930, 1] | 0 | 0 [0, 7.0%] |
| F2 policy packet | 51 / 51 [.930, 1] | 4 | **4 [2.2%, 18.9%]** |
| F3 unmet condition | 51 / 51 [.930, 1] | 0 | 0 [0, 7.0%] |
| **pooled** | **153 / 153 [.976, 1]** | 4 | 4 / 153 = 2.6% [0.7, 6.6] |

**The defect.** All four flags are one rendering bug, confirmed by the adjudicator (4 of 4 "defect") and traced in
code.

- The travel-expense domain's "nights" condition renders its threshold with the unit ("8 nights"), inside a
  template that already says "consecutive nights". The result was "at least 8 nights consecutive nights".
- It is present in 150 of 19,968 TRAIN rows (6.3% of F2) and in 20 dev rows. Gold is unaffected.
- No gold error was found.

**Fix** (`05a0d391d`): the template now reads "at least {t} in a row". A regression test over every F2 domain fails
on the old template with 24 subtests.

**Rebuild** (`46702e2fc`, build + independent rebuild byte-identical on node A):

- Exactly the 150 TRAIN and 20 dev travel-expense rows changed.
- 138 rows differ only in the phrase. In 12, the packet also gained one filler section, because of the generator's
  length rule, and its clause numbers shifted to match.
- Ids, labels, options, instructions and groups are identical. The oracle re-check passed on every row.

**Re-audit** (the original commands, node A, CPU):

| Audit | Result |
| --- | --- |
| PI-v4 full scan | the same 52 report-only groups |
| PI-v4 quarantining roles | 0 |
| PI-hs1 | 0 |
| Structural scan | 0 of 22,364 |
| Isolation, new TRAIN against the published dev | PASS |
| Local audit (A2, A3a–d, A7) | PASS |
| Native tokens | 14,030,228 (+531) |

**Published:** TRAIN only, so the `hs1-dev` panel and its baselines stay comparable. The dev file keeps the
duplicated word in its 20 rows.

## 5. Qualification-pool relabel

**Registry.** [`v2/common/eval_only_pools.json`](../../common/eval_only_pools.json) lists two pools:

- **The AutoJev-27B runtime-qualification pools** (M3a runs `m3a` / `m3a2` and their copies):
  - 4 path patterns;
  - 29 file SHA-256 values: `qual.prompts.jsonl` `c1d6d922…`, warm-up prompts, sets, outputs p1–p4 and warm, and the
    qualification reports;
  - the SHA-256 of the 487 spot-check ids (128 typed FINAL, 128 CSS15 and 231 public-231).
- **The installed gold-free panel prompts.** Panel gold, including SELECT and CAL, is not listed, because builders
  read it for isolation checks.

**Manifest relabel.** The eval track's `training-corpora.json` (`cbf8a731…`, left in place) was rewritten on node A
into **`training-corpora.evalonly.json` (`2af8e7f3…`)**. `TRAINING-MANIFEST.evalonly.json` (`70847ee8…`) was
produced the same way.

- 89 files moved into `<label>:evaluation-only` labels of kind `evaluation-only`; labels went from 34 to 41.
- The three pools (`local-m3a` 15, `local-m3a2` 17, `r2-local` 17) moved in full.
- 40 further files are copies of the two qualification report JSONs, matched by hash. They sit in code snapshots,
  committed records and the HF teacher folder, and none of them is a training row.
- Scans that should treat these as evaluation-only use the new file. The receipt is
  `dq/receipts/corpora-relabel.json`.

**Guard** (`07d45b055`). [`v2.common.eval_only`](../../common/eval_only.py) is called by every mixture builder:

- `guard` walks the parsed arguments and loaded specs, checking path patterns and the SHA-256 of files up to 8 MB.
- `check_rows` refuses spot-check ids, including `#r` resamples, before a row is written.
- `check_file` reuses a hash the builder has already computed.

These calls were added across tracks (one or two lines each), in:

- the data track's `m2/mixtures`, `m3/xl`, `m3/xl_r2`, `m3/xl_prompts`, `m3/teacher_targets` (only its target
  sources, not the `--qualification` gate report) and `replay_targets`;
- the decoder's `build_mixture` and `m5_block`;
- the 0.6B track's `mixture` and `m6_partition`;
- the 9B track's `lux9b/mix`;
- the ~27B track's `build_mixtures` and `build_replay`.

**Tests.**

- `v2/common/tests/test_eval_only.py`: the pools match and training paths do not; copies are caught by hash; ids are
  refused; the relabel is idempotent.
- The same file checks that every registered builder calls the guard, and a discovery test fails on any new
  builder-like module with a `main()` that does not.
- Two builders refuse a qualification pool before writing anything.
- **Shared-module change** (`26a05f024`): the manifest writer `v2.eval.htdev_iso.manifest` applies the registry to
  future manifests, with a test.
- The existing suites all pass: data 386, dec 51, 06b 141, 27b 83, eval 106, common 24, and the 9B tests.

## 6. GPU, subagents, commits

- **GPU:** **0.088 GPU-h**, all of it the embedding scan on node A GPU1 under the shared lease `owner.data-dq`
  (316 s; lease released). Everything else ran on CPU.
- **Subagents:** 8, all Claude `claude-opus-5-5-max`, all blind.
  - PN1 round 1: R1 and R2. Round 2: R1, R2 and R3.
  - HS1: two reviewers.
  - One adjudicator, who also served as PN1 round-1 R3.
- **Commits on `xunzhuo/decision-2-training-data`:**

  | Commit | What |
  | --- | --- |
  | `429173dfc` | prereg and tooling |
  | `546eb14f4` | fix-pn1 P2 weights |
  | `05a0d391d` | HS1 template fix |
  | `46702e2fc` | amendment 1 and the PN1-r2 tooling |
  | `07d45b055` | the guard |
  | `26a05f024` | the manifest writer |
  | `fa55020fc` | dataset cards |
  | this record | |

- **Node A.** Keys, packets, answers and private receipts are under `/data/dev2/private/data/m4-dq/`, and public
  receipts and `OPERATIONS.log` under `/data/dev2/runs/data/m4-dq/`.
  - The HS1 rebuild is at `/data/dev2/private/data/hs1/46702e2fce2b/`.
  - A first HS1 rebuild launch failed on shell quoting and wrote five empty or error files to `/`. They were
    removed at once and the event was logged.
