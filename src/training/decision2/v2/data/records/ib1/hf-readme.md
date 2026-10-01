# IB1: licence-clean breadth data for the Decision Index task families (`m6/ib1/`, round 3)

> **Status: RELEASE-SAFE.** Round 3 passed the preregistered fresh decisive blind label review:
>
> - 3 gold errors in 216 reviewed TRAIN rows: 1.39% (exact 95% CI 0.29–4.01%);
> - population-weighted 2.14% (bootstrap 95% CI 0.00–5.36%);
> - no family above the per-family bar.
>
> The bars are ≤ 5.0% with an upper bound ≤ 8.0%, a weighted error ≤ 5.0%, and no family with an exact 95% lower
> bound above 5%. Every audit passed. The 3 rows are removed from TRAIN. Rounds 1 and 2 are earlier revisions of this
> path; both were NOT release-safe. See `status.json`.

Training rows for task families where Decision 2.0 has no training data:

- spam and fraud;
- tool-call appropriateness, and function selection and relevance;
- stance;
- sarcasm;
- original vs perturbed poetry;
- entity-level financial sentiment;
- commonsense (causal alternatives).

**No row of the Decision Index suite is included:** every candidate group with an exact, normalized or 13-gram match
to any suite row was dropped. The preregistration is `v2/data/records/ib1-prereg-2026-10-01.md` with amendments 1, 2
and 3 (branch `xunzhuo/decision-2-training`).

**In-distribution disclosure.** `w2c` (When2Call `train_pref`) and `isarc` (iSarcasmEval `train`) are the train splits
of two datasets whose test files the Index uses. Their test files were never downloaded or read, and every row
matching an Index row was dropped. To measure transfer from the other families alone, leave these two families out.

## Round 3 changes (amendment 3)

- **`sentfin` is a true two-option item.** The options are negative and positive; neutral is neither an option nor
  gold.
- **Removed families:**
  - `wands`, whose relevance boundary is ambiguous;
  - `csqa`, already in the released 2.0 mixtures;
  - `sumedit`. The SummEdits domain rule kept `billsum` and `sales_call`, but rounds 1 and 2 had already sampled every
    remaining group, so the family could not be freshly reviewed.

As a result, IB1 no longer covers the faithfulness, product-search relevance and commonsense MCQ families.

## Files

| File | Rows | Groups | Native tokens (Qwen3.5-0.8B-Base) |
| --- | ---: | ---: | ---: |
| `ib1.train.jsonl` (Choice 20,473, Noul 3,852) | 24,325 | 20,206 | 4,748,622 |
| `ib1.dev.jsonl` (split `select`; diagnostic only, never selection) | 2,067 | 1,714 | 341,276 |

The path also holds:

- per-row tokens (`*.tokens.jsonl`) and the freeze manifests;
- `build.json` (candidate build) and `final.json` (every drop, by reason);
- `stats.json` (per-family sizes, labels and balance) and `isolation.json`;
- `audits/` (count-only receipts, including `audits/rescan/` for the re-scans of the finalized files);
- `license-registry-ib1.json`, `status.json` and `registry.json` (the SHA-256 of every file).

## Families, sources and licences

| Family | Target family | Type | TRAIN | DEV | Source (pin) | Licence | Gold |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| `sms` | spam / fraud | Noul | 1,152 | 62 | SMS Spam Collection (UCI 228) `ucirvine/sms_spam@cae486f9` | CC BY 4.0 | curated spam / ham |
| `w2c` | tool-call appropriateness (in-distribution) | Choice (2) | 475 | 0 | When2Call `nvidia/When2Call@0582f774` `train_pref` | CC BY 4.0 | preferred vs dispreferred behaviour |
| `snips_sel` | function selection | Choice (4) | 2,672 | 310 | SNIPS NLU benchmark `sonos/nlu-benchmark@b86ac7f1` | CC0 1.0 | crowd utterances for a known intent |
| `snips_rel` | function relevance | Noul | 2,700 | 12 | same | CC0 1.0 | intent's function offered or not |
| `args` | stance | Choice (2) | 5,182 | 374 | args.me `webis/args_me@042bdf64` (idebate, debatewise, debatepedia) | CC BY 4.0 | portal pro / con structure; one of each per statement |
| `isarc` | sarcasm (in-distribution) | Choice (2) | 874 | 82 | iSarcasmEval `iabufarha/iSarcasmEval@dfc708b5` `train` (en 464, ar 410) | MIT | author's sarcastic tweet vs author's rephrase |
| `poem` | arts (original vs perturbed) | Choice (2) | 3,584 | 411 | Gutenberg Poetry Corpus `biglam/gutenberg-poetry-corpus@fcd42e24` | CC0 1.0 | original stanza vs a one-word rhyme-breaking substitution |
| `sentfin` | entity-level financial sentiment | Choice (2: negative / positive) | 6,808 | 708 | SEntFiN 1.0 `pyRis/SEntFiN@eba43610` | MIT | annotator entity decisions (neutral not used) |
| `copa` | commonsense | Choice (2) | 878 | 108 | Balanced COPA `pkavumba/balanced-copa@813bd03c` | CC BY 4.0 | COPA alternatives |

Attributions (full lines in `license-registry-ib1.json`):

- Almeida, Gómez Hidalgo and Yamakami 2011 (SMS Spam Collection);
- Ross et al. 2025 (When2Call; generated with Mixtral-8x22B-Instruct);
- Coucke et al. 2018 (SNIPS; function schemas written for IB1);
- Ajjour et al. 2019 (args.me);
- Abu Farha et al. 2022 (iSarcasmEval);
- Allison Parrish (Gutenberg Poetry Corpus; perturbations constructed for IB1);
- Sinha et al. 2022 (SEntFiN);
- Kavumba et al. 2019 and Roemmele, Bejan and Gordon 2011 (Balanced COPA).

No source is NC or ShareAlike. The licence registry still lists every source that was read, including the removed
ones.

Instructions and option descriptions are English. Content stays in its own language (Arabic only in `isarc`). DEV has
no `w2c` rows: every DEV group sits near a TRAIN group.

**Known limitation.** About 1.4% of `args` rows (71 of 5,182) have an argument of three words or fewer, mostly
debate-portal headings. Two of them are just "Summary", and one such row was a review error. These rows were left in
place, because changing data after the review would be tuning.

Families admitted but not in IB1:

- `expqa` (ExpertQA): the rule found no "unsupported" claim with evidence text.
- Removed by the shortcut gate:
  - `medmcqa` (MedMCQA): option-only .346 vs .253;
  - `procb` (ProcessBench): state-removed .500 vs a cross-validated majority of .430;
  - `wanli` (WANLI): hypothesis-only .436 vs .320.
- `maud` (MAUD): emptied by the re-scans.
- `wands`, `csqa` and `sumedit`: removed by amendment 3.

## Construction and audits (counts only)

- **Index rows (G0).** The text leaves of every suite row (selected and added rows) are compared with IB1's data
  leaves after NFKC normalisation, casefolding and `\w+` tokenisation. A group is flagged on any of:
  - raw equality (≥ 20 characters);
  - whole-leaf equality (≥ 6 tokens);
  - sentence or leaf equality (≥ 8 tokens);
  - any shared word 13-gram.

  3,272 candidate groups were dropped whole. No re-scan of a finalized file found a new G0 match. Positive controls
  (re-run in round 3): 2,000 / 2,000 suite rows rendered as candidates and 2,000 / 2,000 perturbed copies were flagged.
- **Overlap with held-out panels (G2).** The E / S / L / N rules run against PI-v4 and PI-ib1:
  - PI-v4: quarantining roles including typed FINAL / DEV, SELECT / CAL, CSS15 + pilot, public 231, mlx-diag and
    Decision Bench v4;
  - PI-ib1: HT-DEV v1, HT-DEV v2, Score5-DEV, score5t-dev fit + check, hs1-dev, PN1 dev, CAL698 and HR2 DEV.

  On the candidates, 531 groups were dropped whole and 1,053 DEV groups near TRAIN were dropped. The boilerplate
  exemption depends on the candidate set, so the finalized files were re-scanned until clean before sampling, and every
  newly flagged group was dropped:
  - pass 1: 53 groups and 7 DEV groups;
  - pass 2: 2 groups;
  - pass 3: clean;
  - pass 4 (after `sumedit` left): clean.

  After the review drops, the final files were scanned again: 1 group was dropped, then the scan was clean.
- **Names (G1) and C1 (G3).** No IB1 source or parent appears in the C1 registry or the panel records. The C1
  source-term scan of the raw files, the candidates and every finalized file found none. The custodian C1 content
  recheck is required before any C1-scored model is trained on IB1.
- **Shortcut gate (G4).** Group-disjoint 5 folds, gated views ≤ majority + 0.05, on the sampled TRAIN file: the
  9 families above pass.
- **Balance (G5).**
  - Noul families: exactly 50 / 50.
  - `sentfin`: 3,404 / 3,404.
  - Rotated `snips_sel`: near-uniform gold position.
  - A / B families: gold-A and gold-longer shares 0.45–0.55.
- **Isolation (G7).** No shared id, group or input with IB1 DEV, HS1 dev, PN1 dev or HR2 DEV.
- **Leak guard.** The 12 round-1 `w2c` rows with IPv4-like strings are dropped; the final files have none.
- **Blind review (round 3).** Fresh AI reviewers of one model family (not human annotation) were blind to gold,
  source and family. Every row sampled in rounds 1 and 2, and its group, was excluded, and the salts were new.
  - Sample: 24 rows per family across 9 families (216). Two reviewers answered every row, and they agreed on all 216,
    so no adjudication was needed.
  - Errors:
    - `args` 1: the argument text is only "Summary";
    - `sentfin` 1;
    - `w2c` 1.
  - Agreement with gold: 98.6% per reviewer.

## Use

Add `ib1.train.jsonl` as one block against a matched-token control (the same base mixture plus the same number of
tokens from its own rows). Screen with HT-DEV v2, and read `ib1.dev.jsonl` per family as a diagnostic only. Measure
Index effects privately on frozen packages.
