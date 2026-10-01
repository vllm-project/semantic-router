# IB1: licence-clean breadth data for the Decision Index task families (`m6/ib1/`)

> **Status: NOT RELEASE-SAFE.** The preregistered decisive blind label review failed: 13 gold errors in 216 reviewed
> TRAIN rows, 6.02% (exact 95% CI 3.24–10.07%), population-weighted 8.33% (bootstrap 95% CI 3.86–13.54%), against
> the bars of ≤ 5.0% with an upper bound ≤ 8.0%. No family reached the per-family failure count (4 of 18). There is
> no fix rule, so IB1 is published flagged and not tuned further; the 13 gold-error rows and the 4 rows the screen
> reviewer disagreed with are removed from TRAIN. Use it for experiments only; a model trained on it is not
> releasable unless a later, separately reviewed round replaces it. See `status.json`.

Training rows for task families where Decision 2.0 has no training data: faithfulness, spam / fraud, tool-call
appropriateness and function selection / relevance, stance, sarcasm, original-vs-perturbed poetry, entity-level
financial sentiment, product-search relevance and commonsense multiple choice. **No row of the Decision Index suite
is included:** every candidate group with an exact, normalized or 13-gram match to any suite row was dropped.
Preregistration `v2/data/records/ib1-prereg-2026-10-01.md` with amendment 1 (branch `xunzhuo/decision-2-training`).

**In-distribution disclosure.** `w2c` (When2Call `train_pref`) and `isarc` (iSarcasmEval `train`) are the train splits
of two datasets whose test files the Index uses. Their test files were never downloaded or read, and every row
matching an Index row was dropped. Leave the two families out to measure transfer from the other families alone.

## Files

| File | Rows | Groups | Native tokens (Qwen3.5-0.8B-Base) |
| --- | ---: | ---: | ---: |
| `ib1.train.jsonl` (Choice 32,182, Noul 5,132) | 37,314 | 22,669 | 10,468,429 |
| `ib1.dev.jsonl` (split `select`; diagnostic only, never selection) | 2,664 | 1,912 | 509,711 |

Also: per-row tokens (`*.tokens.jsonl`), freeze manifests, `build.json` (candidate build), `final.json` (every drop
by reason), `stats.json` (per-family sizes, labels, balance), `isolation.json`, `audits/` (count-only receipts),
`license-registry-ib1.json`, `status.json` and `registry.json` (SHA-256 of every file).

## Families, sources and licences

| Family | Target family | Type | TRAIN | DEV | Source (pin) | Licence | Gold |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| `sumedit` | faithfulness | Noul | 1,280 | 44 | SummEdits `Salesforce/summedits@ce0c479a` (billsum, sales call / email, Shakespeare domains) | CC BY 4.0 | human-verified consistent / inconsistent summaries |
| `sms` | spam / fraud | Noul | 1,152 | 62 | SMS Spam Collection (UCI 228) `ucirvine/sms_spam@cae486f9` | CC BY 4.0 | curated spam / ham |
| `w2c` | tool-call appropriateness (in-distribution) | Choice (2) | 483 | 0 | When2Call `nvidia/When2Call@0582f774` `train_pref` | CC BY 4.0 | preferred vs dispreferred behaviour |
| `snips_sel` | function selection | Choice (4) | 2,673 | 310 | SNIPS NLU benchmark `sonos/nlu-benchmark@b86ac7f1` | CC0 1.0 | crowd utterances for a known intent |
| `snips_rel` | function relevance | Noul | 2,700 | 16 | same | CC0 1.0 | intent's function offered or not |
| `args` | stance | Choice (2) | 5,192 | 374 | args.me `webis/args_me@042bdf64` (idebate, debatewise, debatepedia) | CC BY 4.0 | portal pro / con structure; one of each per statement |
| `isarc` | sarcasm (in-distribution) | Choice (2) | 874 | 82 | iSarcasmEval `iabufarha/iSarcasmEval@dfc708b5` `train` (en 464, ar 410) | MIT | author's sarcastic tweet vs author's rephrase |
| `poem` | arts (original vs perturbed) | Choice (2) | 3,584 | 411 | Gutenberg Poetry Corpus `biglam/gutenberg-poetry-corpus@fcd42e24` | CC0 1.0 | original stanza vs a one-word rhyme-breaking substitution |
| `sentfin` | entity-level financial sentiment | Choice (3) | 7,158 | 744 | SEntFiN 1.0 `pyRis/SEntFiN@eba43610` | MIT | annotator entity decisions |
| `wands` | product-search relevance | Choice (3) | 6,357 | 0 | WANDS `wayfair/WANDS@3b74dcf4` | MIT | annotator Exact / Partial / Irrelevant |
| `copa` | commonsense | Choice (2) | 878 | 108 | Balanced COPA `pkavumba/balanced-copa@813bd03c` | CC BY 4.0 | COPA alternatives |
| `csqa` | commonsense MCQ | Choice (5) | 4,983 | 513 | CommonsenseQA `tau/commonsense_qa@94630fe3` train | MIT | crowd answer key |

Attributions (full lines in `license-registry-ib1.json`): Laban et al. 2023 (SummEdits; summary edits generated with
an LLM and human-verified); Almeida, Gómez Hidalgo and Yamakami 2011 (SMS Spam Collection); Ross et al. 2025
(When2Call; generated with Mixtral-8x22B-Instruct); Coucke et al. 2018 (SNIPS; function schemas written for IB1);
Ajjour et al. 2019 (args.me); Abu Farha et al. 2022 (iSarcasmEval); Allison Parrish (Gutenberg Poetry Corpus;
perturbations constructed for IB1); Sinha et al. 2022 (SEntFiN); Chen et al. 2022 (WANDS, Wayfair); Kavumba et al.
2019 and Roemmele, Bejan and Gordon 2011 (Balanced COPA); Talmor et al. 2019 (CommonsenseQA). No source is NC or
ShareAlike. CommonsenseQA is already in earlier Decision 2.0 mixtures (disclosed lineage).

Instructions and option descriptions are English; content stays in its language (Arabic only in `isarc`).

Families admitted but removed before the review: `expqa` (ExpertQA: the rule found no "unsupported" claim with
evidence text), and by the shortcut gate `maud` (MAUD; option-only .655 vs .430), `medmcqa` (MedMCQA; option-only
.342 vs .254), `procb` (ProcessBench; state-removed .500 vs a cross-validated majority of .430) and `wanli` (WANLI;
hypothesis-only .440 vs .320). Their target families (contracts, knowledge MCQ, math verification, adversarial NLI)
are not covered.

## Construction and audits (counts only)

- **Index rows (G0):** text leaves of every suite row (selected and added rows) against IB1's data leaves,
  NFKC + casefold + `\w+` tokens: raw equality (≥ 20 characters), whole-leaf equality (≥ 6 tokens), sentence or leaf
  equality (≥ 8 tokens), any shared word 13-gram. 3,273 candidate groups were dropped whole, 2,849 of them When2Call
  train rows that share tool schemas with Index tool rows. Positive controls: 2,000 / 2,000 suite rows rendered as
  candidates and 2,000 / 2,000 perturbed copies (case, spacing, punctuation, a prepended sentence) were flagged.
- **Overlap with held-out panels (G2):** E / S / L / N rules against PI-v4 (quarantining roles incl. typed FINAL /
  DEV, SELECT / CAL, CSS15 + pilot, public 231, mlx-diag, Decision Bench v4) and PI-ib1 (HT-DEV v1, HT-DEV v2,
  Score5-DEV, score5t-dev fit + check, hs1-dev, PN1 dev, CAL698, HR2 DEV): 530 groups dropped whole; DEV groups near
  any TRAIN group dropped (1,064).
- **Names (G1) and C1 (G3):** no IB1 source or parent appears in the C1 registry or the panel records; the C1
  source-term scan of the raw files and candidates found none. The custodian C1 content recheck is required before any
  C1-scored model trained on IB1.
- **Shortcut gate (G4):** group-disjoint 5 folds, gated views ≤ majority + 0.05: the 12 families above pass.
- **Balance (G5):** Noul families exactly 50 / 50; fixed-label Choice families equal per class; rotated families
  near-uniform gold position; A / B families with gold-A and gold-longer shares 0.45–0.55. **Isolation (G7):** no
  shared id, group or input with IB1 DEV, HS1 dev, PN1 dev or HR2 DEV. **Leak guard:** 12 `w2c` rows with
  IPv4-like strings dropped.
- **Blind review, two stages** (fresh AI reviewers of one model family; not human annotation): a screen of 10 rows
  per family by one reviewer (4 disagreements in 120; no family dropped), then a fresh decisive sample of 18 rows per
  family with two reviewers and an adjudicator on 3 splits. Errors by family: `sentfin` 3 (gold neutral, reviewers
  positive), `sumedit` 3 (gold consistent, reviewers found an unsupported detail; all Shakespeare), `wands` 3 (Partial
  boundary), `args`, `csqa`, `isarc`, `w2c` 1 each, the rest 0. Reviewers agreed with gold on 94.0% / 95.4% of rows;
  κ .985 between reviewers.

## Use

Add `ib1.train.jsonl` as one block against a matched-token control (the same base mixture plus the same number of
tokens from its own rows), screen with HT-DEV v2, and read `ib1.dev.jsonl` per family as a diagnostic only. Measure
Index effects privately on frozen packages. Because IB1 is not release-safe, a successor trained on it cannot be
released.
