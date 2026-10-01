# IB1: licence-clean breadth data for the Decision Index task families (`m6/ib1/`, round 2)

> **Status: NOT RELEASE-SAFE.** Round 2 dropped the three boundary constructions behind most round-1 errors and was
> reviewed again by fresh blind reviewers. The decisive review still failed:
>
> - 12 gold errors in 225 reviewed TRAIN rows: 5.33% (exact 95% CI 2.79–9.13%);
> - population-weighted 5.97% (bootstrap 95% CI 2.61–9.77%);
> - `wands` 4 of 19, above the per-family bar.
>
> The bars are ≤ 5.0% with an upper bound ≤ 8.0%, a weighted error ≤ 5.0%, and no family with an exact 95% lower bound
> above 5%. In 5 of the 12 errors, both reviewers chose the class that round 2 no longer uses as gold. That class is
> still listed as an option (`Partial` in `wands`, neutral in `sentfin`). The 12 rows are removed from TRAIN. Use IB1
> for experiments only: a model trained on it is not releasable. See `status.json`. Round 1 is the previous revision
> of this path.

Training rows for task families where Decision 2.0 has no training data:

- faithfulness;
- spam and fraud;
- tool-call appropriateness, and function selection and relevance;
- stance;
- sarcasm;
- original vs perturbed poetry;
- entity-level financial sentiment;
- product-search relevance;
- commonsense multiple choice.

**No row of the Decision Index suite is included:** every candidate group with an exact, normalized or 13-gram match
to any suite row was dropped. The preregistration is `v2/data/records/ib1-prereg-2026-10-01.md` with amendments 1
and 2 (branch `xunzhuo/decision-2-training`).

**In-distribution disclosure.** `w2c` (When2Call `train_pref`) and `isarc` (iSarcasmEval `train`) are the train splits
of two datasets whose test files the Index uses. Their test files were never downloaded or read, and every row
matching an Index row was dropped. To measure transfer from the other families alone, leave these two families out.

## Round 2 changes (amendment 2)

Round 2 drops three constructions from TRAIN and DEV and changes nothing else:

- `sentfin` rows with neutral gold;
- the SummEdits Shakespeare domain;
- `wands` rows with `Partial` gold.

The option sets are unchanged. So in `sentfin` the neutral option is never gold, and in `wands` the `Partial` option is
never gold. This gives a trained model a prior against those classes, so a track that needs them must get them from
other data.

## Files

| File | Rows | Groups | Native tokens (Qwen3.5-0.8B-Base) |
| --- | ---: | ---: | ---: |
| `ib1.train.jsonl` (Choice 27,495, Noul 4,428) | 31,923 | 20,755 | 7,880,678 |
| `ib1.dev.jsonl` (split `select`; diagnostic only, never selection) | 2,366 | 1,708 | 406,966 |

The path also holds:

- per-row tokens (`*.tokens.jsonl`) and the freeze manifests;
- `build.json` (candidate build) and `final.json` (every drop, by reason);
- `stats.json` (per-family sizes, labels and balance) and `isolation.json`;
- `audits/` (count-only receipts, including `audits/rescan/` for the round-2 re-scans);
- `license-registry-ib1.json`, `status.json` and `registry.json` (the SHA-256 of every file).

## Families, sources and licences

| Family | Target family | Type | TRAIN | DEV | Source (pin) | Licence | Gold |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| `sumedit` | faithfulness | Noul | 576 | 0 | SummEdits `Salesforce/summedits@ce0c479a` (billsum, sales call, sales email) | CC BY 4.0 | human-verified consistent / inconsistent summaries |
| `sms` | spam / fraud | Noul | 1,152 | 62 | SMS Spam Collection (UCI 228) `ucirvine/sms_spam@cae486f9` | CC BY 4.0 | curated spam / ham |
| `w2c` | tool-call appropriateness (in-distribution) | Choice (2) | 476 | 0 | When2Call `nvidia/When2Call@0582f774` `train_pref` | CC BY 4.0 | preferred vs dispreferred behaviour |
| `snips_sel` | function selection | Choice (4) | 2,672 | 310 | SNIPS NLU benchmark `sonos/nlu-benchmark@b86ac7f1` | CC0 1.0 | crowd utterances for a known intent |
| `snips_rel` | function relevance | Noul | 2,700 | 12 | same | CC0 1.0 | intent's function offered or not |
| `args` | stance | Choice (2) | 5,184 | 374 | args.me `webis/args_me@042bdf64` (idebate, debatewise, debatepedia) | CC BY 4.0 | portal pro / con structure; one of each per statement |
| `isarc` | sarcasm (in-distribution) | Choice (2) | 874 | 82 | iSarcasmEval `iabufarha/iSarcasmEval@dfc708b5` `train` (en 464, ar 410) | MIT | author's sarcastic tweet vs author's rephrase |
| `poem` | arts (original vs perturbed) | Choice (2) | 3,584 | 411 | Gutenberg Poetry Corpus `biglam/gutenberg-poetry-corpus@fcd42e24` | CC0 1.0 | original stanza vs a one-word rhyme-breaking substitution |
| `sentfin` | entity-level financial sentiment | Choice (3; gold negative / positive) | 4,778 | 494 | SEntFiN 1.0 `pyRis/SEntFiN@eba43610` | MIT | annotator entity decisions |
| `wands` | product-search relevance | Choice (3; gold Exact / Irrelevant) | 4,068 | 0 | WANDS `wayfair/WANDS@3b74dcf4` | MIT | annotator Exact / Irrelevant |
| `copa` | commonsense | Choice (2) | 878 | 108 | Balanced COPA `pkavumba/balanced-copa@813bd03c` | CC BY 4.0 | COPA alternatives |
| `csqa` | commonsense MCQ | Choice (5) | 4,981 | 513 | CommonsenseQA `tau/commonsense_qa@94630fe3` train | MIT | crowd answer key |

Attributions (full lines in `license-registry-ib1.json`):

- Laban et al. 2023 (SummEdits; summary edits generated with an LLM and human-verified);
- Almeida, Gómez Hidalgo and Yamakami 2011 (SMS Spam Collection);
- Ross et al. 2025 (When2Call; generated with Mixtral-8x22B-Instruct);
- Coucke et al. 2018 (SNIPS; function schemas written for IB1);
- Ajjour et al. 2019 (args.me);
- Abu Farha et al. 2022 (iSarcasmEval);
- Allison Parrish (Gutenberg Poetry Corpus; perturbations constructed for IB1);
- Sinha et al. 2022 (SEntFiN);
- Chen et al. 2022 (WANDS, Wayfair);
- Kavumba et al. 2019 and Roemmele, Bejan and Gordon 2011 (Balanced COPA);
- Talmor et al. 2019 (CommonsenseQA).

No source is NC or ShareAlike. CommonsenseQA is already in earlier Decision 2.0 mixtures (disclosed lineage).

Instructions and option descriptions are English. Content stays in its own language (Arabic only in `isarc`). DEV has
no `sumedit`, `w2c` or `wands` rows: in those families, every DEV group either sits near a TRAIN group or loses a
whole label class after that rule.

Families admitted but not in IB1:

- `expqa` (ExpertQA): the rule found no "unsupported" claim with evidence text.
- Removed by the shortcut gate:
  - `medmcqa` (MedMCQA): option-only .346 vs .253;
  - `procb` (ProcessBench): state-removed .500 vs a cross-validated majority of .430;
  - `wanli` (WANLI): hypothesis-only .436 vs .320.
- `maud` (MAUD): emptied by the round-2 re-scan.

Their target families (contracts, knowledge MCQ, maths verification, adversarial NLI) are not covered.

## Construction and audits (counts only)

- **Index rows (G0).** The text leaves of every suite row (selected and added rows) are compared with IB1's data
  leaves after NFKC normalisation, casefolding and `\w+` tokenisation. A group is flagged on any of:
  - raw equality (≥ 20 characters);
  - whole-leaf equality (≥ 6 tokens);
  - sentence or leaf equality (≥ 8 tokens);
  - any shared word 13-gram.

  3,273 candidate groups were dropped whole. The candidates are byte-identical to round 1. The round-2 re-scans of
  the finalized files found no new G0 match. Positive controls (rerun): 2,000 / 2,000 suite rows rendered as
  candidates and 2,000 / 2,000 perturbed copies were flagged.
- **Overlap with held-out panels (G2).** The E / S / L / N rules run against PI-v4 and PI-ib1:
  - PI-v4: quarantining roles including typed FINAL / DEV, SELECT / CAL, CSS15 + pilot, public 231, mlx-diag and
    Decision Bench v4;
  - PI-ib1: HT-DEV v1, HT-DEV v2, Score5-DEV, score5t-dev fit + check, hs1-dev, PN1 dev, CAL698 and HR2 DEV.

  On the candidates, 530 groups were dropped whole and 1,064 DEV groups near TRAIN were dropped. The boilerplate
  exemption depends on the candidate set, so the finalized round-2 files were re-scanned until clean, and every newly
  flagged group was dropped:
  - pass 1: 62 groups and 7 DEV groups;
  - pass 2: 2 groups;
  - pass 3: clean.

  After the review drops, the final files were scanned again: 1 group was dropped, then the scan was clean.
- **Names (G1) and C1 (G3).** No IB1 source or parent appears in the C1 registry or the panel records. The C1
  source-term scan of the raw files, the candidates and every finalized file found none. The custodian C1 content
  recheck is required before any C1-scored model is trained on IB1.
- **Shortcut gate (G4).** Group-disjoint 5 folds, gated views ≤ majority + 0.05, on the round-2 TRAIN file: the
  12 families above pass.
- **Balance (G5).**
  - Noul families: exactly 50 / 50 (`sumedit` per domain).
  - `sentfin` and `wands`: the two gold classes are equal and the dropped class has 0 rows.
  - Rotated families: near-uniform gold position.
  - A / B families: gold-A and gold-longer shares 0.45–0.55.
- **Isolation (G7).** No shared id, group or input with IB1 DEV, HS1 dev, PN1 dev or HR2 DEV.
- **Leak guard.** The 12 round-1 `w2c` rows with IPv4-like strings are dropped; the round-2 final files have none.
- **Blind review (round 2).** Fresh AI reviewers of one model family (not human annotation) were blind to gold,
  source and family. The round-1 stage-S and stage-R rows and their groups were excluded, and the salts were new.
  - Sample: 19 rows per family (16 for `sumedit`, which has 29 groups). Two reviewers answered every row, and an
    adjudicator answered the 4 splits.
  - Errors by family: `wands` 4 (reviewers chose `Partial`), `sumedit` 3, `csqa` 2, and 1 each in `sentfin`
    (reviewers chose neutral), `snips_sel` and `w2c`. The other families had 0.
  - Agreement with gold: 94.2% and 93.3% per reviewer. Inter-reviewer κ .980.

## Use

Add `ib1.train.jsonl` as one block against a matched-token control (the same base mixture plus the same number of
tokens from its own rows). Screen with HT-DEV v2, and read `ib1.dev.jsonl` per family as a diagnostic only. Measure
Index effects privately on frozen packages. IB1 is not release-safe, so a successor trained on it cannot be released.
