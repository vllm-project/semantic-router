# HR2: licence-clean human-rated decision data (`m5/hr2/`)

> **Status: NOT RELEASE-SAFE.** The preregistered blind label review failed: 13 gold errors in 216 reviewed TRAIN
> rows, 6.02% (exact 95% CI 3.24–10.07%), population-weighted 6.50% (bootstrap 95% CI 3.09–10.40%), against the
> bars of ≤ 5.0% with an upper bound ≤ 8.0%. No family reached the per-family failure count, so the fix rule had
> nothing to drop. By the preregistration, HR2 is published flagged and not tuned further; the 13 rows judged gold
> errors are removed from TRAIN. Use it for experiments only; a model trained on it is not releasable unless a
> later, separately reviewed round replaces it. See `status.json`.

Human judgments from sources that no Decision 2.0 arm, panel or Decision 1.0 lineage has used, converted to the
native interfaces: preferences → Choice, yes/no labels → Noul, ratings → Score. Preregistration
`v2/data/records/hr2-prereg-2026-09-30.md` with amendments 1 and 2 (branch `xunzhuo/decision-2-training`).

## Files

| File | Rows | Groups | Native tokens (Qwen3.5-0.8B-Base) |
| --- | ---: | ---: | ---: |
| `hr2.train.jsonl` | 27,725 | 22,353 | 16,802,442 |
| `hr2.dev.jsonl` (split `select`; diagnostic only, never selection) | 1,615 | 1,322 | 561,317 |

Also: per-row tokens (`*.tokens.jsonl`), freeze manifests, `build.json` (candidate build), `final.json` (every drop
by reason), `stats.json` (per-family sizes, labels, balance), `isolation.json`, `audits/` (count-only receipts),
`license-registry-hr2.json`, `status.json` and `registry.json` (SHA-256 of every file).

## Families, sources and licences

| Family | Type | TRAIN | DEV | Source (pin) | Licence | Human judgment |
| --- | --- | ---: | ---: | --- | --- | --- |
| `hs3_pref` | Choice (2) | 7,119 | 263 | HelpSteer3 `nvidia/HelpSteer3@f6d14577` preference TRAIN | CC BY 4.0 | ≥ 2 annotators, strict agreement (overall preference ±2 or ±3, every individual score the same sign) |
| `hs3_help` | Score (5) | 4,027 | 126 | HelpSteer3 feedback TRAIN | CC BY 4.0 | 3 helpfulness ratings, max − min ≤ 1, gold = median |
| `eth_util` | Choice (2) | 3,108 | 354 | ETHICS `hendrycks/ethics@b8b47c58` utilitarianism TRAIN | MIT | MTurk rankings (more pleasant scenario) |
| `eth_cs` | Noul | 1,084 | 94 | ETHICS commonsense TRAIN, short split | MIT | MTurk labels |
| `eth_deon` | Noul | 1,094 | 100 | ETHICS deontology TRAIN | MIT | MTurk labels |
| `eth_just` | Noul | 1,080 | 106 | ETHICS justice TRAIN | MIT | MTurk labels |
| `prm_step` | Noul | 3,562 | 60 | PRM800K `openai/prm800k@7ecc7947` phase 1 + 2 TRAIN | MIT | per-step human ratings (+1 / −1; 0 excluded) |
| `vitc` | Noul | 3,526 | 262 | VitaminC `tals/vitaminc@be6febb7` TRAIN, real revisions | CC BY-SA 3.0 | annotator-written claims and verdicts |
| `allegro` | Score (5) | 3,125 | 250 | Allegro Reviews (KLEJ) `allegro/klej-allegro-reviews@b9ca796d` | CC BY-SA 4.0 | the review author's star rating |

Attributions (full lines in `license-registry-hr2.json`): Wang et al. 2025 (HelpSteer3; responses by
commercially-permissive LLMs, annotations by Scale AI and Translated); Hendrycks et al., ICLR 2021 (ETHICS); Lightman
et al. 2023 (PRM800K; problems from MATH, Hendrycks et al. 2021; steps written by a model and rated by humans);
Schuster, Fisch and Barzilay, NAACL 2021 (VitaminC; Wikipedia-derived); Rybak et al., ACL 2020 (KLEJ, Allegro
Reviews). **Share-alike:** rows of `vitc` (CC BY-SA 3.0) and `allegro` (CC BY-SA 4.0) are derived works under the same
licence. HelpSteer2 (`990b2711`, CC BY 4.0) was read only to drop HelpSteer3 prompts shared with it; no HelpSteer2
row is included.

TRAIN languages: en 22,191, pl 3,183, zh 879, ko 320, ja 192, fr 177, es 158, it 138, ru 111, pt 109, de 107, vi 60,
nl 56, id 44 (non-English rows other than Polish are HelpSteer3 multilingual). Instructions and option descriptions
are English.

Two admitted families were removed by the preregistered shortcut gate: KB-BoolQ (Korean; the question alone predicts
the answer, .559 vs a .534 bar) and IndoNLI (Indonesian; hypothesis-only .613 vs .535).

## Construction and audits (counts only)

- Labels (prereg §2 and amendment 1): ties, slight preferences, split annotators and low-agreement ratings are
  excluded; repeated upstream items must agree on every copy or are dropped. Noul families are exactly 50 / 50
  (PRM800K per step-index bucket), Choice gold position and gold-is-longer shares are 0.45–0.55, `hs3_help` has no
  level above 30%, `allegro` levels are equal.
- Groups: HelpSteer3 by the normalized first user turn (shared across both families), PRM800K by problem, VitaminC by
  Wikipedia page, ETHICS by scenario prefix, Allegro by review text. DEV is 1 in 10 groups by salted hash.
- Overlap (E / S / L / N rules) against every panel: PI-v4 (58 quarantining roles, incl. typed FINAL / DEV,
  SELECT / CAL, CSS15 + pilot, public 231, mlx-diag, Decision Bench v4) and PI-hr2 (HT-DEV v1, HT-DEV v2, Score5-DEV,
  score5t-dev fit + check, hs1-dev, PN1 dev, CAL698). 1,202 candidate groups with a hit were dropped whole (TRAIN and
  DEV). DEV groups near any TRAIN group were dropped (1,180). Report-only hits on training roles are disclosed in
  `audits/`: 3,083 final groups share an n-gram with earlier training rows, 3,055 of them by the N rule only (mostly
  HelpSteer3 vs OASST1-derived A7q TRAIN rows).
- C1: no C1 registry source or parent; the C1 source-term scan of the raw files and candidates found none. The C1
  custodian content recheck is required before any C1-scored model trained on HR2.
- Shortcut gate (group-disjoint 5 folds, gated views ≤ majority + 0.05): 9 families pass; VitaminC claim-only is at
  chance (.501).
- Isolation: no shared id, group or input between HR2 TRAIN and HR2 DEV, HS1 dev or PN1 dev.
- Blind review: 216 TRAIN rows (24 per family), two independent AI reviewers and an adjudicator on splits (one model
  family; not human annotation). Errors by family: VitaminC 4, `hs3_help` 3, `eth_just` 2, `allegro`, `eth_cs`,
  `hs3_pref`, `prm_step` 1 each, `eth_deon` and `eth_util` 0. Reviewers agreed with gold on 93.5% / 94.0% of rows.

## Use

Add `hr2.train.jsonl` as one block against a matched-token control (the same base mixture plus the same number of
tokens from its own rows), screen with HT-DEV v2, and read `hr2.dev.jsonl` per family as a diagnostic only.
Formal CSS15 decides human transfer. Because HR2 is not release-safe, a successor trained on it cannot be released.
