# HR2-r2: licence-clean human-rated decision data (`m5/hr2/`, round 2)

> **Status: NOT RELEASE-SAFE.** The preregistered fresh blind label review of HR2-r2 (amendment 4) failed: 13 gold
> errors in 288 reviewed TRAIN rows, 4.51% (exact 95% CI 2.43–7.60%; the pooled bar P1 passes), but the
> population-weighted error is 8.02% (bootstrap 95% CI 3.83–12.82%; bar ≤ 5%) and the HelpSteer3 preference family
> has 7 errors in 48 rows (exact lower bound above 5%). Amendment 4 has no fix rule, so HR2-r2 is published flagged
> and not tuned further; the 13 rows judged gold errors are removed from TRAIN. Use it for experiments only; a model
> trained on it is not releasable unless a later, separately reviewed round replaces it. See `status.json`.

This revision replaces round 1 (`m5/hr2/` at `afc3bc1e`, also not release-safe: 13 / 216 errors, 6.02%). The round-1
files stay at that revision.

**What changed from round 1** (amendment 4, `v2/data/records/hr2-prereg-amendment-4-2026-10-01.md`):

- **Licence:** VitaminC (CC BY-SA 3.0 with per-article Wikipedia terms) and Allegro Reviews (CC BY-SA 4.0, no licence in
  the pinned snapshot) are dropped: ShareAlike is not established as compatible with Apache-2.0 model releases. HR2-r2
  has no ShareAlike source.
- **Construction:** the HelpSteer3 helpfulness ratings (`hs3_help`, absolute 5-level ratings; 3 of round 1's errors)
  are dropped, and so are the PRM800K yes rows that are the last +1 step before the rater's first −1 (362 TRAIN, 8 DEV
  rows). HR2-r2 has no Score family.
- **Overlap recheck:** the overlap scan's boilerplate exemption depends on the candidate set, so the re-scan of the
  smaller r2 files flagged 211 more groups with quarantining hits (three rounds: 163, 41, 7) and 8 DEV groups near
  TRAIN; all were dropped whole before sampling, and the final files re-scan clean.

## Files

| File | Rows | Groups | Native tokens (Qwen3.5-0.8B-Base) |
| --- | ---: | ---: | ---: |
| `hr2.train.jsonl` (Choice 9,983, Noul 6,112) | 16,095 | 15,029 | 10,606,126 |
| `hr2.dev.jsonl` (split `select`; diagnostic only, never selection) | 954 | 891 | 338,713 |

Also: per-row tokens (`*.tokens.jsonl`), freeze manifests, `build.json` (the round-1 candidate build these rows are
taken from), `final.json` (every drop by reason), `stats.json` (per-family sizes, labels, balance), `isolation.json`,
`audits/` (count-only receipts, including the round-1 error analysis and both review rounds),
`license-registry-hr2.json`, `status.json` and `registry.json` (SHA-256 of every file).

## Families, sources and licences

| Family | Type | TRAIN | DEV | Source (pin) | Licence | Human judgment | Review round 2 errors |
| --- | --- | ---: | ---: | --- | --- | --- | ---: |
| `hs3_pref` | Choice (2) | 6,876 | 256 | HelpSteer3 `nvidia/HelpSteer3@f6d14577` preference TRAIN | CC BY 4.0 | ≥ 2 annotators, strict agreement (overall ±2 or ±3, every individual score the same sign) | 7 / 48 |
| `eth_util` | Choice (2) | 3,107 | 354 | ETHICS `hendrycks/ethics@b8b47c58` utilitarianism TRAIN | MIT | MTurk rankings (more pleasant scenario) | 1 / 48 |
| `eth_cs` | Noul | 1,084 | 94 | ETHICS commonsense TRAIN, short split | MIT | MTurk labels | 1 / 48 |
| `eth_deon` | Noul | 1,094 | 100 | ETHICS deontology TRAIN | MIT | MTurk labels | 0 / 48 |
| `eth_just` | Noul | 1,078 | 106 | ETHICS justice TRAIN | MIT | MTurk labels | 1 / 48 |
| `prm_step` | Noul | 2,856 | 44 | PRM800K `openai/prm800k@7ecc7947` phase 1 + 2 TRAIN | MIT | per-step human ratings (+1 / −1; 0 and error-boundary +1 steps excluded) | 3 / 48 |

Attributions (full lines in `license-registry-hr2.json`): Wang et al. 2025 (HelpSteer3; responses by
commercially-permissive LLMs, annotations by Scale AI and Translated); Hendrycks et al., ICLR 2021 (ETHICS); Lightman
et al. 2023 (PRM800K; problems from MATH, Hendrycks et al. 2021; steps written by a model and rated by humans).
HelpSteer2 (`990b2711`, CC BY 4.0) was read only to drop HelpSteer3 prompts shared with it; no HelpSteer2 row is
included.

TRAIN languages: en 14,536, zh 559, ko 218, ja 129, fr 106, es 99, it 92, ru 73, pt 72, de 70, vi 40, nl 36, pl 35,
id 30 (every non-English row is HelpSteer3 multilingual). Instructions and option descriptions are English.

## Construction and audits (counts only)

- Labels (prereg §2, amendments 1 and 4): ties, slight preferences and split annotators are excluded; repeated upstream
  items must agree on every copy or are dropped; PRM800K rating-0 steps and error-boundary +1 steps are excluded. Noul
  families are exactly 50 / 50 (PRM800K per step-index bucket); Choice gold position and gold-is-longer shares are
  0.45–0.55.
- Groups: HelpSteer3 by the normalized first user turn, PRM800K by problem, ETHICS by scenario prefix. DEV is 1 in 10
  groups by salted hash.
- Overlap (E / S / L / N rules) against every panel: PI-v4 (58 quarantining roles, incl. typed FINAL / DEV,
  SELECT / CAL, CSS15 + pilot, public 231, mlx-diag, Decision Bench v4) and PI-hr2 (HT-DEV v1, HT-DEV v2, Score5-DEV,
  score5t-dev fit + check, hs1-dev, PN1 dev, CAL698). 1,413 candidate groups with a hit were dropped whole (1,202 in
  round 1, 211 by the r2 rechecks); 1,188 DEV groups near TRAIN were dropped. The final files re-scan with no
  quarantining hit and no DEV group near TRAIN. Report-only hits on earlier training roles are disclosed in
  `audits/quarantine-recheck-final.public.json` (mostly HelpSteer3 groups vs OASST1-derived A7q TRAIN rows: 2,097).
- C1: no C1 registry source or parent; the C1 source-term scan of the raw files and the r2 files found none. The C1
  custodian content recheck is required before any C1-scored model trained on HR2-r2.
- Shortcut gate (group-disjoint 5 folds, gated views ≤ majority + 0.05): all 6 families pass on the r2 files.
- Isolation: no shared id, group or input between HR2-r2 TRAIN and HR2-r2 DEV, HS1 dev or PN1 dev.
- Blind review: round 2 drew 288 TRAIN rows (48 per family) that share no row or group with round 1; two independent
  AI reviewers (R1 / R2 agree with gold on 95.1% / 95.5%, with each other on 98.96%, κ .986) and an adjudicator on
  the 3 splits; one model family, so this is AI adjudication, not human annotation. All 7 HelpSteer3 preference
  errors had medium or low reviewer confidence; 6 of them have overall margin 2 (6 / 37 reviewed margin-2 rows, 1 / 11
  margin-3 rows). Over both rounds: HelpSteer3 preferences 8 / 72, PRM800K 4 / 72, ETHICS 6 / 288.

## Use

Add `hr2.train.jsonl` as one block against a matched-token control (the same base mixture plus the same number of
tokens from its own rows), screen with HT-DEV v2, and read `hr2.dev.jsonl` per family as a diagnostic only. Formal
CSS15 decides human transfer. Because HR2-r2 is not release-safe, a successor trained on it cannot be released.
