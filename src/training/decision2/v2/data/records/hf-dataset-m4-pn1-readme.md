# `m4/pn1/`: PN1, a paraphrase-style non-English Noul arm and dev slice (research & data Milestone 4)

This tree is private. Private access does not waive upstream terms: the sentences are CC BY 2.0 FR (CC0 1.0 where
the contributor opted in), and attribution is required. Keep this dataset private.

## Revision PN1-r2 (2026-09-30): the TRAIN file was replaced after a blind label review

`arms/pn1.train.jsonl` is now **PN1-r2**: 4,364 rows, `c1cec06bd5f1caadb7631338796054aae120cbf8b17dcb0bbf4c694470c52cfb`.

- **Why.** A blind review of 224 rows of the first TRAIN (4,888 rows, `f6f6a531…`, still available at revision
  `5ad36287`) found 12 gold errors: 5.36%, exact 95% CI [2.80%, 9.17%]. That failed the preregistered bar.
  - Every error was a false "no".
  - Most came from `pn-near` pairs in es, fr, ar, ru and ko, which were really paraphrases.
  - Two came from Russian `pn-name` role swaps, where case marking keeps the roles unchanged.
- **Changes.** PN1-r2 drops every `pn-near` row in es, fr, ar, ru and ko, every Russian `pn-name` row, and the 12
  reviewed errors. Labels are then rebalanced to exactly 50/50 per language × construction group.
  - TRAIN now holds ja 2,236, zh 948, de 822, ru 148 (twins only), ar 108 and ko 102 (swap rows only).
  - es and fr are no longer in TRAIN.
- **Certification.** A fresh blind review of 204 other rows found 1.96% label error, exact 95% CI [0.54%, 4.94%].
  The population-weighted error is 0.73%. All 4 errors are generated `pn-twin` no-edits (zh 1, ko 3).
- **Other audits.**
  - House embedding scan: 0 groups ≥ 0.93 and 0 in the review band.
  - Isolation against the H7 / H8 slices: PASS.
  - A4 overlap learner on PN1-r2: at most .485 per language.
- **The dev slice is unchanged.** Its near rows in es, fr, ar, ru and ko and its Russian name swaps use the
  constructions dropped from TRAIN, so some of their "no" labels may be wrong. The review estimated 20% for those
  TRAIN constructions.
- **Receipts:** `manifests/self-check-r2.json` and `audits/dq/`.

## What it is

Native Noul yes/no rows ("do these two sentences mean the same?"): English instructions, English No/Yes options,
and two native-language sentences in the state.

- **Languages:** TRAIN (PN1-r2) ar, de, ja, ko, ru, zh; the dev slice also has es and fr.
- **Label balance:** exactly 50/50 per language (PN1-r2: per language × construction group).
- **Constructions (the row `family`):**
  - `pn-hop` (yes): two Tatoeba sentences that share a translation.
  - `pn-near` (no): high-overlap Tatoeba near-misses.
  - `pn-name`: rule-based name swaps; coordination swaps are yes and role swaps are no.
  - `pn-twin`: a seed sentence plus two minimal edits written by Qwen3.5-27B (Apache-2.0). A meaning-preserving edit
    is yes; a word-swap edit that changes the meaning is no.
- **Label filter:** every label is kept only if Qwen3.8-27B (Apache-2.0) agrees with it. Every edited sentence also
  passed a fluency check.
- **Overlap control:** yes and no rows are 1:1 caliper-matched on eight overlap features, so lexical overlap does not
  predict the label.

## Files

| Path | Rows | SHA-256 |
| --- | --- | --- |
| `arms/pn1.train.jsonl` (PN1-r2) | 4,364 | `c1cec06bd5f1caadb7631338796054aae120cbf8b17dcb0bbf4c694470c52cfb` |
| `dev/pn1.dev.jsonl` (split `select`) | 1,974 | `c3b68ac113fc1f2afb5554530112f472a9c62f7997285bcaf89a6b655bb7f4cf` |
| `dev/pn1.dev.prompts.jsonl` (gold-free, formal runner input) | 1,974 | `79dbf9996ad09caa622c4a9ea2f28075ddf80a5ae1ba9db8cd633573621bc14a` |
| `dev/pn1.dev.gold.jsonl` | 1,974 | `e0d3e57cbba88139ecd326f7837bff1374090960a3d6d21e340140d491d4be5e` |
| `attribution/attribution.tsv` (sentence id, language, username, licence, edited) | | `293d0491fc2a9cd6cc40dd48197e1066606670c2a3eea60a79a485354142e5a1` |

`manifests/` holds the freeze manifests, `build-manifest.json` (`34a19252…`) and the Tatoeba `source-manifest.json`.
`audits/` holds the overlap, isolation and drop receipts. `validity/` holds the N4XF and Nox 1.0 dev predictions and
`validity-final.json`.

## Card lines

- **Credit:** "PN1 contains sentences from Tatoeba (https://tatoeba.org) by its contributors, CC BY 2.0 FR (export
  2026-09-26); per-sentence authors are listed in the dataset's attribution file. Some sentences were changed (word
  swaps and reorderings, by rule or by Qwen3.5-27B, Apache-2.0); labels were checked with Qwen3.8-27B (Apache-2.0)."
- **Independence:** PN1 shares no source with mlx-diag (PAWS-X / PAWS, MASSIVE, XNLI) or with the C1 source registry.

Records: `src/training/decision2/v2/data/records/m4-pn1-results-2026-09-29.md` and, for PN1-r2,
`m4-dq-results-2026-09-30.md`.
