# `m3/arms/` — research & data Milestone 3b gap arms H7 and H8 (private)

Two new human data arms built with the data-arms v2 framework, frozen like the `m2/arms/` arms:
content hash = SHA-256 of the canonical JSONL (rows sorted by `id`). Rows follow the Decision 2.0
training contract. They carry **gold labels only**: M3b produced no teacher targets for them.

| Arm | Content | Types | Languages |
| --- | --- | --- | --- |
| `H7` | Long evidence. HoVer claims over page-scale Wikipedia paragraph sets: answerability twins (Noul) and fact-coverage levels (Score, 3–5 levels). Natural Questions page-window answer-removal twins (Noul). | Noul, Score | en |
| `H8` | Multilingual. TyDi QA page-window removal twins; MIRACL relevance and judged-pool twins (es, fa, fr, hi, zh); JCommonsenseQA (Choice, 5 options); SentiMix Spanglish sentiment (Score, 3 levels). | Choice, Noul, Score | 17 |

Per arm: `train.jsonl`, `aho.jsonl` (held-out diagnostic slice, split `select`; never for
checkpoint selection), `train.tokens.jsonl` / `aho.tokens.jsonl` (`{id, native, kai}` per row:
Qwen3.5-0.8B-Base native encode and raw Kai-0.6B), `train.manifest.json` / `aho.manifest.json`
(freeze manifests: counts, token totals, licence table), `build.json`, `stats.json` (per-slice and
per-family counts, long-evidence share, gate results), and the quarantine, gate and held-out dedup
reports. Each arm also has a sealed slice (SHO) that is never uploaded; only its row count and hash
appear in `stats.json`. `audits/` holds the aggregate overlap, shortcut, length-baseline,
embedding and A7k receipts. `protected-inventory/` holds the PI-v4 receipts. `registry.json` lists
every file here with its SHA-256. Token files join to rows by `id`.

**Gates** (preregistration amendment 2, section 2):

- Licences: `license-registry-m3b.json`.
- Sources checked against the evaluation track's C1 source registry.
- Group, id and input isolation from every published arm, SELECT and CAL.
- Four-method lexical overlap against PI-v4, with whole-group quarantine. PI-v4 is the
  evaluation/development panels, the v1 held-out slices and the A7 held-out slices. A7 TRAIN and
  the v2 AHO slices are report-only. A group is removed when either the full PI-v4 scan or a scan
  of its quarantining roles alone flags it against a quarantining role. The report-only rows would
  otherwise hide some of those matches.
- Embedding scan with Qwen3-Embedding-0.6B (quarantine at cosine ≥ 0.93).
- Shortcut gates per (source, family, type) cell: state-removed and option-only at majority + 5
  points.
- Held-out dedup against TRAIN.
- An 8,192-token native budget.

**Disclosure.** The v2 AHO slices are report-only. For a model trained with H7 or H8, the H1, H3,
H5, H6 and E11 AHO readouts are therefore passage-familiar: they share Wikipedia paragraphs with
these arms.

Rules and results: branch `xunzhuo/decision-2-training-data`,
`src/training/decision2/v2/data/records/` (`m3b-prereg-amendment-2-2026-09-28.md`,
`m3b-gap-sources-2026-09-28.md`). Private access does not waive upstream terms (CC BY-SA 3.0/4.0
share-alike for HoVer, Natural Questions, JCommonsenseQA and Wikipedia text; Apache-2.0 for TyDi QA
and MIRACL; CC BY 4.0 for SentiMix). Keep this dataset private.
