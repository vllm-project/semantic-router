# Decision 2.0 training data — v2 arms (private)

This folder adds separately ablatable **data arms** for Decision 2.0 on top of the
original rights-clean control at the repository root (TRAIN 7,455 / SELECT 700 /
CAL 700, unchanged). Every arm has a frozen content hash (SHA-256 of the canonical
JSONL: rows sorted by `id`, one canonical JSON object per line) and a manifest with
counts by decision type, language, source, family, Score level count and grade,
Choice option count and gold position, Noul balance and native token totals.

Rows follow the training contract used by the Decision 2.0 trainers (`state`,
`instructions`, `options[{key, description}]`, zero-based `label`, `task_type`,
`family`, `group_id`, `language`, `split`, `source`, `evaluation_role`,
`render_template`, `audit_metadata`, `input_sha256`). Replay rows add
`teacher_probs` keyed by option key.

Rules, audits and the experiment plan are versioned in the source branch
`xunzhuo/decision-2-training-data` under `src/training/decision2/v2/data/records/`
(`data-arms-v1-prereg-2026-09-28.md`, its amendment 1, `arms-v1-2026-09-28.md`,
`experiment-matrix-v1-2026-09-28.md`). `v2/registry.json` lists every file with its
SHA-256.

## Admission (summary)

- Permissive rights only; attribution and share-alike obligations per source are in
  each manifest's licence table. Project-generated rows contain no third-party text.
- Whole-group isolation from SELECT, CAL and every evaluation or development panel,
  and source-level isolation from the evaluation panels' sources.
- Overlap scan against a 39-role, 40,857-row input-only protected inventory (exact,
  short-leaf near, long-leaf windowed near, rare n-gram containment) plus an
  embedding scan; flagged groups are removed from every new arm.
- Shortcut gates: state-removed, option-only and (where applicable) hypothesis-only
  baselines must stay within majority + 5 points (group-disjoint 5-fold).
- Held-out slices (`aho.jsonl`, `split=select`) are diagnostics only; checkpoint
  selection stays on the root SELECT.

## Contents of this revision

| Path | What |
| --- | --- |
| `arms/A0/` | Manifest and audits for the root TRAIN (rows not duplicated) |
| `arms/A0s/` | A0 without FLUTE and without protected-overlap groups |
| `arms/A0p/` | One fixed-derangement option-order copy of each eligible A0 Choice row |
| `replay/RP-v1q/` | Frozen replay prompt set (training-row form, gold labels) |
| `replay/R2/<tier>/` | Own Decision 1.0 teacher distributions on RP-v1q (Kai, Lex, Eos, Sol, Nox, Lux) with per-type teacher-quality reports |
| `audits/` | Aggregate overlap, embedding and shortcut receipts (no text, no ids) |
| `protected-inventory/` | Aggregate receipt of the protected inventory used for scans |

Private access does not waive upstream terms. Keep this dataset private.
