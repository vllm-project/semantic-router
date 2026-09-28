# Decision 2.0 training data — `v2/a7/`: own Decision 1.0 corpora (private)

This folder holds data arm **A7**: the decoder-family training corpora of our own
Decision 1.0 models (Sol, Nox, Eos, Lux and Kai's shared Choice/Score TRAIN),
re-audited and split into separately ablatable sub-arms. It is owned by the A7
track and never modifies the root files, `v2/arms/`, `v2/replay/` or
`v2/registry.json` of the research & data track. `a7/registry.json` lists every
file here with its SHA-256.

| Sub-arm | Content | Labels |
| --- | --- | --- |
| `A7h` | Natural reading/inference: Cosmos QA, SNLI, SQuAD 2.0 answerability | publisher human labels |
| `A7m` | MultiNLI government/slate/telephone/travel | publisher human labels |
| `A7g` | Stage4 v2 generated relations, scope, Boolean, ordinal Score (3–8 levels), dense tables, arithmetic, registers, automata (English/Chinese) | program oracles, independently re-verified by the 1.0 builders |
| `A7i` | Public intents: BANKING77, CLINC150 | publisher human labels |
| `A7p` | Stage4 v2 replay of older generator families | program oracles |
| `A7o` | Stage1–3 generated rows not in Stage4 v2 | program oracles |

Each sub-arm has `train.jsonl`, `aho.jsonl` (held-out diagnostic slice, split
`select`; never for checkpoint selection) and freeze manifests with counts by
type, language, source, family, Score level and Choice gold position plus native
token totals. Rows follow the Decision 2.0 training contract (`state`,
`instructions`, `options[{key, description}]`, zero-based `label`, `task_type`,
`family`, `group_id`, `language`, `split`, `source`, `evaluation_role`,
`render_template`, `audit_metadata`, `input_sha256`); `audit_metadata.a7` keeps
the 1.0 source file, original id, original source object and, where rule 7d
renumbered construction-order keys, the original keys.

`views/` reproduce 1.0 training mixtures as lists of A7 ids: `dec10-natural24k`
(Eos, Lux, Kai-admitted and Sol/Nox natural adaptation), `dec10-replay24k` (its
replay-only control), `dec10-semantic24k` (Lux's final-stage variant; its 2,671
re-described rows are not materialized) and `dec10-stage4v2` (Sol/Nox fourth
round). A track replicating an exact 1.0 mixture may train on TRAIN ∪ AHO of a
view and must then not report AHO.

Admission (`admission.json`, `audits/`): training rights and redistribution per
source (`license-registry-a7-v1.json`); whole-group quarantine of any hit against
the 39-role protected inventory (exact, short-leaf near, long-leaf windowed near,
rare n-gram); construction-defect rules (legacy numeric-choice rows excluded,
opaque Noul/Score keys held, construction-order Choice keys renumbered, generated
family × type cells with option-only or state-removed shortcuts dropped); an
8,192-token native budget; isolation from SELECT/CAL, A0 and every published arm.
Human-label shortcut results are disclosed diagnostics, not filters.

Rules and results: branch `xunzhuo/decision-2-training-a7`,
`src/training/decision2/v2/data/a7/records/` (`a7-prereg-2026-09-28.md`,
amendment 1, `a7-inventory-2026-09-28.md`, `a7-arms-v1-2026-09-28.md`).
Private access does not waive upstream terms. Keep this dataset private.
