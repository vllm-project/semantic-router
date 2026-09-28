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

**Version `a7-dec10-v3`** (this revision's `A7h`, `A7m`, `A7i`, `A7g`, `A7p`,
`A7o`): the `a7-dec10-v2` files minus every group flagged by a GPU embedding scan
(Qwen3-Embedding-0.6B, quarantine at cosine ≥ 0.93) and a lexical rescreen, both
against PI-v3 (the evaluation/development panels plus the multilingual
diagnostic `mlx-diag` and the v1 arm held-out slices). `admission.json` keeps the
v2 admission and adds a `requarantine` section; `audits/embed.public.json` and
`audits/<sub>/rescreen-overlap.public.json` are the aggregate receipts. Sub-arms
without a removed group keep their v2 bytes.

**Recovered sub-arm `A7r` (version `a7-rec10-v2`; records under
`versions/rec10/`):** the Stage1/Stage2 Noul rows that rule 7b had held because
of opaque option keys, re-keyed `true`/`false` from their option descriptions
only (rule 7e), balanced to a 45–55% true share per family and language, and
screened like the other generated sub-arms (the rubric Score rows failed the
option-only gate and are not included).

**Encoder-family sub-arms (version `a7-enc10-v3`; records under
`versions/enc10/`)**, rebuilt from the pinned upstream files of the sources the
Kai/Lex rosters used, with human labels only and fixed English templates written
in this repository (no 1.0 question, rubric or translated schema text):

| Sub-arm | Content | Labels |
| --- | --- | --- |
| `A7q` | OpenAssistant OASST1 assistant replies in 23 languages: quality, helpfulness, humor, creativity (Score, 5 levels) | mean of human ratings, levels balanced per axis |
| `A7k` | KLUE-STS (ko) and JSTS (ja) sentence similarity (Score, 6 levels), excluding every pair the research & data track uses | mean human similarity |
| `A7s` | AfriSenti Swahili and SentiMix Hinglish sentiment (Score, 3 levels) | publisher human labels |
| `A7x` | MASSIVE intents, 12 locales (Choice among same-scenario intents) — **ablation-only**: same source as the multilingual diagnostic `mlx-diag`; never part of a default mixture | publisher human labels |

`license-registry-a7-v2.json` covers these sources (Apache-2.0, CC BY-SA 4.0,
CC BY 4.0). Status of the GPU embedding scan for `A7r` and the encoder sub-arms
is recorded in `a7-arms-m2-2026-09-28.md` on the branch; until it is recorded
there as passed, `A7r` and the encoder sub-arms are for development runs only.

Rules and results: branch `xunzhuo/decision-2-training-a7`,
`src/training/decision2/v2/data/a7/records/` (`a7-prereg-2026-09-28.md`,
amendments 1–4, `a7-enc10-prereg-2026-09-28.md` with its amendments,
`a7-inventory-2026-09-28.md`, `a7-arms-v1-2026-09-28.md`,
`a7-arms-v3-2026-09-28.md`, `a7-arms-m2-2026-09-28.md`).
Private access does not waive upstream terms. Keep this dataset private.
