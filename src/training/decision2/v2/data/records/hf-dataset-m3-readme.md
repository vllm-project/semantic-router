# `m3/` — research & data Milestone 3a (private)

## `m3/pk1/` — positional option keys (pk1)

Decision 1.0 builders numbered Choice keys `result_<n>` in construction order and then shuffled the
options, so a key number can reveal the gold (A7 amendment 1, rule 7d). pk1 re-keys every such Choice
row `result_0 … result_{K-1}` by display position with `v2.common.option_keys`; option order,
descriptions and labels are unchanged, `input_sha256` is recomputed, and the original keys and input
hash are kept in `audit_metadata.option_key_renumbering`. Every other row is byte-identical to its
source. **Release candidates use pk1 instead of the v1 files.**

| pk1 file | Replaces | Rows | Renumbered |
| --- | --- | ---: | ---: |
| `A0/train.jsonl` | root `rights_clean.train.jsonl` (A0) | 7,455 | 117 |
| `A0s/train.jsonl` | `v2/arms/A0s/train.jsonl` | 7,299 | 117 |
| `A0p/train.jsonl` | `v2/arms/A0p/train.jsonl` | 3,709 | 108 |
| `RP-v1q/train.jsonl` | `v2/replay/RP-v1q/train.jsonl` | 1,517 | 22 |
| `R2/<tier>/replay.jsonl` | `v2/replay/R2/<tier>/replay.jsonl` | 1,517 (Kai, Lex 1,288) | 22 |
| `lux1/A0-train.canonical.jsonl` | `m2/teachers/lux1/A0-train.canonical.jsonl` | 7,455 | 117 |

Teacher targets of renumbered rows were **re-derived on the renumbered prompts**, never moved with the
old keys: own Lux1 on node A with the runtime of the canonical file (64 unchanged control rows
bitwise identical), and the six own-1.0 R2 teachers on node B GPU7 with their Milestone 1 runtimes (42
control rows per tier). `rederive.json` lists the renumbered and control ids; each `receipt.json` /
`report.json` gives counts and hashes. `registry.json` (one level up) lists the SHA-256 of every file.
Join teacher files to rows by `id`; pk1 `input_sha256` values match the pk1 rows.
