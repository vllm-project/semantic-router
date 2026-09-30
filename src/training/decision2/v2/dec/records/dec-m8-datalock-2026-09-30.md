# Decoder Milestone 8 — data lock (2026-09-30)

Identities of every M8 training input, recorded and pushed before the jobs that use them. The chains re-hash each
TRAIN and teacher file against READY files written only after the part that locks them is committed.
Preregistration: [`dec-m8-prereg-2026-09-30.md`](dec-m8-prereg-2026-09-30.md) (`a629a6ce2`).

## Part 1 — top-up slice, human-rated split, C teacher, label prompts (locked 2026-09-30 09:45 UTC)

Built on node B, CPU only, by `ops/m8/m8-prep.sh` from the exact mirror of `a629a6ce2` (receipts beside every output
under `/data/dev2/runs/dec/m8/`: `data.launch.json`, `exposure.launch.json`, `data/build/manifest.json`
`09803b7b…`).

| Input | SHA-256 |
| --- | --- |
| Base: N4XF's mixture `m4-xl-full-29m` (58,742 rows) | `c7d51219e9f0fdd107b914df5c7e9f81f2b53496bb43a94796857b8ea3fcfa60` |
| Mixture spec `specs/m4-xl-full-29m.json` | `cf31117830d886af417da0b77be19a83ab0063e748da8e242818ab9ef41a06ae` |
| Pool id files: `mx-v2-full-M.ids.jsonl` / `mx-xl-full-r2.ids.jsonl` | `5b8d5d29…` / `7843afb7…` |
| N4XF's own-Lux teacher (C) | `e2ff27ce2fc397c8c203e37133dcfac52ac3d1f172c6f3e9412ec28c2ffe296c` |
| Quarantine `ops/m7/specs/m7-quarantine-groups.json` | 1 group, 3 rows dropped |
| Human-rated rule `ops/m8/specs/m8-human-rated-rule.json` | the 9B M6 `soft_target_rule` verbatim + pool alias `A0s` → `A0s-strict` (test-checked) |
| Tokenizer | Eos / Sol / Nox tokenizer (`363c4a5e` snapshot), native `encode` lengths |

| Output | Rows | Native tokens | SHA-256 |
| --- | ---: | ---: | --- |
| **Slice TRAIN** `slice/train.jsonl` | 12,794 (C / N / S 4,571 / 5,154 / 3,069; 8,025 whole groups) | 6,323,582 (budget 6,200,000; +2.0%) | `bd529767070e929d0c502ab5c7b48d8c0247da0abaff0546b7ae08c48c35fe83` |
| Pools and classes `slice/train.ids.jsonl` | 12,794 | | `6fd9aeca7ee123534a1c5f352af2ca98f7ff44233b711874123f68623d4c1813` |
| **C teacher** `teacher/C/teacher.jsonl` (own-Lux records copied unchanged) | 12,018 (gold-only H7 138, H8 638) | | `7ddaff8499f58ca31d02aa854592787cfe0211fba67319a9260aa69b30de3b22` |
| Label prompts `label/prompts.jsonl` (all slice rows) | 12,794 (0 conversion failures) | | `eeedab987b55c400c763be7189d7860d640ed73875319ca72600d0e4193b57ec` |
| Label shards 0 / 1 / 2 | 4,265 / 4,265 / 4,264 | | `d172dd0d…` / `f6900232…` / `3f73fd27…` |
| Exposure receipt `exposure/exposure-slice.json` (r2 payload `2194716a…`) | `groups: []`, `methods_agree: true` | | `a9a979a9b6f00f6e0ccbe31a861cd7a5fc3dd2cc449a0f1d9417c1eb9291a9d6` |

- **Human-rated rows S** (D2's teacher rows): 3,467 rows, 924,825 tokens (14.6% of the slice's tokens), C / N / S
  805 / 737 / 1,925. By pool: A7q 735, A0s 693 (GoEmotions / SNLI / human-labelled replays), H1 554, A7s 392, H6 362,
  V1:A6h 186, H8 182, V1:A1 142, H5 109, A7k 68, V1:A5 44. **Typed rows**: 9,327 rows, 5,398,757 tokens.
- The slice follows the recipe's composition (strata source × type × language): base 58,739 rows / 29.40M tokens
  (C / N / S 21,400 / 23,070 / 14,269) → slice 21.5% of tokens; languages en 7,165, zh 1,995, ja 582, es 421, hi-en
  373, ko 311, ru 210, de 189, es-en 182, fr 180, th 138, zh-hant 132.
- **Guards, all pass:** `v2.common.eval_only` on every training input and the slice rows; isolation from SELECT700 /
  CAL698 (ids and input hashes); 0 shared states with the gold-free typed DEV, CSS pilot, `hs1-dev`, HT-DEV v2
  (`90cd409a…`) and Score5-typed-DEV (`8e35bfff…`) prompts; C1 registry (`db00d4fe…`, 38 keys): 0 source hits, family
  name `pilot_narrative_reading` reported only (as in M6 / M7); every prompt renders the row's exact prompt text and
  option keys, also after a JSON round trip.
- The slice is a subset of the released model's own TRAIN, so a successor adds no new training text; the quarantined
  HotpotQA group (in the released model's TRAIN) is not in the slice.

## Part 2 — A20r teachers (D1, D2)

Appended when the three label shards and their parity smokes have finished.
