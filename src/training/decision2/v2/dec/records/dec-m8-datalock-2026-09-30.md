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

## Part 2 — A20r teachers (D1, D2) (locked 2026-09-30 10:30 UTC)

- **Labels:** `ops/m8/m8-label.sh` from the mirror of `a629a6ce2`, node A GPU0 / GPU1 (shared leases
  `owner.dec-m8-label`) and GPU5, 09:57:52–10:12:53Z (896 + 897 + 901 GPU-s = **0.75 GPU-h**). Each shard =
  the first 80 typed-final gold-free prompts + its slice prompts, through A20r's unchanged scored collector
  (`v2.27b.typed_collect_kernel`, image `dbe5f32b`, 32,768 tokens, the package's T = 1 `calibration.json` `518e19cd…`,
  a fresh copy of the frozen cache `f474e2e9…`, `HIP_FORCE_DEV_KERNARG=1`) on the frozen C1 package (manifest
  `0d7953cc…`, identity `2e07451107a2…` = DEV2.0-27B@`5323310` weights).
- **Parity smoke: PASS on all three shards** — 80 / 80 typed-final answers identical to A20r's stored formal run
  (`ea573b2d…`), max probability drift **0.0** (receipts `c1d558c9…`, `2cfe4b48…`, `f3e8458c…`).
- **Predictions** (gold-free): shard 0 `8c387f07…`, shard 1 `f2f87c82…`, shard 2 `f1fb32f9…` (4,265 / 4,265 / 4,264
  slice rows after the 80 smoke rows). Provenance `label/provenance.json` `5f72b740…`.
- **Conversion** (`m8_teacher.py convert`, host CPU, node B; manifest `teacher-a20r/manifest.json` `3f2dea64…`):
  **coverage 12,794 / 12,794 slice rows, 0 failures.**

| Teacher file | Rows | SHA-256 |
| --- | ---: | --- |
| **D1** `teacher-a20r/D1/teacher.jsonl` (every slice row) | 12,794 | `0ead0d1b9e6bca040d8d3aad72ca4af8d9e4211851fb78c72c29ade14a70ffdf` |
| **D2** `teacher-a20r/D2/teacher.jsonl` (human-rated rows only) | 3,467 | `015de0685fb666ff9cbc85d0c3094f519b0d764a93befa8e1f94efd8955a735d` |

**Teacher quality on the slice** (argmax vs gold; mean max probability in brackets):

| Rows | A20r C / N / S | own-Lux C / N / S (C teacher) |
| --- | --- | --- |
| All slice rows (Lux: its 12,018 covered rows) | .893 / .850 / .632 (.861 / .864 / .691) | .834 / .798 / .511 (.863 / .852 / .650) |
| Same rows (the 12,018 Lux-covered rows) | .890 / .859 / .633 | .834 / .798 / .511 |
| Human-rated rows | .882 / .881 / .498 | .807 / .809 / .448 |
| Typed rows | .896 / .845 / .858 | .840 / .796 / .609 |

- A20r and own-Lux pick the same option on .850 / .863 / .684 of the rows both cover, so about one row in seven
  (Choice, Noul) and one in three (Score) gets a different target. A20r's typed-Score advantage (.858 vs .609) is the
  largest single difference; on human-rated Score rows both are near .45–.50 (5-level ratings).
- The same A20r target files serve the 2B / 0.8B decoder worker by row id and input hash where rows overlap
  (COORDINATION 17:05).
