# Decoder Milestone 6 — data lock (2026-09-29)

Identities of every M6 training input, recorded and pushed before any M6 training job. The chains re-hash each
TRAIN and teacher file against a `READY` file written only after the lock section that names its hash is committed.
Preregistration: [`dec-m6-prereg-2026-09-29.md`](dec-m6-prereg-2026-09-29.md) (`68d31c358`).

- **Builds:** node B, CPU only, from the exact mirror of `6c482ce0e285dee3d2e695b7b0e0b49a5bc839a7` with
  `v2/dec/ops/m6/m6-prep.sh`. Receipts sit beside every output under `/data/dev2/runs/dec/m6/`.
- **Lock check:** `ops/m6/m6-lockcheck.py part1` from the mirror of `8e936c03a` gives **PASS**.
  `lock-part1.json` is `57ad5e2b931a4e65e650c259974a9f9e7a75dc63ea6ac1ea55cdbab96bb60941`.

## Part 1 — mixtures, own-Lux teachers, exposure (locked 2026-09-29 ≈17:25 UTC+8)

| Item | `m6-xl-full-59m` (N6D, S6D) | `m4-xl-full-29m` (N6A, S6X; N4XF's, unchanged) | `m6-e8f-r2clean` (E6K) |
| --- | --- | --- | --- |
| Rows (C / N / S) | 110,728 (39,375 / 43,307 / 28,046) | 58,742 (21,400 / 23,070 / 14,272) | 162,696 (105,634 / 35,342 / 21,720) |
| Native tokens | 54,741,176 | 29,407,326 | 138,244,874 |
| TRAIN sha256 | `160812e2c3d39bd09fcbe6c98e2b4b1633f6680bd435241d3a83e791d5dfe33b` | `c7d51219e9f0fdd107b914df5c7e9f81f2b53496bb43a94796857b8ea3fcfa60` | `f9f3c0229551357ad1cb8e6923c64c636ee697e71d162f059548ae223e78d8ae` |
| Spec / source | `ops/m6/specs/m6-xl-full-59m.json` `1d06d6c572140d65899fe8a0163f145349dc073dc719a94b044a10194aad2f1c` | `specs/m4-xl-full-29m.json` `cf311178…` | E8F `d1dc33fc…` filtered by `ops/m6/m6-e8f-clean.py` |
| Own-1.0 teacher (part 1) | `lux-all-59m` `a1bafad59411901cd8a13451b741a186fece445ab78ce9b0466fc5adc7f1f793`, 110,728 / 110,728 | `lux-all-29m` `dc937c42df0950c853bb725eeacf347fde83e938802a2d6c39d2c7d0bf089549`, 58,742 / 58,742 | own Eos: part 2 |
| Exposure receipt (305 r2 groups, payload `2194716a…`) | `8453af8a8a617948f24fbc9452e993927f2cd1ac20cf2929c74eadebe3630704`: 0 groups, methods agree | `52c7ccad…` (M4): 0 groups | `5d6df944fce47c2f436a82ae665307b15c0ac9cf69bf551b006dd0c4e35ec135`: 0 groups, methods agree |
| Top languages (rows) | en 62,347, zh 17,243, ja 5,224, es 3,556, hi-en 3,537, ko 2,864 | en, zh, ja (M4 lock) | en 101,925, zh 54,735, ja 2,910, ko 2,614, de 512 |

No arm needs `--teacher-partial`: every teacher covers every row.

### `m6-xl-full-59m`

- **Build:** the M4 builder and spec with every pool's `budget_fraction` doubled (0.1442258 → 0.2884516). Everything
  else is unchanged: the sampler seed `dec-m4-xl-full-29m-v1`, the input files (all input hashes are identical to the
  29m manifest's), the tokenizer, the 305-group exclusion (the builder list equals the payload's 305 groups), the
  excluded families, and A0s. There are 0 exposed rows (group id, row id or input hash), no A7x component, and no
  source outside the 29m mixture's.
- **Nested: yes.** The sampler takes whole groups per source × task × language stratum, in a seed-keyed hash order,
  up to the stratum's share. So the same seed at a larger share takes a longer prefix of the same order. All
  58,742 rows of `c7d51219` are in the 59m file, byte-identical, and A0s is row-for-row identical.
- **Tokens by block:**
  - A0s 3.98M, taken whole as in the recipe;
  - A7 19.34M: A7g 8.66M, A7q 2.90M, A7o 2.89M, A7i 2.02M, A7p 1.05M, A7m 0.63M, A7s 0.62M, A7r 0.32M,
    A7k 0.17M, A7h 0.07M;
  - v2 pools 19.96M;
  - H7 4.72M and H8 3.63M;
  - v1 arms 3.11M.
- **Deviation from the prereg's "about 58.8M native tokens" (−6.9%).** That figure is 2 × N4XF's total, but the
  recipe takes the A0s rows whole and unchanged, and only the budgeted pools double. The recipe as specified gives
  2 × (29.41M − 3.98M) + 3.98M = 54.84M. The build is at −0.18% of that, and its non-A0s tokens are 1.996 × N4XF's.
  The A7 dose (19.34M, prereg "about 19.4M") is as preregistered.
- **C1 registry:** 0 source hits (the 9B guard's rule: the 38 C1 name keys against every TRAIN `source`). One
  family name matches a key, reported only. `pilot_narrative_reading` matches "narrative" (the C1 candidate
  `CLS-Lab/narrative-gold-annotations`). Its 100 rows are project-generated (`decision2_programmatic_original_v1`,
  generator `textual-oracle-v1`), and the same 100 rows are in N4XF's A0s.

### Own-Lux 1.0 teachers (`bd45a30a`; `v2.dec.compose_teacher`, first source wins)

- **`lux-all-29m`:** N4XF's composed file `e2ff27ce…` for its 54,983 rows (0 targets changed), then `h-w1`
  (`75e557f1`, `679ef009…`) for H7 607 + H8 3,152 rows. Gold agreement of the h-w1 targets: Choice .939 /
  Noul .695 / Score .523.
- **`lux-all-59m`** (the M5 amendment 2 source list): N4XF's file 54,983; A0 canonical 0 (its rows come from the
  first source); RP-v2 waves 1–4: 4,123 / 7,967 / 5,179 / 72; XL w1–w4: 8,752 / 8,635 / 8,856 / 4,743 (w5 and c-w1
  0); h-w1 7,418 (H7 + H8). On the 58,742 shared rows it equals `lux-all-29m` line for line.

### `m6-e8f-r2clean`

The source is E8F's TRAIN `d1dc33fc…`: 162,777 rows, 138,393,350 tokens. Kept rows are byte-identical and in order.
Removals are counted under the first reason that matches:

| Reason | Rows | Groups | Where |
| --- | ---: | ---: | --- |
| (a) the r2 rescreen's 305 excluded groups (group id, row id or input hash) | 62 | 33 | V1-A3 `musique_full_v1.0_train` 58 (29 groups); A7m `multinli_nonfiction` 4 |
| (b) rows A7 v3 (`3a4efc77`) removed relative to A7 v2 | 19 | 19 | A7i `clinc150_train` 10; A7m `multinli_nonfiction` 9 |
| (c) A7-quarantined rows per the DEV2.0-0.8B release | 0 more | — | all 19 listed ids (`e8f-a7v3-removed-in-mixture.ids.json` `5eaa9e71…`) are the (b) rows |
| Total | 81 | 52 | 148,476 native tokens |

- **Matching:** (a) matched by group id alone gives the same 62 rows. The (b) comparison is by id against the six A7
  v3 arm files; their hashes match the `3a4efc77` A7 registry (`14587510…`).
- **After removal:** 0 rows of excluded groups and 0 quarantined ids remain.
- **Components after removal:** A3 1,528 rows, A7 130,134 rows / 121.45M tokens; the other components are unchanged.

### E6K's SELECT / CAL

E8F's seeds ran on node B with `/data/dev2/runs/dec/data-cleanv2` mounted: SELECT `32a4352d…`, CAL `3e34f6cb…`
(receipts `m2/arms/full/m2-E8F-s*.launch.json`). That CAL is not CAL698 (`19cc1a8c…`). E6K runs on node A per the
prereg, with `drive_arm.sh`'s node-A default directory, whose `select.jsonl` / `cal.jsonl` hashes are identical
to those.

`m6-e8f-r2clean` is streamed to node A through the workstation (zstd). Chain a5 starts only after
`m6-lockcheck.py nodea` on node A passes: the copy's SHA-256 equals `f9f3c022…`, and SELECT / CAL equal E8F's.

N6D / N6A / S6X / S6D use `/data/dev2/runs/dec/m3/data-sel700-cal698` (SELECT `32a4352d…`, CAL698 `19cc1a8c…`),
the directory N4XF and S2T used.

## Part 2a — own-Sol labels (locked 2026-09-29 ≈17:35 UTC+8)

Labeled by chain b4 on node B GPU4 (`ops/m6/m6-label.sh sol`, mirror of `8f5699bdf`), following M3's procedure:
`v2.dec.teacher_label` over every row of `m6-xl-full-59m`, then `merge_labels`, then `compose_teacher` for the
29m subset (the mixtures are nested). `m6-lockcheck.py sol` gives **PASS**; `lock-sol.json` is
`92fc66228ef0381a29d137ab0cf7fa79ade401e525e1b37f278f78418b62de1c`.

| Item | `sol-59m` (S6D) | `sol-29m` (S6X) |
| --- | --- | --- |
| Teacher file sha256 | `53e4adc80d3833fb7a690259660cb5d598b4f4d4640bb9a2e5fda18a8a8e0115` | `b8dec62776ea4f80a63b4ba863a381aa8fd5c85f63c65e70d1715d7b8a9c6222` |
| Coverage | 110,728 / 110,728 (0 bad, 0 extra) | 58,742 / 58,742 (0 bad, 0 extra) |
| Gold agreement (Choice / Noul / Score) | .770 / .719 / .430 | .772 / .723 / .439 |

- **Teacher:** `llm-semantic-router/Decision-1.0-Sol-2B` at `ce0c018a`, package temperature 1.3003552029656025 for
  every type (`temperature.json` `f0cbe732…`), max length 8192.
- **Labels:** `sol-labels/labels.jsonl` is one shard and is byte-identical to `sol-59m`. The identity and TRAIN
  hash checks pass, and 529 s of labeling used 0.168 GPU-h.
- **Subset check:** `sol-29m` differs from `sol-59m` on 0 of the 58,742 shared rows.
