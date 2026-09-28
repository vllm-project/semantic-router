# Decoder Milestone 4 — data lock (2026-09-29)

Identities of every M4 training input, recorded and pushed before the arms that use them launch (the chains wait
for a per-mixture READY marker written after this record's commit). Preregistration:
[`dec-m4-prereg-2026-09-29.md`](dec-m4-prereg-2026-09-29.md) (`0b8ebfc54`). Built on node B from the exact
mirror of `0b8ebfc5411d5dfc47f33f36bc94c585a1c926a1` with `v2/dec/ops/m4/m4-prep.sh` (receipts beside every
output under `/data/dev2/runs/dec/m4/`).

## 1. `m4-v2m-ret-r2` (arms N4LR, N4LR2) — locked 2026-09-28 19:45 UTC

| Item | Value |
| --- | --- |
| TRAIN | 56,141 rows, 29,162,107 native tokens (−0.30% vs 29,249,047), Choice / Noul / Score 16,643 / 26,740 / 12,758, `1527b38b1ba888695fe48dd43e92827d1719d57674009cfc29d5ab759b08dd2c` |
| Spec | `specs/m4-v2m-ret-r2.json` `d2ddd348…` |
| vs M3 `m3-v2m-ret` | exactly the 57 rows of r2-excluded groups removed (E11 12, H1 2, H3 30, H6 13); A0s (6,547) and retention (8,276 rows, 9,754,781 tokens) ids identical to M3; 0 rows of the 305 excluded groups left |
| Own-Lux teacher | `0c35263a…` (below); 56,141 of 56,141 rows |

Teacher file `teacher/m4-v2m-ret-r2/lux-teacher.jsonl` `0c35263aa3a2ffde7e466f01cdc9ed2edace50211ec01a26b707209b7d623091`
(`v2.dec.compose_teacher`). Rows by source: A0 canonical 6,547; rp-v2 wave1 12,672, wave2 28,600, wave4 46;
XL `w2` 2,507, `w3` 2,071, `w4` 1,550, `c-w1` 1,037; M4 gap labels 1,111.

**Gap labels** (`teacher/ret-gap-lux/labels.jsonl` `1de1ec4f…`, 1,411 rows = 1,111 uncovered retention rows
(Choice 710 / Noul 349 / Score 52) + 300 cross-check rows; slice `teacher/ret-gap-rows/rows.jsonl` `7f49918c…`):
Lux 1.0 `bd45a30a` via `v2.dec.teacher_label` on node B GPU0 (kernel image `dbe5f32b`), 2.3 min.
**Cross-check vs published own-Lux targets: argmax agreement 298 / 300 = 0.993** (preregistered bar 0.97; both
disagreements on rp-v2 wave2 rows), mean absolute probability difference 0.0003–0.0025 per source. Gold
agreement of the 1,111 gap targets: Choice .727, Noul .946, Score .827.

Identity checks re-run with `ops/m4/m4-lockcheck.py` after a fix (the first run sliced components in the
manifest's alphabetical order; A0s was unaffected): A0s and retention ids are identical to M3's, as stated.

## 2. `m4-v2m-ret-r2-q20`, `m4-xl-a7v1-29m`, `m4-xl-full-29m` (arms N4LRQ, N4XA, N4XF) — locked 2026-09-28 19:50 UTC

| Mixture | Rows (C / N / S) | Native tokens (vs budget) | TRAIN sha256 | Spec | Own-Lux teacher (rows covered) |
| --- | --- | --- | --- | --- | --- |
| `m4-v2m-ret-r2-q20` | 62,595 (13,616 / 18,473 / 30,506) | 29,256,331 (+0.03%) | `0399562a2d9eb2c81bc9df81e19ee21b035125dfd14a9b34ef5b042821a653b4` | `0ad26842…` | `297ef490d6e0e1eba63d410aa19d09eeaa067e595b4ea3ce67b6951e2a7c6615` (62,595 / 62,595) |
| `m4-xl-a7v1-29m` | 47,719 (24,019 / 10,260 / 13,440) | 29,294,910 (+0.16%) | `5d3b60d725f4f1e3bc9004f004dc5d44601c00a9efa2c91a1a77bbf01e3c0166` | `89b03be6…` | `efc8910625dbf4d2beddbed74fbd87747d2feeb1f32f98237e2b73b555adafe8` (47,719 / 47,719) |
| `m4-xl-full-29m` | 58,742 (21,400 / 23,070 / 14,272) | 29,407,326 (+0.54%) | `c7d51219e9f0fdd107b914df5c7e9f81f2b53496bb43a94796857b8ea3fcfa60` | `cf311178…` | `e2ff27ce2fc397c8c203e37133dcfac52ac3d1f172c6f3e9412ec28c2ffe296c` (54,983; H7 607 + H8 3,152 gold-only) |

All three: 0 rows of the 305 excluded groups; A0s ids identical to M3; the q20 retention ids identical to M3.

- **q20** (A7q/k/s swap): v2-M pools 9.66M tokens (62.3%); A7k 2,190 / 0.59M, A7s 12,860 / 2.17M, A7q 6,895 /
  3.11M = 20.05% of tokens. Rows are now 49% Score and 20% Hinglish (A7s SentiMix) — a large shift in row shares
  that the token-matched design implies (the loss is per row); disclosed for interpretation.
- **XL A7-only** (22.12% per pool): A7g 8,577 rows / 12.54M tokens (43%), A7o 3.72M, A7q 2.34M, A7i 1.72M, v1
  arms 2.78M, A7 h/k/m/p/r/s 2.21M; languages en 27,868 / zh 12,449 / hi-en 2,714 rows.
- **XL full** (14.42% per pool): v2 pools 10.0M tokens, H7 / H8 4.19M (gold-only, 6.4% of rows, so the KL term is
  diluted by about that share under `--teacher-partial`), A7 9.68M, v1 1.56M.
