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
