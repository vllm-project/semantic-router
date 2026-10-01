# Decoder M13 — state

## 2026-10-01 13:30Z — prereg, prep, self-distillation labeling

- Prereg [`dec-m13-prereg-2026-10-01.md`](dec-m13-prereg-2026-10-01.md) + ops `c2610143b` (before any M13 data or GPU
  job). `teacher_label` gains `--uncalibrated` / `--teacher-source-path`; its tests pass in the decoder image on node E.
- Prep (CPU) on nodes E and F, 13:21Z: every TRAIN byte-identical across the nodes — `4b-LHA5` `4e5316aa…` (65,789
  rows, IB 7,050 rows / 1,469,208 tokens = 5.0% of T₄), `08b-RAAG` `69b8d00d…` (359,919 rows = `08b-RA` + 2 × 25,347
  proxy-family rows), and the M12 hard links `4b-LHA10SD` (`d41cdd1a…`), `2b-RASD` (`08140409…`), `08b-RASD`
  (`12bd63d8…`). References: `08b-C0-e`, `4b-LH-f` from M12 (8 panels); `2b-C0-f` from M11 (7 panels).
- SD labeling started 13:23Z: node E 0.8B (DEV2.0-0.8B) on GPU0–3; node F 4B (LH soup) then 2B (DEV2.0-2B) on GPU2, 3,
  6, 7. Pre-warm shards passed (agreement with TRAIN gold — 0.8B choice / Noul / Score .845 / .821 / .895; 4B .950 /
  .923 / .846).
