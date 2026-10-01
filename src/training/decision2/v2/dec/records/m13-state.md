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

## 2026-10-01 13:50Z — data lock, launch

- SD targets done 13:27–13:29Z (full coverage). Data lock [`dec-m13-datalock-2026-10-01.md`](dec-m13-datalock-2026-10-01.md)
  `debd36e1a` pushed; `READY-m13.json` written on E and F; a dry READY check passed for all five arms.
- Launched 13:46Z from mirror `c2610143b`: node F GPU2 `4b-LHA5-s1` (pre-warm 4B) then `2b-RASD-s1` (pre-warm 2B),
  GPU3 `4b-LHA5-s2` then `2b-RASD-s2`, GPU6 / GPU7 `4b-LHA10SD-s1` / `s2`; node E GPU0 `08b-RAAG-s1` (pre-warm 0.8B),
  GPU1 `08b-RAAG-s2`, GPU2 / GPU3 `08b-RASD-s1` / `s2`. Post chains: F GPU6 `4b-LHA5`, GPU7 `4b-LHA10SD`, GPU3
  `2b-RASD`; E GPU1 `08b-RAAG`, GPU3 `08b-RASD`.
- Node A: M13 tier probe golds built (identical hashes to M12's); M12 readouts staged as `08b-RA-m12`, `2b-RA-m12`,
  `4b-LHA10-m12` for the report-only contrasts.

## 2026-10-01 14:50Z — training

- All pre-warm preflights passed (`warm-4b-f` 13:50Z, `warm-08b-e` 13:55Z); every 4B / 0.8B seed passed preflight.
- `4b-LHA5-s1` DONE 14:48Z (886 steps); `2b-RASD-s1` (2B pre-warm on F) started on GPU2. `4b-LHA5-s2` and
  `4b-LHA10SD-s1` / `s2` near their end (~860 / 1,138 steps). Node E's four 0.8B seeds run at ≈ 40 steps / min (four
  0.8B jobs at once; M12 ran two): `08b-RASD` ≈ 2,370 / 4,831 steps, `08b-RAAG` 2,492 / 2,066 of 5,624; expected end
  ≈ 16:05–16:25Z.

## 2026-10-01 14:50Z — training

- All pre-warm preflights passed (`warm-4b-f` 13:50Z, `warm-08b-e` 13:55Z); every 4B / 0.8B seed passed preflight.
- `4b-LHA5-s1` DONE 14:48Z (886 steps); `2b-RASD-s1` (2B pre-warm on F) started on GPU2. `4b-LHA5-s2` and
  `4b-LHA10SD-s1` / `s2` near their end (~860 / 1,138 steps). Node E's four 0.8B seeds run at ≈ 40 steps / min (four
  0.8B jobs at once; M12 ran two): `08b-RASD` ≈ 2,370 / 4,831 steps, `08b-RAAG` 2,492 / 2,066 of 5,624; expected end
  ≈ 16:05–16:25Z.
