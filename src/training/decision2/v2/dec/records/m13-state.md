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

## 2026-10-01 15:20Z — 4B rules: finalist `4b-LHA10SD`; 4B formal launched

- 4B soups and panels done 15:11Z; scored on node A 15:15Z; rules run once 15:17Z. **`4b-LHA10SD` passes every gate**
  (typed T .867 vs LH .868; choice / Noul / Score 708 / 307 / 372 vs LH 728 / 290 / 371; HT-DEV v2 −.003 [−.018, +.012]
  TIE; retention −.022 [−.047, +.002]; transfer +.046 [+.034, +.058]; IB DEV +.082; false-yes .202 vs .199).
  `4b-LHA5` fails (Noul type floor 266 < 278; Noul `rule_precedence` floor; retention −.033 [−.057, −.010]) despite
  HT-DEV v2 +.019 [+.004, +.034] and typed T .912.
- Formal (node F, M6 library, `m13-formal.sh`): entries `m13/select/formal/4b-finalists.json` (slot 0 = the LH soup as
  the formal-path parity reference, slot 1 = `4b-LHA10SD`); launched 15:18Z, GPU6 `4b-LH`, GPU7 `4b-LHA10SD`.
- 2B (`2b-RASD`, node F GPU2 / 3) and 0.8B (node E) still training.

## 2026-10-01 15:40Z — 4B formal relaunched under amendment 1; hand-off

- 4B formal: the 15:18Z smokes stopped at the formal library's input check (relative readout paths in the entries; no
  container, no CAL fit). [Amendment 1](dec-m13-amendment-1-2026-10-01.md) `6c1236f18` committed first; entries
  regenerated with absolute paths (same checkpoints, `files_sha256_list_sha256` unchanged), originals kept under
  `formal/m13/logs/amendment-1/`; relaunched 15:31Z. **Both smokes passed**; collection running since 15:34Z (node F
  GPU6 `4b-LH`, GPU7 `4b-LHA10SD`).
- 2B: `2b-RASD-s1` / `s2` DONE 15:19Z / 15:24Z (1,251 / 1,249 steps); soup built; post (node F GPU3) on its last panel.
- 0.8B: `08b-RASD-s1` / `s2` DONE (4,115 / 4,109 steps); soup built; post finished 15:29Z. `08b-RAAG-s1` / `s2` at
  4,561 / 4,074 steps (≈ 4,790 total), then soup and post on node E GPU1 (≈ 16:30Z).

### Hand-off: remaining steps (in order, from mirror `c2610143b`; `<E>` / `<F>` from `nodes.env`)

1. Node A, 2B, once `post-2b-RASD post 2b-RASD finished`: `m13-score.sh pull <F> 2b-C0-f 2b-RASD`;
   `points 2b-C0-f 2b-RASD 2b-RA-m12`; `ibdev 2b-C0-f 2b-C0-f 2b-RASD 2b-RA-m12`; `contrast 2b-RASD 2b-RA-m12`;
   `readout 2b 2b-C0-f 2b-RASD`; `rules 2b 2b-RASD=2b-C0-f` (once).
2. Node A, 0.8B, once `post-08b-RAAG post 08b-RAAG finished`: `pull <E> 08b-C0-e 08b-RASD 08b-RAAG`;
   `points 08b-C0-e 08b-RASD 08b-RAAG 08b-RA-m12`; `ibdev 08b-C0-e 08b-C0-e 08b-RASD 08b-RAAG 08b-RA-m12`;
   `contrast 08b-RASD 08b-RA-m12`; `contrast 08b-RAAG 08b-RA-m12`; `readout 08b 08b-C0-e 08b-RASD 08b-RAAG`;
   `rules 08b 08b-RASD=08b-C0-e 08b-RAAG=08b-C0-e -- 08b-RASD:08b-RAAG` (once).
3. Passers (≤ 2 per tier): `m10_formal_select.py --tier <tier>` with **absolute** readout paths (amendment 1), the
   tier reference in slot 0; `M13_NODE=e|f m13-formal.sh launch <mirror> <gpu> <tier> <point>` (2B / 0.8B via
   `M6_SMALL_NODE`).
4. 4B formal: when both `formal/m13/status/m13-4b-*.COLLECTED` exist, score on node A (M6 library, as M12), then
   mlx-diag; successor items 1–7 vs LH formal v3 67.34.
5. Close-out: `python3 $O/m13_gpuh.py table` on E and F (cap 80); results record `dec-m13-results-2026-10-01.md`,
   gist 04 entry, merge into integration (`check_no_private.sh --strict --log`). Hand-offs per prereg (package on the
   repo's current `main`; custodian C1 IB content recheck before the item-8 spec; private Index request).

## 2026-10-01 15:45Z — continuation; 2B rules: no finalist

- Continuation resumed from `f15cb7139`. 4B formal collection running (node F GPU6 `4b-LH`, GPU7 `4b-LHA10SD`).
- 2B post finished 15:36Z; scored on node A 15:37Z; rules run once 15:38Z. **`2b-RASD` fails** the Score type floor
  (237 < 257 − 12; choice / Noul / Score 479 / 233 / 237; typed T .593 vs .610). Everything else held: HT-DEV v2
  +.017 [−.001, +.034] TIE, retention +.008 [−.018, +.035], IB DEV +.179, transfer +.155 [+.126, +.186], hs1 false-yes
  .494 vs .625. Contrast vs M12 `2b-RA`: HT-DEV v2 +.020 [+.006, +.034] GAIN, retention +.018 [−.007, +.044]; the
  Score count moved 227 → 237, short of the floor. No 2B finalist; no exception (COORDINATION 23:15).
- 0.8B: `08b-RAAG-s2` at step 4,571 of ≈ 4,790; then soup and post on node E GPU1.

## 2026-10-01 16:05Z — 4B formal item 1 fails; 0.8B rules: no finalist

- [Amendment 2](dec-m13-amendment-2-2026-10-01.md) `0ea6b3e62` (before any formal scoring): two-bar formal scoring
  (`m13-fscore.sh`, `m13_successor.py`; 4B bars = the released LH's stored formal run and node F's `m13-4b-LH`), an
  mlx-diag step in `m13-formal.sh`. Tests 20 / 20.
- 4B formal collected 15:39Z (both points); scored on node A 15:53Z. Parity: `m13-4b-LH` reproduces the stored LH run
  exactly (0 answer changes on typed FINAL, CSS15, public 231; v3 67.345). **`m13-4b-LHA10SD` v3 67.187: vs LH −0.158
  [−2.836, +3.636] (both bars) — item 1 FAILS** (lower bound ≤ 0). Report-only: vs DEV2.0-4B +4.035 [+1.571, +9.636],
  vs adopted Nox 1.0 +10.72 [+6.53, +13.66]; types OK; public 231 174 vs 172. Item 6(a) exposure: 0 groups.
  Items 2, 4–7 are being completed for the record (no change to the verdict).
- mlx-diag: the 15:53Z launch used the amendment-2 mirror; the containers could not read the receipts' adapter spec
  (collecting mirror `c2610143b` not mounted) and stopped before inference. [Amendment 3](dec-m13-amendment-3-2026-10-01.md)
  `4f8bd3c34`; relaunched 15:55Z from the collecting mirror (node F GPU6 `4b-LH`, GPU7 `4b-LHA10SD`).
- 0.8B post finished 15:49Z; scored 15:57Z; rules run once 15:59Z. **No 0.8B finalist.**
  - `08b-RASD`: choice floor 574 < 586 (C / N / S 574 / 208 / 205 vs C0 610 / 212 / 159; `attribute_gate` 293 ≥ 280
    holds); HT-DEV v2 +.040 [+.019, +.060] GAIN; retention +.026 [+.001, +.054]; transfer +.201 [+.165, +.237];
    false-yes .783 vs .720.
  - `08b-RAAG`: choice floor 498 < 586 and `attribute_gate` 228 < 280; HT-DEV v2 +.038 GAIN; retention +.009
    [−.018, +.037]; transfer +.194.
  - vs M12 `08b-RA` (C / N / S 600 / 216 / 202, `attribute_gate` 277): RASD HT −.009 tie, `attribute_gate` +16, choice
    −26; RAAG retention −.035 [−.062, −.007]. Upweighting the proxy families lowered `attribute_gate` itself.

## 2026-10-01 16:20Z — M13 done: no model passes items 1–7

- 4B mlx-diag collected 15:57Z (both points); scored 16:01Z. **`m13-4b-LHA10SD` successor: FAIL** — item 1 (v3 tie
  vs LH) and item 4 (mlx-diag card-eligible −.0287 [−.0394, −.0184]); items 2, 3, 5, 6(a), 7 pass; 6(b) not
  evaluated (overlap spec error in `m13-fscore.sh`, fixed afterwards; not rerun).
- No hand-offs. Results [`dec-m13-results-2026-10-01.md`](dec-m13-results-2026-10-01.md); GPU-h 14.21 of 80.
