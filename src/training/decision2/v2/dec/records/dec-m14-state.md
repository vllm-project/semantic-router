# Decoder M14 — state (polled ≤ 30 min)

Prereg [`dec-m14-prereg-2026-10-01.md`](dec-m14-prereg-2026-10-01.md). Branch `xunzhuo/decision-2-training-dec-m14`
(from integration `4cd24bcd9`). GPUs: node A GPU3–5, node B GPU2–4 only. Cap 80 GPU-h.

## 2026-10-01 15:25Z — prereg

- Trainer `--example-weights` (`11ec9c5e4`, tests in `test_dec.py`); prereg and `ops/m14/` committed before any M14
  data build or GPU job. Arms `08b-RAUP` (released rows ×1.5), `2b-RAUP` (Choice / Noul ×1.5, Score ×2), `4b-LHA10UP`
  (×1.5), IB rows ×1, window-normalized loss weights on M12's locked TRAIN files.
- Lease check: node A GPU3–5 have no live owner (MoE entries released 11:23–11:40Z); node B GPU2 (27B M4b, idle since
  09-29), GPU3 / GPU4 (decoder M8, idle since 09-30) idle, 0% use, no VRAM in use.
- Next: mirror, prep on A and B (pulls from E / F), data lock, READY, launch.

## 2026-10-01 15:12Z — data lock, launch

- Mirror `e48a66f46` on A and B. Prep 15:04–15:05Z (CPU): M12 TRAIN + ids, teachers, C0 packages, LH soup and the
  `ib-dev` / `m10-probes` prompts pulled from E / F and hash- or manifest-checked; weights byte-identical on both
  nodes (`99b77df5…`, `e1ae6144…`, `3f741da8…`). Data lock [`dec-m14-datalock-2026-10-01.md`](dec-m14-datalock-2026-10-01.md)
  `940fb56a2` pushed; `READY-m14.json` written and checked on A and B. Trainer tests (39) pass in the image.
- Leases `track=dec-m14` on node A GPU3–5 and node B GPU2–4 (previous idle owners kept as `owner.prev-*`).
- Launched 15:08Z from mirror `e48a66f46`: node A GPU3 `08b-RAUP-s1` (pre-warm), GPU4 `08b-RAUP-s2` (waits),
  GPU5 refs `08b-C0-a` then post `08b-RAUP`; node B GPU2 `2b-RAUP-s1` (pre-warm) then `4b-LHA10UP-s1` (pre-warm),
  GPU3 `2b-RAUP-s2` then `4b-LHA10UP-s2`, then post `4b-LHA10UP`; GPU4 refs `2b-C0-b`, `4b-LH-b`, then post `2b-RAUP`.
- 15:11Z: both pre-warm preflights running (2B zero-step done); reference reads at `ht-dev2` on both nodes.

## 2026-10-01 15:50Z — poll

- Every preflight PASSED: `2b-RAUP-s1` 15:11:48Z (pre-warm), `-s2` 15:16Z, `08b-RAUP-s1` 15:16:34Z (pre-warm), `-s2`
  15:24:58Z. 2B seeds 1,251 / 1,249 updates (≈ 37 / min), 0.8B 4,115 each (≈ 37 / min on node A; ETA ≈ 17:10–17:30Z).
- References read (eight panels each): `08b-C0-a` 15:17Z, `2b-C0-b` / `4b-LH-b` 15:29Z. **Path check: each equals
  M12's node E / F read of the same weights exactly** (`08b-C0-e`, `2b-C0-e`, `4b-LH-f`; 0 decisions differ, drift 0.0
  on all eight panels), so the M12 contrasts are exact comparisons.
- Formal prep: `M6_SMALL_NODE=B` library option (`915fcb87a`, test), node-B wrapper `m14-formal.sh` (`a09bd732f`);
  masters copied to `formal/m14/masters` (manifests equal to node B's frozen ones). Node-B formal collections take
  minutes (M6 / M8 receipts: finalist 0.07–0.10 GPU-h), so reference parity runs follow each tier's rules directly.
- The workstation relay was too slow (≈ 11 MB per point in > 10 min); `m14-relay.sh` now hops through an M14-only
  transit directory on node E (two points in 72 s). The interrupted first copy on node A was removed before the hop.

## 2026-10-01 16:00Z — 2B closed (no finalist)

- `2b-RAUP` seeds DONE 15:39Z / 15:43Z; soup 15:44Z; eight panels 15:53Z (node B GPU4); relayed through node E.
  Rules (run once, 15:55Z, readout `4193a337…`): **not eligible — Score type floor 236 < 257 − 0.03·400.** Otherwise:
  T .629 vs C0 .610 (choice / Noul / Score 538 / 232 / 236; C0 – / 230 / 257); HT-DEV v2 +.010 [−.006, +.026] TIE;
  retention .564, −.004 [−.034, +.025]; Score5t no flags; `hs1-dev` false-yes .557 vs .625; IB DEV +.180 [+.156,
  +.206], transfer +.156 [+.127, +.188].
- Contrast vs M12 `2b-RA` (exact path): typed T −.013 (choice −20, Noul −9, **Score +9**); HT-DEV v2 +.014 [+.002,
  +.025] (TIE band); retention +.006 [−.021, +.032]. Score ×2 moved Score up by 9 items and Choice / Noul down: the loss
  moves between heads again, and the Score floor is still 9 short. No 2B formal; DEV2.0-2B stays.
- 4B seeds training on node B GPU2 / GPU3 (s1 15:39Z pre-warm, s2 15:44Z); 0.8B seeds on node A GPU3 / GPU4.

## 2026-10-01 16:15Z — poll; amendment 1; bars unchanged

- [Amendment 1](dec-m14-amendment-1-2026-10-01.md) `efb303494` (before any 0.8B / 4B rules output): M13's two-bar
  formal scoring adopted for node B (`m14-fscore.sh`, `m14_successor.py`, tests 7 / 7), the mlx-diag step, and
  `formal-pull` / `formal-mark` through node E; `4f853c0fe` applies M13's overlap-spec fix before first use.
- **Bars stay the current releases.** The 0.8B fast track's `08b-RA` failed formal (items 1, 4, 6(b); COORDINATION
  23:15), so DEV2.0-0.8B `4afea305` stands; M13 closed with no model passing items 1–7 (its `4b-LHA10SD` tied LH on v3
  and failed mlx-diag), so LH (`13d42143`) and DEV2.0-2B stand.
- Pre-staged on node B for a possible 0.8B formal: the DEV2.0-0.8B package from node E (manifest `26baab01…`, equal to
  node A's) and the `08b-C0-a` readouts (relayed).
- Training at 16:10Z: `08b-RAUP` 2,699 / 4,115 and 2,283 / 4,109 (≈ 54 / min; ETA ≈ 16:36 / 16:45Z);
  `4b-LHA10UP` 416 / 980 and 335 / 991 (≈ 17 / min; ETA ≈ 16:42 / 16:51Z).
- GPU-h of finished jobs: node A 0.41, node B 1.68. `m14_gpuh.py` now counts only receipts of jobs that ran on the
  node (`--node` / `$M14_NODE`): the staged `<point>-m12` readouts and relayed node-B readouts carry their original
  receipts and were being double-counted (the running chains' 30 GPU-h gate only became more conservative).

## 2026-10-01 16:58Z — 0.8B closed (no finalist)

- `08b-RAUP` seeds DONE 16:36Z / 16:43Z (4,115 / 4,109 updates, = M12's windows; BEST 3,086 / 4,109); soup 16:44Z;
  eight panels 16:52Z (node A GPU5). Provenance confirms the weights (`99b77df5…`, distinct 1.0 / 1.5).
- Rules (run once, 16:54Z, readout `ce79fdb6…`): **not eligible — Choice type floor 524 < 610 − 24; `attribute_gate`
  family floor 268 / 400 < 320 / 400 − .10; yes-bias guard `hs1-dev` false-yes .827 > .720 + .10.** Otherwise: T .592
  vs C0 .613 (C / N / S 524 / 219 / 204; C0 610 / 212 / 159); HT-DEV v2 **+.028 [+.007, +.049] GAIN**; retention .445,
  +.011 [−.016, +.036]; Score5t COLLAPSE / NO-GAIN as C0 (M8s amendment 3 floor holds); IB DEV +.211, transfer +.197.
- Contrast vs M12 `08b-RA` (exact path): typed T −.044 (**Choice −76**, Noul +3, Score +2); HT-DEV v2 −.021 [−.039,
  −.002] FLAG; retention −.033 [−.060, −.005] (GSM8K −.096). Giving the released rows 1.5× the IB rows' weight did not
  protect the Choice head at 0.8B; it lost more Choice items and raised the unmet-condition yes-bias.

## 2026-10-01 17:15Z — 4B closed; M14 complete (no finalist)

- `4b-LHA10UP` seeds DONE 16:45Z / 16:50Z (980 / 991 updates; BEST 858 / 867); merges 128 / 128 and 127 / 128 SELECT
  argmax; soup 16:53Z; eight panels 17:05Z (node B GPU3); relayed through node E.
- Rules (run once, 17:07Z, readout `3b11fb2e…`): **not eligible — Score type floor 345 < 371 − 12; retention −.042
  [−.071, −.014].** Otherwise: T .908 vs LH .868 (C / N / S 799 / 309 / 345 vs 728 / 290 / 371); **HT-DEV v2 +.028
  [+.013, +.043] GAIN**; false-yes .196 vs .199; IB DEV +.078; transfer +.045 [+.034, +.058]. vs M12 `4b-LHA10`: typed T
  +.035 (C +56, N −16, S +16), HT-DEV v2 +.024 [+.011, +.038] GAIN, retention −.019 [−.046, +.006].
- **M14 has no finalist at any tier**: no formal, no successor items, no hand-offs, no Index requests. Results
  [`dec-m14-results-2026-10-01.md`](dec-m14-results-2026-10-01.md). GPU-h 7.12 of 80 (A 3.18, B 3.94); no M14 job
  running; all M14 leases released; node E transit directory removed.
