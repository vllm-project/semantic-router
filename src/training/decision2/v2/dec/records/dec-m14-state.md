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
