# Decoder M16 — state

Branch `xunzhuo/decision-2-training-dec-m16`, worktree `vllm-sr-dev2-dec-m16`. Prereg `9923e8b51`, ops `dfb86c08d`
(mirror on nodes A / B), data lock `5eb61705f`. M15 (nodes E / F) untouched.

## 2026-10-01 17:47Z

- Prep done on both nodes (17:40–17:44Z): all 8 release / arm pairs pass lineage; MLX-DEV panels equal M15's; M14's
  reference readouts staged; leases A GPU3–5 / B GPU2–4 `track=dec-m16`.
- Chains launched 17:47Z from `dfb86c08d`:
  - A GPU3 `08b-C0-a` MLX-DEV, then `08b-RAUP` a25 / a50 / a75; GPU4 `08b-RASD` ×3; GPU5 `08b-RA` ×3.
  - B GPU2 `2b-C0-b` MLX-DEV, `2b-RAUP` ×3, `2b-RASD` ×3; GPU3 `4b-LH-b` MLX-DEV, `4b-LHA10UP` ×3; GPU4 `2b-RA` ×3,
    `4b-LHA10SD` ×3.
- GPU-h so far: 0 (prep was CPU only).

## 2026-10-01 18:12Z

- Builds and readouts running cleanly (≈ 10 min per 0.8B / 2B point). Read: 0.8B a25 and a50 of all three lines;
  2B `2b-RA` / `2b-RAUP` a25, a50; 4B `4b-LHA10UP-a25`. Reference MLX-DEV reads done on both nodes (17:49–17:50Z).
  No failure.
- Node B formal masters copied into `formal/m16/masters` (manifests equal). `m16-fscore.sh` 2B bar fix (bar-t1's
  mlx-diag run is node A's `formal/m3/m3-S2T-soup-mlx`, the DEV2.0-2B weights; the release dir holds only its score)
  committed here before any formal data exists.

## 2026-10-01 18:36Z

- Node A finished all nine 0.8B points (18:20Z); node A GPU3–5 leases released. 0.8B scored on node A and **rules run
  once** (`m16/select/08b-finalists.json`, 18:31Z): 8 of 9 points pass every gate including the MLX-DEV guard
  (`08b-RAUP-a75` fails the Choice type floor). **Finalists `08b-RASD-a75`, `08b-RA-a75`** (largest transfer deltas;
  the second from another line by rule).
- Formal (node B, mirror `5b246b110`): finalists and their readouts relayed from node A through node E (tree hashes
  checked); select file `m16/select/formal/08b-finalists.json` (slot 0 `08b-C0` = the C0 package, manifest
  `26baab01…`); launched 18:36Z: GPU3 parity `m16-08b-C0` then `m16-08b-RASD-a75`, GPU4 `m16-08b-RA-a75`
  (co-tenants of the node-B readout chains).
- Node B readouts: 2B `2b-RA` / `2b-RAUP` ×3 and `4b-LHA10UP` a25 / a50 done; the rest running.

## 2026-10-01 19:40Z

- 0.8B formal (node B, scored on node A): parity `m16-08b-C0` is exact vs node B's stored M8s reference of the same
  weights (0 / 0 / 0 differing answers) and not exact vs bar-t1 (9 / 20 / 1), so both bars bind. `m16-08b-RASD-a75`
  and `m16-08b-RA-a75` both pass items 2, 3, 4 (mlx-diag card-eligible upper bounds ≥ 0 vs both bars), 5, 6(a) and 7,
  and **fail items 1 and 6(b)**. No 0.8B successor.
- 2B: all nine points read on node B and scored on node A; **rules run once** (19:11Z): `2b-RAUP` a50 / a75 and
  `2b-RASD` a50 / a75 fail the Score floor; the other five pass. **Finalists `2b-RA-a75`, `2b-RASD-a25`.** Formal
  collected on node B (GPU3 parity `m16-2b-C0` then `m16-2b-RA-a75`, GPU4 `m16-2b-RASD-a25`; done 19:24Z), relayed
  to node A (19:33Z); scoring running.
- 4B: all six points read on node B (chains finished 19:08Z), relayed to node A (filtered tree content checked equal
  to node B's), scoring running. A duplicate relay invocation from the workstation stopped at the relay's
  "exists on the target" guard; it changed nothing.
- Node B GPU2–4 now idle until the 2B mlx-diag and 4B formal collections.

## 2026-10-01 20:09Z

- 2B formal: parity `m16-2b-C0` is exact vs bar-t1 (0 / 0 / 0). `m16-2b-RA-a75` −2.05 [−3.31, +3.14] vs bar-t1 and
  `m16-2b-RASD-a25` +.05 [−.60, +1.68]; both **fail items 1 and 6(b)**; `2b-RA-a75` also fails item 4 (mlx-diag
  card-eligible −.0184 [−.0295, −.0075]). No 2B successor.
- Scoring fix (`ab7be2f15`, committed before any 2B mlx-diag pair existed): m6-score's report-only mlx-diag pair for 2B
  looked for an `m6-ref-S2T-soup-mlx` under M16's root, which does not exist; it now uses node A's `m3-S2T-soup-mlx`
  (bar-t1's run). The first attempt stopped at that check before writing any pair file; it was rerun on the new mirror.
- 4B: six points scored on node A; **rules run once** (19:40Z): the three `4b-LHA10UP` points and `4b-LHA10SD-a25`
  fail retention (CI upper < 0). **Finalists `4b-LHA10SD-a75`, `4b-LHA10SD-a50`** (no other line eligible). Formal on
  node B GPU3: parity `m16-4b-LH` exact vs the stored LH run (0 / 0 / 0); `m16-4b-LHA10SD-a75` −.07 [−2.53, +2.38] vs
  bar-lh (item 1 fails). `-a50` collecting; 4B mlx-diag collecting on GPU4.
