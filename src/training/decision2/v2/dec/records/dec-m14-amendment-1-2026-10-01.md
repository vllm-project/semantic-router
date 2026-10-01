# Decoder Milestone 14 — amendment 1 (formal scoring: two bars per tier, mlx-diag step, node-B relay; 2026-10-01)

Committed ≈16:15Z, before any 0.8B or 4B rules output exists (the 2B rules ran at 15:55Z: no finalist) and before any
M14 formal job. Prereg [`dec-m14-prereg-2026-10-01.md`](dec-m14-prereg-2026-10-01.md), "Formal, successor items,
hand-offs". It changes no development gate, threshold or result.

## Why

The prereg names the bars (4B the released LH, `m10-4b-LH`'s formal run; 0.8B DEV2.0-0.8B) and says each tier's
reference is collected first on the node-B formal path as its parity run, but the inherited M6 scorer
(`ops/m6/m6-score.sh`) hard-codes `bar-t1` as the tier's **previous** release (for 4B DEV2.0-4B, with N4XF's mlx-diag
reference), and node A cannot reach node B to pull runs or return the v3 seal that gates mlx-diag. M13 met the same
gap with its amendment 2 (two bars, `m13-fscore.sh`, `m13_successor.py`); M14 adopts that rule unchanged.

## Scoring path (`ops/m14/m14-fscore.sh`, `ops/m14/m14_successor.py`, `ops/m14/m14-relay.sh`, `ops/m14/m14-formal.sh`)

| Tier | First bar (stored) | Second bar (M14 parity run, node B) | Item-4 mlx-diag references |
| --- | --- | --- | --- |
| 4B | `bar-lh` = `formal/m10/m10-4b-LH-t1-derived` (the released LH, T = 1, v3 67.345) | `bar-b` = `m14-4b-LH` (the LH soup) | `m10-4b-LH-mlx`; `m14-4b-LH-mlx` |
| 0.8B | `bar-t1` = `runs/release/dev2-0p8b-t1-derived` (v3 50.236) | `bar-b` = `m14-08b-C0` (DEV2.0-0.8B's weights) | node A's `m2-E8F-soup-nodeA-mlx`; `m14-08b-C0-mlx` |

- Items 1 (v3 lower bound > 0), 2 (H upper bound ≥ 0), 4 (mlx-diag card-eligible upper bound ≥ 0), 6(b) (rules 1 and
  5 on the reduced panels) and 7 (public 231 not REGRESSION) must pass **against both bars**; items 3, 5 and 6(a) are
  read once. 2B has no finalist, so no 2B path is added.
- Parity: the parity run is scored first and repeated against the stored first bar (`PARITY-EXACT.json`; 0 differing
  answers on typed FINAL, CSS15 and public 231 is "exact"). At 0.8B it is also repeated, report only, against node B's
  stored M8s reference of the same weights (`m8s-ref-08b-I`; M8s found the node-A and node-B paths differ). A non-exact
  parity run is reported; it does not replace either bar.
- Item 6(a): the finalist's TRAIN (node A's staged copy of M12's locked file, hash-checked) through
  `overlap_effects exposure` against the r2 payload (`2194716a…`). M14 finalists contain no incumbent weight, so they
  inherit no exposure.
- **Then-current release.** If an M13 or fast-track candidate of the same tier is released before M14's verdict for
  that tier, the first bar becomes that release's stored formal run (and its mlx-diag run), the second bar its weights
  collected on the node-B path, and the results record says so. (At 16:10Z: the fast track's `08b-RA` failed formal,
  so DEV2.0-0.8B `4afea305` stands; M13's `4b-LHA10SD` is in formal, not released.)
- Relay: node A does not reach node B, and the workstation link is too slow for model or run directories, so runs move
  through an M14-only transit directory on node E (`m14-relay.sh formal-pull`, gold- and weight-free, manifest
  checked); the v3 seal returns to node B as `V3-SEALED.json` (`m14-relay.sh formal-mark`, seals compared), which gates
  `m14-formal.sh mlx-launch`. A 0.8B finalist's soup reaches node B the same way (`m14-relay.sh dir a`).
- Order per finalist: node-B collection of the reference and the finalist → `formal-pull` → `parity` → `run` →
  `formal-mark` → `mlx-launch` (finalist and parity run) → `formal-pull` of the `-mlx` runs → `mlx` → `exposure` →
  `overlap` → `successor`. A failed step stops that point (no rerun). Item 8 stays the custodian's (C1 content recheck
  for IB1-r3 + IB2, worker 3c7679b0, then the item-8 spec).
