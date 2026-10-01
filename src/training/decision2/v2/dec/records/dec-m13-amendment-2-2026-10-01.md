# Decoder Milestone 13 — amendment 2 (formal scoring: two bars per tier; mlx-diag step; 2026-10-01)

Committed before any M13 formal run is scored on node A (the 4B collections are still running on node F) and before
any 0.8B rules output exists. Prereg [`dec-m13-prereg-2026-10-01.md`](dec-m13-prereg-2026-10-01.md), "Formal,
successor items, hand-offs"; amendment 1 [`dec-m13-amendment-1-2026-10-01.md`](dec-m13-amendment-1-2026-10-01.md).

## Why

The prereg names the bars (4B: the released LH, `m10-4b-LH`'s formal run, v3 67.34; 0.8B: DEV2.0-0.8B; 2B:
DEV2.0-2B) and says each tier's reference is collected first as the formal-path parity run. The M6 scorer it inherits
(`ops/m6/m6-score.sh`) hard-codes `bar-t1` as the stored formal run of the tier's **previous** release; for 4B that is
DEV2.0-4B (`runs/release/dev2-4b-t1-derived`), not LH, and its item-4 mlx-diag reference is N4XF's. M13 also had no
node-A pull / seal-mark step for its node-E / F formal runs and no mlx-diag launcher. This amendment fixes the scoring
path; it changes no gate, threshold or development result.

## Scoring path (`ops/m13/m13-fscore.sh`, `ops/m13/m13_successor.py`; the 0.8B fast track's two-bar rule)

| Tier | First bar (stored) | Second bar (M13 parity run) | Item-4 mlx-diag references |
| --- | --- | --- | --- |
| 4B | `bar-lh` = `formal/m10/m10-4b-LH-t1-derived` (the released LH, T = 1, v3 67.345) | `bar-f` = `m13-4b-LH` (node F, the LH soup, same weights) | `m10-4b-LH-mlx`; `m13-4b-LH-mlx` |
| 0.8B | `bar-t1` = `runs/release/dev2-0p8b-t1-derived` (v3 50.236) | `bar-e` = `m13-08b-C0` (node E, DEV2.0-0.8B's weights) | node A's `m2-E8F-soup-nodeA-mlx`; `m13-08b-C0-mlx` |

- 2B has no finalist (rules output 15:38Z), so no 2B path is added.
- Items 1 (v3 lower bound > 0), 2 (H upper bound ≥ 0), 4 (mlx-diag card-eligible upper bound ≥ 0), 6(b) (rules 1 and 5
  on the reduced panels) and 7 (public 231 not REGRESSION) must pass **against both bars**. Items 3 (types), 5 (tier
  gates vs the adopted 1.0 run and peers, as M6) and 6(a) are read once. m6-score.sh's own compares stay in the run
  directory; for 4B its `bar-t1` (DEV2.0-4B) is reported only.
- Parity: the parity run is scored first and repeated against the stored bar (`PARITY-EXACT.json`, 0 differing
  answers on typed FINAL, CSS15 and public 231 is "exact"). A non-exact parity run is reported; it does not replace
  either bar.
- Item 6(a): the finalist's M13 TRAIN file (data lock `debd36e1a`; 4B `4b-LHA10SD` `d41cdd1a…`, 0.8B `12bd63d8…` /
  `69b8d00d…`) streamed from its node, hash-checked, `overlap_effects exposure` against the r2 payload `2194716a…`. M13
  finalists contain no incumbent weight (each retrains from its Qwen base or released recipe), so they inherit no
  exposure. The self-distillation targets are soft labels on rows already in that TRAIN (no new text).
- Order per finalist: node F / E collection → `formal-pull` → `parity` (once per tier) → `run` → `formal-mark` →
  `m13-formal.sh mlx-launch` (node E / F, the finalist and the parity run) → `formal-pull` of the `-mlx` runs → `mlx`
  → `exposure` → `overlap` → `successor`. A failed step stops that point (no rerun).
- Item 8 stays the custodian's (C1 content recheck for IB1-r3 + IB2 first, worker 3c7679b0, then the item-8 spec).
