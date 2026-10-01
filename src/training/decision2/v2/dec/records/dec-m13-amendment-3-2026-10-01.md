# Decoder Milestone 13 — amendment 3 (4B mlx-diag launched from the wrong mirror; 2026-10-01)

Committed before the 4B mlx-diag collections are relaunched. Amendment 2
[`dec-m13-amendment-2-2026-10-01.md`](dec-m13-amendment-2-2026-10-01.md) added the mlx-diag step.

## What happened

At 15:53:58Z `m13-formal.sh mlx-launch` ran with the amendment-2 mirror (`0ea6b3e62`) as its source for both 4B
points (node F GPU6 `m13-4b-LH`, GPU7 `m13-4b-LHA10SD`). The M6 library replays each run's collection arguments from
its receipt, and those name the adapter spec inside the collection mirror (`c2610143b`). The container mounts only
the source mirror, so `same_panel collect` stopped at the spec load (`FileNotFoundError ... c2610143b.../m5-adapter-
infer-dec-t1.json`) within a second, before any inference. Both points were marked `MLX-FAILED`.

No mlx-diag prediction exists for either point; nothing was measured. The v3 reports, seals and every node-A score
are unchanged.

## Change

- `m13-formal.sh mlx-launch` is rerun with `c2610143b` (the mirror that collected the runs) as `<mirror-dir>`; the
  wrapper itself is the amendment-2 copy. `ops/m6/` is byte-identical in both mirrors.
- The failed attempt's `-mlx` run and cache directories, collect logs, `MLX-FAILED` markers and launch locks move to
  `formal/m13/logs/amendment-3/` (kept, not deleted).
- Every later mlx-diag launch uses the collecting mirror. A failure of the relaunched collections is final.
