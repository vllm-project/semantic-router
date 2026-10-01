# Decoder Milestone 9 — amendment 2 (the H9 arm cap: 4.8 → 5.8 GPU-h from C9's unused budget; GPU6 chain replaced in flight)

Written 2026-10-01 ≈12:05 UTC+8 (≈04:05Z), **before any H9 readout** (no H9 checkpoint has been read on any panel;
the H9-s1 early read follows that seed's postrun). Preregistration `f22a5797c`; amendment 1 `e7f8e9858`.

## Why

- The preregistered per-arm cap (4.8 GPU-h) assumed ≈ 1.45 GPU-h per seed (M7's 1.35 at 43.5M tokens). H9 seeds are
  measuring ≈ 1.75–1.8 GPU-h each (46.2M tokens, 1,128 updates at ≈ 12 updates per minute, plus co-tenant readouts on
  the same GPU), so ≈ 5.3–5.4 GPU-h for three seeds.
- With the cap at 4.8, the GPU6 chain would refuse H9-s3 (arm used ≈ 3.5 with s2 still in its postrun, plus an
  expected ≈ 1.78 > 4.8), and the arm would end as a two-seed soup set against the three-seed control N7C. The
  preregistration allows a two-seed soup only when a seed fails or stops; here it would come from my under-estimate
  of the per-seed cost, and it would bias the H9 − N7C contrast against H9 (fewer members in the soup).
- C9's 4.8 GPU-h are unused (amendment 1). The milestone cap (16 GPU-h) and the 14 GPU-h stop for new seeds are
  unchanged.

## Change

- **H9 arm cap: 5.8 GPU-h** (was 4.8). No other rule changes. The cap is a cost control, not a selection rule, and it
  is set without any H9 readout.
- **Implementation (no job restarted):** the running GPU6 chain's code fixes the cap, so its bash, watchdog and flock
  processes are stopped while H9-s1's `drive_arm.sh` (and its training container) keep running. A new chain `g6b`
  from mirror `<this commit>` with `M9_CAP=5.8` takes the GPU6 lock once H9-s1's process tree exits, **adopts H9-s1**
  through `m9/chains/adopt-m9-H9-s1.pid` (it writes the same markers the old chain would have, then runs the seed-1
  early read), marks C9-s2 STOPPED (failed C9 preflight) and runs H9-s3. The GPU7 chain is unchanged (H9-s2, then
  C9-s3 STOPPED). The soup is built by whichever chain finishes H9's last seed.
