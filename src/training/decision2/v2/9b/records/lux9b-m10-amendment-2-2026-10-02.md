# 9B M10 amendment 2: KSW teacher rebuilt at K-a13IB's member paths (one engineering repeat; 2026-10-02)

Written ≈13:10 UTC+8 (05:10Z) by worker 7e1c9ce8, before any KSW teacher target was computed and before any KSW GPU
job ran.

- **What failed:** amendment 1's teacher-identity check. The node-B rebuild `[KIB soup, Lux zero-step, Lux
  zero-step]` has model SHA-256 `6edf3062…`, not K-a13IB's `4701ba41…`.
- **Cause (diffed file by file against node A's K-a13IB):** every weight file, the head (`37567454…`) and the
  members' identities (`794ebfd2…`, `45fa6e49…` twice) are byte-identical. Only `decision_config.json` differs, in the
  three member `path` strings it records (`/runs/m10/...` instead of M9's `/runs/m9/...`). The model SHA-256 covers
  that file.
- **Repeat (once):** the members are hard-linked on node B at M9's container paths (`/runs/m9/soup/KIB/build/KIB-soup`,
  `/runs/m9/arms/pre/m9-KIB-s1-zero/checkpoint-0000000`) and the soup is rebuilt; the check is unchanged (model
  SHA-256 must equal `4701ba41…`). The failed build stays in `ksw/teacher-try1`. A second failure stops the arm.
- The KSW data build (TRAIN `e5cc44bb…`: 146,600 rows, 104,565 x60 rows kept, including the 54,837 rows of the
  35,329 groups with a non-English row (kept whole), 60,272,054 native tokens, IB share .1567, 6,808 `sentfin` rows dropped) is unaffected.
