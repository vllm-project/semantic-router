# 9B M10 amendment 8: the arm factory's unmeasured 9B arm points (2026-10-03)

Written 2026-10-02 ≈17:05Z (10-03 01:05 UTC+8) by the M10 continuation worker, before any arm-factory 9B soup is
built and before any arm-factory 9B Index result exists. No X7-a40 or X8-a40 Index result exists yet either.
Amendment 7 (`d802c97c4`) is unchanged, except as noted here.

## Why

The factory's node A measurement queue holds only `KF-a40`, `KF-a50` and `KFxKIB-a40`. It builds `KIB4W2-a40` and
`KIB4L2-a40` (each arm's two-seed soup at α = 2/5) and hands them over unmeasured. They are the only points that
test the two new recipes alone. KIB4W2 doubles the loss weight of the IB4 phase-1 rows, and IB4 is the data change
behind KIB4's gain over K-a13IB. In `KF` both recipes are diluted to 2/9 of the soup.

## Candidates

- **`KIB4W2-a40`** and then **`KIB4L2-a40`**: the factory's points, linked read-only into the M10 soup tree
  (`m10/xpts.sh link`, equal SHA-256 lists). Their BF16 release copies are made by `ix.sh bf16`. The IX1 names are
  `M10-KIB4W2-a40-bf16` and `M10-KIB4L2-a40-bf16`.
- They are measured once each, on M10 GPUs (node A GPU7, node E GPU3 / 7 after X8-a40, node B GPU7 after X7-a40), in
  that order, as soon as each soup exists. The gate is amendment 7's IF1 vs `M10-KIB4-a40-bf16`.
- If the factory measures one of them after all, that run is used and M10 does not measure it.
- F* (amendment 7, `Y1` / `Y2`) ranges over every measured arm-factory point, these two included.
- **Audit:** both trained on KIB4's TRAIN (`2e72bcfd…`), which audit `m10c` covers (0 item rows).

## Budget

The two runs cost about 2.7 GPU-h each, within amendment 7's 30 GPU-h. The 27 GPU-h stop rule is unchanged.
