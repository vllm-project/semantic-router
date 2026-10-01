# Decoder Milestone 10 — amendment 3 (NT2's formal step refused by a finished runner's lease entry; one run)

Written 2026-10-01 ≈14:15 UTC+8 (06:15Z), **before any NT2 formal output exists**. Preregistration `2dea44d6f`;
amendments 1 (`04088322b`) and 2 (`88aacb9fb`).

## What happened

- The wave-2 rules (`m10/select/4b-finalists-w2.json`, node A, 06:06Z) passed NT2 on all five development gates
  (HT-DEV v2 −.006 [−.021, +.008] TIE; Score5t clean; typed floors pass; retention level with C0). Under amendment 2
  it takes the free second finalist slot.
- `m10-formal.sh` (node F GPU3, 06:08:54Z) stopped NT2's smoke at its first action: the M6 library's CAL698-fit lease
  entry refused to start because `gpu3.lock/owner.dec-formal` still held the **finished** entry of LH's mlx-diag
  runner (track `dec`, `last_job_end_utc=05:26:46Z`, exit 0). No container started, no GPU second was used, and no NT2
  calibration or prediction exists (`formal/m10/logs/m10-4b-NT2-smoke.log`).
- This is the M6b lease incident: M8's formal wrapper moves such finished decoder entries aside before each step;
  `m10-formal.sh` did not.

## Amendment

- The refusal is recorded (`status/m10-4b-NT2.FAILED` moved to `logs/attempts/`). Since no NT2 measurement was
  taken, this is not a failed arm or a rerun of a measured step: NT2's formal runs **once**, unchanged, after
  `m10-formal.sh` gains M8's stale-entry clearing (finished `track=dec` runner entries with a `last_job_end_utc`
  line, and no `dev2-dec-gpu<N>-` container on the GPU, move to `formal/m10/logs/stale-leases/` before each step).
- Everything else is unchanged: the same package staging, 23:15 rule, smoke, collection, mlx-diag and successor
  items as LH.
