# Decoder Milestone 9 — state (keep current; newest first)

Assignment: COORDINATION 2026-10-01 10:30 (4B HR2 efficacy pilot, NOT releasable: full seeds from Nox + N4XF + an HR2
block vs a matched control, screened with HT-DEV v2). Budget 16 GPU-h. GPUs: node A GPU6–7 only (9B-owned, lent).
Preregistration [`dec-m9-prereg-2026-10-01.md`](dec-m9-prereg-2026-10-01.md) (`f22a5797c`); data lock
[`dec-m9-datalock-2026-10-01.md`](dec-m9-datalock-2026-10-01.md) (`2f20c2223`); amendments
[1](dec-m9-amendment-1-2026-10-01.md) (`e7f8e9858`) and [2](dec-m9-amendment-2-2026-10-01.md) (`29057a4de`);
**results [`dec-m9-results-2026-10-01.md`](dec-m9-results-2026-10-01.md)**.

## Now

- 2026-10-01 14:20 UTC+8 (06:20Z) — **M9 DONE: HR2 is not a human-transfer lever at 4B; no formal run.**
  - HT-DEV v2, H9 soup − control (M7's N7C) soup: −.004 [−.018, +.010] TIE; H9 vs DEV2.0-4B −.012 [−.025, +.001]
    TIE (α ½ −.003 TIE). No H9 point passed the gates (typed Noul floor: `rule_precedence` 241 / 258 < 260), so no
    formal run (prereg). HR2 DEV +.140 (in distribution). Preregistered verdict: not a lever; no HR2-r2 RC milestone
    at 4B; 2B / 0.8B / 9B priors negative.
  - 5.94 of 16 GPU-h. Nothing uploaded; C1 not opened; no hand-off. No M9 job or container is running; node A GPU6–7
    leases set idle and returned to the 9B track (03:57Z chain replacement and all markers recorded in
    `m9/OPERATIONS.log`).
  - Node artifacts stay on node A under `/data/dev2/runs/dec/m9/` (H9 soup `m9/soup/H9/build/H9-soup`, list
    `49783cc3…`; N7C copy `m9/control/N7C-soup`; formal-path parity `m9/formal/m9-ref-N4XF`).

## History (newest first)

- 06:13Z post chain: L-H9 read and scored; rules → no H9 pick (Noul floor) → STOPPED, no formal run.
- 05:57Z H9 soup built (BEST 846 / 847 / 1121); g6b finished; C9 no soup.
- 04:33Z g6b adopted H9-s1 (DONE, BEST 846) and started H9-s3; 04:36Z H9-s2 DONE (BEST 847); g7 finished.
- 03:57Z amendment 2 applied: GPU6 chain replaced by g6b (cap 5.8), H9-s1 kept running.
- 03:50Z control line L-N7C read (α 1 −.008 TIE, typed floors fail; α ½ +.001 TIE, passes).
- 03:30Z formal-path parity exact (0 category changes vs the bar; v3 63.151).
- 03:12Z readout-path parity exact (5 panels, 0 answer differences; HT-DEV v2 = eval reference).
- 03:10Z amendment 1 (control = N7C) after C9-s1's preflight FAIL at 03:04Z.
- 02:58Z chains launched; 02:55Z data lock PASS; 02:37Z prereg.

## GPU-hours

| Item | GPU-h |
| --- | ---: |
| H9 training (3 seeds, incl. preflights / postruns) | 4.460 |
| C9-s1 (failed preflight) | 0.100 |
| Reads (references, control line, H9 line, early read) | 1.031 |
| Formal-path parity (v3 + mlx-diag) | 0.348 |
| **Total (cap 16)** | **5.939** |
