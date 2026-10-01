# ~27B M6 state (resume file)

Updated: 2026-10-01 14:53 UTC+8 (06:53Z; M6 worker 1, started 06:17Z).
Prereg `m6-prereg-2026-10-01.md` (`90d38aba7`). Tooling `20af2e4a1` (mirrored to node B).
Assignment: COORDINATION 2026-10-01 14:25 (27B M6, worker 11741ee2). Branch `xunzhuo/decision-2-training-27b`
(worktree `/home/xunliu/code/vllm-sr-dev2-27b`; merge-only into `xunzhuo/decision-2-training`). Gist file
`06-decision-2-27b.md`. Budget 140 GPU-h. Index numbers are private: never in this file, commits or the gist.

## Goal

A successor to DEV2.0-27B (A20r, `main` `4e89288d`, post-key v3 72.36) that passes successor items 1–8 and also
improves the private Decision Index standing (deficit families: tool-call decisions, phishing; private report only).
Index runs only on frozen finalists, never for selection.

## Inputs (status)

- IB1: round 2 (`82bf70a7`) NOT release-safe; IB1-r3 in progress (data worker 1bad770e). Stage 1 waits for a
  release-safe IB1 record on the integration branch.
- IB2: being built (data worker 24a520c1). Stage 2 waits for it.
- IX1 follow-up (c0ce08eb): runtime fix + private M5-L128 Index diagnostic; holds node C / D GPUs until released.

## Done

- **Step 0, PN1-guard validation: VALIDATED → G5 is a gate.** Node B, mirror `20af2e4a1`, 06:49–06:54Z; A20r on GPU0
  (0.080 GPU-h), M5-L128 on GPU1 (0.086 GPU-h); PN1 dev `c3b68ac1…` (1,974 rows), raw T = 1; caches: no autotune
  entry added. `m6/slices/M5-L128/pn1-vs-A20r.json`:

  | PN1 dev yes-rate | A20r | M5-L128 | Δ [95%] |
  | --- | ---: | ---: | --- |
  | clean gold-no (850 rows) | .1541 | .1729 | **+.0188 [+.0095, +.0294]** |
  | hop, true paraphrases (236) | 1.000 | 1.000 | .000 [.000, .000] |
  | all eight languages | .5760 | .5866 | +.0106 |
  | accuracy | .9200 | .9103 | |

  The guard sees L128's yes-bias significantly, in the same direction as its formal mlx-diag failure (PAWS-X gold-no
  yes .200 → .294).

## Running now

Nothing (no GPU job).

## Leases

node B GPU0, GPU1, GPU5 and node A GPU2: track 27b, `reserved-idle`. Node D: IX1 follow-up (not ours yet).

## Next steps

1. Tooling: M6 gates / verdicts for formal finalists, the stage-1 chain, node A relay and mlx watchers, the node link;
   tests; mirror to node A / B.
2. When IB1-r3 is release-safe: data-lock amendment, build, IB1 DEV references (A20r, M5-L128), stage 1 launch
   (4 seeds) with detached chains.

## Poll log (newest first)

- 06:56Z: step 0 done: PN1 guard VALIDATED (L128 − A20r clean gold-no +.0188 [+.0095, +.0294]; hop level); 0.166
  GPU-h. Gates module `m6_devgates.py`, build / arm drivers written (not yet committed).
- 06:53Z: prereg committed (`90d38aba7`); tooling `20af2e4a1` (slices, PN1 / breadth scoring, M6 allocations, tail
  driver; 13 new tests + the existing 27B suites pass) mirrored to node B; PN1 dev fetched on node B (hash equal);
  step 0 launched on node B GPU0 / GPU1 (containers up, first log lines present).
- 06:35Z: worker 1 started; integration merged (fast-forward to `50bae2ddd`); inputs read (M5 results, COORDINATION
  to 14:25, IB1 r2 records, IX1 public records and its private report). mlx-diag diagnosis of M5-L128 (node A, CPU, from
  the scored files): its Noul loss is a PAWS-X yes-bias (gold-no yes-rate .200 → .294), the 9B failure mode.
