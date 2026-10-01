# ~27B M6 state (resume file)

Updated: 2026-10-01 14:35 UTC+8 (06:35Z; M6 worker 1, started 06:17Z).
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

## Running now

Nothing (no GPU job).

## Leases

node B GPU0, GPU1, GPU5 and node A GPU2: `reserved-idle`, track 27b (from M5). Node D: IX1 follow-up (not ours yet).

## Next steps

1. Prereg `m6-prereg-2026-10-01.md` (before any GPU job).
2. Tooling: slices readout (PN1 dev, IB DEV), M6 gates, build / arm / chain drivers, tests; mirror to node A / B.
3. PN1-guard validation on node B (A20r vs M5-L128; L128 failed item 4 by a PAWS-X yes-bias).
4. When IB1-r3 is release-safe: data-lock amendment, build, stage 1 launch (4 seeds) with a detached chain.

## Poll log (newest first)

- 06:35Z: worker 1 started; integration merged (fast-forward to `50bae2ddd`); inputs read (M5 results, COORDINATION
  to 14:25, IB1 r2 records, IX1 public records and its private report). mlx-diag diagnosis of M5-L128 (node A, CPU, from
  the scored files): its Noul loss is a PAWS-X yes-bias (gold-no yes-rate .200 → .294), the 9B failure mode.
