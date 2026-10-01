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

## Running now

- Step 0 (PN1-guard validation), node B, mirror `20af2e4a1`, launched 06:50Z (detached `m6-tail.sh slices`):
  - A20r on GPU0: driver PID 2744346, container `d2-27b-m6-A20r-slices`, log `m6/logs/slices-A20r.log`;
  - M5-L128 on GPU1: driver PID 2744347, container `d2-27b-m6-M5-L128-slices`, log `m6/logs/slices-M5-L128.log`.
  - Inputs: PN1 dev `c3b68ac1…` (node B HF cache, revision `27b1d2f1`); outputs `m6/slices/<name>/probs/`.
  - Then (host CPU): `m6-tail.sh pn1 <sha> M5-L128 A20r <pn1 rows>` → `m6/slices/M5-L128/pn1-vs-A20r.json`.

## Leases

node B GPU0, GPU1, GPU5 and node A GPU2: track 27b (from M5); GPU0 / GPU1 run step 0. Node D: IX1 follow-up (not
ours yet).

## Next steps

1. Score step 0; record the validation (prereg rule: validated if L128 − A20r clean gold-no Δ > 0).
2. Tooling: `m6_devgates.py`, build / arm / chain drivers, node link and relay; tests; mirror to node A / B.
3. When IB1-r3 is release-safe: data-lock amendment, build, IB1 DEV references, stage 1 launch (4 seeds) with a
   detached chain.

## Poll log (newest first)

- 06:53Z: prereg committed (`90d38aba7`); tooling `20af2e4a1` (slices, PN1 / breadth scoring, M6 allocations, tail
  driver; 13 new tests + the existing 27B suites pass) mirrored to node B; PN1 dev fetched on node B (hash equal);
  step 0 launched on node B GPU0 / GPU1 (containers up, first log lines present).
- 06:35Z: worker 1 started; integration merged (fast-forward to `50bae2ddd`); inputs read (M5 results, COORDINATION
  to 14:25, IB1 r2 records, IX1 public records and its private report). mlx-diag diagnosis of M5-L128 (node A, CPU, from
  the scored files): its Noul loss is a PAWS-X yes-bias (gold-no yes-rate .200 → .294), the 9B failure mode.
