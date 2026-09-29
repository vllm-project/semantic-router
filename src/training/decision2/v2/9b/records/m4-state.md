# 9B M4 state (resume file)

Updated: 2026-09-29 12:10 UTC+8 (worker 3)
Branch: `xunzhuo/decision-2-training-9b` (merge-only into `xunzhuo/decision-2-training`)
Base: `eead88a60` + merge of origin `xunzhuo/decision-2-training` (`f3aae7155`) = `1424df742`

## Done

- Merged origin `xunzhuo/decision-2-training` (95 commits, no 9B file changes) -> `1424df742`.
- Step 1 (12:15): `post-gpu6` ended 03:52:38Z, `post-gpu7` ended 04:02:18Z, every step exit 0.
  All 29 dev readouts complete (`<run>-{cal,dev,css-pilot}`, exit 0). Node-A re-reads of
  node-B seeds identical (paired delta 0). No formal run yet (`/data/dev2/runs/9b/formal-m4` absent).
- Scores already on node A (written outside the chains, likely worker 2): `m4/readout-{seeds,dline,lines}/readout.json`,
  `m4/rules/seed-{K,P,KN}.json` (soup wins: K 67.23 vs 64.12, P 68.86 vs 63.17, KN 69.37 vs 67.46),
  `m4/rules/alpha.json` (unverified: K 1/3, U 1/3, D 1/2 = DW, KN 1/2, P 1/2; no proxy drop).

## Running

- nothing (node A GPU6-7 and node B GPU0-2 idle at 12:15).

## Next

1. (done) chains + readouts.
2. Independently re-run `m4_rules.py` (seed x3 + alpha) as preregistered into `m4/rules/verify-w3/`, diff vs `m4/rules/`;
   expected finalists K-a13, U-a13, KN-a12 (D slot passes: alpha*_D = 1/2 is DW, formally run in M3).
3. Formal post-key runs for finalists (node A, 16K vs Lux1 16K 65.808 adopted / 65.231 same-renderer), mlx-diag, public 231, unless the chains already did.
4. Records + gist 05 entry; merge into `xunzhuo/decision-2-training`.

## Run dirs / hashes

(none yet)

## GPU-hours (this worker)

0
