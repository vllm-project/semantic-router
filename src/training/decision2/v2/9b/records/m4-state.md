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

- Step 2 (12:35): independent re-run of `m4_rules.py` into node A `m4/rules/verify-w3/` is byte-identical to
  `m4/rules/` (alpha `258835e7c7cc`, seed-K `51c559bce1fd`, seed-P `0e0d6d53af51`, seed-KN `0e9237c608a4`); key->run
  mapping checked; alpha*_D = 1/2 reproduced; 8/8 unit tests. alpha*: K 1/3, U 1/3, D 1/2 (= DW), KN 1/2, P 1/2;
  no proxy drop (best 76.06). Finalists (K, U, D, KN, P priority; D slot passes): **K-a13, U-a13, KN-a12**.
- Lock: `records/lux9b-m4-formal-lock-2026-09-29.md` (sha256 prefixes: K-a13 `b9d973b3ef55`, U-a13 `8c1ab3719947`,
  KN-a12 `e0dffa0ec943`; cache tree `af623300d71a`).

## Next

3. Launch formal runs per the lock's Launch section (GPU6: K-a13 then KN-a12; GPU7: U-a13 after `formal-m4/triton-cache` exists);
   then `v2.eval.gates types` per finalist; read `PAIRED-vs-Lux1-16K.json` (`ci95.low`, `axis_ci95.H.delta.high`).
3. Formal post-key runs for finalists (node A, 16K vs Lux1 16K 65.808 adopted / 65.231 same-renderer), mlx-diag, public 231, unless the chains already did.
4. Records + gist 05 entry; merge into `xunzhuo/decision-2-training`.

## Run dirs / hashes

(none yet)

## GPU-hours (this worker)

0
