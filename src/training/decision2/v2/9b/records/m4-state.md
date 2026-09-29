# 9B M4 state (resume file)

Updated: 2026-09-29 12:55 UTC+8 (worker 3)
Branch: `xunzhuo/decision-2-training-9b` (merge-only into `xunzhuo/decision-2-training`)
Base: `eead88a60` + merge of origin `xunzhuo/decision-2-training` (`f3aae7155`) = `1424df742`

## Done

- Merged origin `xunzhuo/decision-2-training` (95 commits, no 9B file changes) -> `1424df742`.
- Step 1 (12:15): `post-gpu6` ended 03:52:38Z, `post-gpu7` ended 04:02:18Z, every step exit 0.
  All 29 dev readouts complete (`<run>-{cal,dev,css-pilot}`, exit 0). Node-A re-reads of
  node-B seeds identical (paired delta 0). Readouts: node A `m4/readout-{seeds,dline,lines}/readout.json`.
- Step 2 (12:35): `m4/rules/{seed-K,seed-P,seed-KN,alpha,finalists}.json` (written by worker 2) verified by an independent
  re-run into `m4/rules/verify-w3/`: byte-identical (alpha `258835e7c7cc`, seed-K `51c559bce1fd`, seed-P `0e0d6d53af51`,
  seed-KN `0e9237c608a4`); key->run mapping checked; alpha*_D = 1/2 reproduced; 8/8 unit tests.
  Seed rule: soup wins (K 67.23 vs 64.12, P 68.86 vs 63.17, KN 69.37 vs 67.46).
  alpha*: K 1/3, U 1/3, D 1/2 (= DW), KN 1/2, P 1/2; no proxy drop (best 76.06).
  Finalists (priority K, U, D, KN, P; D slot passes since DW ran in M3): **K-a13, U-a13, KN-a12**.
- Lock pushed `d9c04cbda` (`records/lux9b-m4-formal-lock-2026-09-29.md`) before any formal prediction. sha256 prefixes:
  K-a13 `b9d973b3ef55`, U-a13 `8c1ab3719947`, KN-a12 `e0dffa0ec943`; cache tree `af623300d71a`.

## Running (step 3, launched per lock; do NOT relaunch - `formal.sh` exits 66 on existing dirs)

- node A GPU6 driver PID 2828062 (04:46:41Z): K-a13, then KN-a12 if K-a13 exits 0. GPU7 driver PID 2829031 (04:47:24Z): U-a13.
- Poll: `/data/dev2/runs/9b/m4/logs/formal-gpu{6,7}.{log,console}` (`end step=NAME ... exit=N`, `done NAME`);
  outputs `/data/dev2/runs/9b/formal-m4/NAME-{smoke,16k,16k-mlx}`, `NAME-cache.jsonl`.
- Both smokes exit 0; hashes / image match the lock. Cache tree hash changed during the smokes `af623300d71a` ->
  `d10b24d792c8` (4,785 files; both 16k runs started from `d10b24d7`): report, do not rerun.

## Next

3. After all three end: `v2.eval.gates types` per finalist (lock bash block); read `NAME-16k/PAIRED-vs-Lux1-16K.json`
   (`ci95.low`, `axis_ci95.H.delta.high`) and `-shared`; mlx-diag; public 231; SHA-256 manifests next to each soup.
4. Result record + gist 05 entry; merge into `xunzhuo/decision-2-training`.

## GPU-hours

M4 before formal runs: 19.31 GPU-h (node-A run records incl. node-B copies, CPU soups excluded); cap 22.
