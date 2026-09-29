# 9B M4 state (resume file)

Updated: 2026-09-29 13:40 UTC+8 (worker 3; M4 complete)
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

- Step 3 (13:15): formal runs ended exit 0 (K-a13 04:58:24Z, U-a13 04:58:55Z, KN-a12 05:09:44Z); identities match the lock;
  types gate run (host python3, lock loop). Post-key same-panel v3, paired vs native Lux1 16K 65.808 [95% CI]:
  - **K-a13 67.737, +1.929 [+0.607, +4.144]; H +.0081 [-.011, +.042]; types OK x3 -> PASS.**
    Scored run: node A `/data/dev2/runs/9b/formal-m4/K-a13-16k` (SEAL `11abc1cc7818`).
  - U-a13 67.866, +2.058 [-0.215, +4.040]; H +.0103 [-.024, +.041]; types OK x3 -> FAIL (lower bound <= 0).
  - **KN-a12 67.982, +2.174 [+0.358, +4.962]; H +.0057 [-.022, +.049]; types OK x3 -> PASS.**
    Scored run: node A `/data/dev2/runs/9b/formal-m4/KN-a12-16k` (SEAL `92084d7ed1e0`).
  - vs same-renderer 65.231: K-a13 +2.507 [+1.037, +4.599], U-a13 +2.636 [+0.265, +4.496], KN-a12 +2.751 [+0.733, +5.511].
- Kept on node A (no HF upload): `m4/NAME-build/soup` + `m4/NAME-cal/`, manifest `m4/NAME-build/SHA256SUMS`
  (23 files, re-hash 23/23 OK): K-a13 `6913eb61836a`, U-a13 `d426c1a4f18f`, KN-a12 `1159daf23803`.
- Cache: 13 Triton `__grp__*.json` files rewritten with `formal-m4` paths (kernels, `.hsaco`, autotune JSON byte-identical);
  final tree `5604ffdc5f19`. Scratch readout: node A `formal-m4/readout-w3/`.

- Step 4a (13:30): result record `records/lux9b-m4-formal-result-2026-09-29.md` (+ README m4_rules sentence).
  Prereg is silent on several passers: K-a13 and KN-a12 both go to the coordinator (amendment 1: report at once;
  upload only a coordinator-approved release candidate, by release engineering, after `hf_headroom.sh`).

- Step 4b (13:40): gist 05 entry "2026-09-29 13:30" posted above 08:35 (gist revision `313b507b044a`).
- Step 4c: this state commit merged into `xunzhuo/decision-2-training` (merge-only; see `git log` of that branch).

## Running

- nothing. Node A GPU6-7 idle (leases still `track=9b-clm status=reserved`, as `chain-step.sh` leaves them);
  node B GPU0-2 idle (leases `status=idle`).

## Next (M4 complete)

- Coordinator: choose between K-a13 (first in priority, higher lower bound) and KN-a12 (higher v3); the prereg is silent.
  Only a coordinator-approved release candidate is uploaded, by release engineering, after `hf_headroom.sh` (amendment 1).

## GPU-hours

M4 total 19.86 GPU-h = 19.31 (training + readouts, node-B copies in, CPU soups out) + 0.55 formal (1,987 GPU-s); cap 22.
