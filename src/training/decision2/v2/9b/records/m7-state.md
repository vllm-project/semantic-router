# 9B M7 state (resume file)

Updated: 2026-09-30 ~10:50 UTC+8 (02:50Z) — prereg frozen; chains about to launch.
Branch: `xunzhuo/decision-2-training-9b` (merge-only into `xunzhuo/decision-2-training`).
Prereg: `records/lux9b-m7-prereg-2026-09-30.md`. Code mirror on node A: `7168f08643eb9297c7f457799dc06223056be656`
(created, verified). Runtime mirror for readouts / formal: `3277dec9d` (M6's, reused).

## Plan (see prereg)

- Arms: **P** = continuation of each of the five K seeds (M4 K-s1..s3, M6 K-s4/s5) on x60 replay + PN1-r2 ×2;
  **C** = matched-token continuation on x60 replay. `--init decision2`, lr 5e-6 / 5e-5, one final checkpoint,
  seeds 20260930 + k.
- Early rule after member 1 (PN1 dev clean gold-no −0.02 vs C-m1, hop ≥ −0.03, SELECT ≥ −0.02).
- Lines P5 / C5 at α ⅓ ½ ⅔; M6 floors + HT-DEV v2 non-FLAG + PN1 guard + MLX-DEV-9B guard; ≤ 2 finalists (P5, C5).

## Done (node A `/data/dev2/runs/9b/m7/`)

- MLX-DEV relayed from node B over the direct link (temporary key removed from both nodes): `mlxdev-src/`
  (`SHA256SUMS.nodeB` checked).
- PN1-r2 + PN1 dev at `27b1d2f1` fetched into node A's HF cache (hashes match the data record).
- Builds: `data/m7-topup/build` (manifest `b2390d28…`; P `e6a74ea9…` / C `eb55dbb2…`), `data/mlxdev/build`
  (panel `d07fe654…`, 6,147 rows). Exposure `exposure/P.json` `47153775…`, `exposure/C.json` `ae8fd45d…`: `groups: []`.
- CPU build attempts that failed before writing any output are kept under `logs/failed-builds/` (guard stat() of the
  spec note; replay overshoot; template segments; dict states; a7q emptied by prompt lines). Each was fixed by a
  commit (`cb55cf967`, `f8e108814`, `de9b1cef8`, `7168f0864`). No GPU time.

## Launch pattern (chain rule)

`bash $L/upload_chain.sh <node> <local chain> /data/dev2/runs/9b/m7/chains/<file>` (scp + size / SHA-256 check), then
in a separate ssh call `bash $L/launch.sh <chain> <remote file> <size> <sha> 7168f08643eb9297c7f457799dc06223056be656`
with `L=/data/dev2/src/7168f08643eb9297c7f457799dc06223056be656-src_training_decision2/src/training/decision2/v2/9b/lux9b/m7`;
liveness `bash $L/alive.sh <chain>` (PID + `docker ps` names `d2-9b-m7-*`, `dev2-9b-m7-*`), never `pgrep -f`.
GPU-hours: `bash -c '. $L/lib.sh; m7_gpu_hours'` on node A.

## Next

1. Launch `m7-gpu6` (GPU6) and `m7-gpu7` (GPU7).
2. After both chains end: `bash $L/rules.sh 7168f0864… rules-lines` (CPU) → `m7/rules/rules-lines/finalists.json`.
3. Lock record per finalist (commit + push) → upload and launch `chains/m7-post.sh SHA GPU NAME`.
4. Successor items 1–7 from `formal-m7/NAME.gates/successor.json` (+ `-16k-t1.gates` if T = 1 ships); item 8 hand-off.
5. Result record, gist 05, merge.
