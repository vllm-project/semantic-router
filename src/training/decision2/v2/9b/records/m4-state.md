# 9B M4 state (resume file)

Updated: 2026-09-29 12:10 UTC+8 (worker 3)
Branch: `xunzhuo/decision-2-training-9b` (merge-only into `xunzhuo/decision-2-training`)
Base: `eead88a60` + merge of origin `xunzhuo/decision-2-training` (`f3aae7155`) = `1424df742`

## Done

- Merged origin `xunzhuo/decision-2-training` (95 commits, no 9B file changes) -> `1424df742`.

## Running

- node A `post-gpu6` (P-a12 chain) and `post-gpu7` (KN-a12 chain): completion not yet verified.

## Next

1. Verify chain completion; collect every development readout (`/data/dev2/runs/9b/m4/` on node A).
2. Run `m4_rules.py` as preregistered -> alpha* per line, finalists (<= 3; priority K, U, D, KN, P), proxy drops.
3. Formal post-key runs for finalists (node A, 16K vs Lux1 16K 65.808 adopted / 65.231 same-renderer), mlx-diag, public 231, unless the chains already did.
4. Records + gist 05 entry; merge into `xunzhuo/decision-2-training`.

## Run dirs / hashes

(none yet)

## GPU-hours (this worker)

0
