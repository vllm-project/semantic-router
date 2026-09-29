# ~27B M3 state (resume file)

Updated: 2026-09-29 15:10 UTC+8 (F2 completion worker; fresh worker after the 14:25 broken-chain note)
Branch: `xunzhuo/decision-2-training-27b` (merge-only into `xunzhuo/decision-2-training`)
Scope: finish F2 exactly as preregistered (amendment 4), mlx-diag for F1 and F2, then apply the coordinator's
F1/F2 rule. Node B GPU5 / GPU7 (27B leases); GPU6 left free.

## Done

- 14:28 (06:28Z): phase 1 launched through `v2/27b/m3f2/f2-phase1.sh`. Upload checked by size and SHA-256
  (`219bf6c2…`); both jobs started (containers up, first log lines present).
  - M3-S-s2 kernel readout at 32,768: exit 0 at 06:46Z.
  - M3-S soup: exit 0 at 06:34Z; max relative diff 2.6e-7; model `d6ad230a…`.
  - Soup kernel readout: exit 0 at 06:48Z.
- 15:04 (07:04Z): development contrast written to `m3-contrast/contrast.json` (`score_m3.sh`, mirror `35fa052d2`).
  - Soup rule: M3-S soup P_dev 75.20 ≥ seed mean 74.42, so **F2 = M3-S-soup**.
  - Proxy screen: kept (1.83 behind F1's 77.03).
- 15:05 (07:05Z): CAL698 fit and adoption running on GPU5 (default frozen cache `583241fb`, as F1).
- Incident recorded in the results record, section "Incident". Cause: upload and launch were sent in one ssh
  command ending in `&`, so `cat` read `/dev/null`.

## Next

1. Push amendment 4.
2. Snapshot F1's post-run cache (`03b172f1…`).
3. `f2-phase2.sh formal 5` (package, smoke, collection, score; paired vs 3 peers and F1), then the M2-S s1 compare.
4. `f2-phase2.sh gates`, then `f2-phase2.sh overlap <spec>`.
5. mlx-diag for F1 and F2 in one pass (`f2-mlx.sh`, smoke first), scored on node A (`f2-mlx-score.sh`).
6. Apply the F1/F2 rule; update the results record, this file and gist 06; merge.

## Paths (node B)

- Runs: `/data/dev2/runs/27b/{M3-S-s2,M3-S-soup,m3-contrast,m3-f2}`.
- Driver logs: `/data/dev2/runs/27b/m3-logs/f2-*`, `M3-S-soup.driver.log`.
- F1: `/data/dev2/runs/27b/M3-A-soup/formal` (SEAL `42ba2fa1…`).
