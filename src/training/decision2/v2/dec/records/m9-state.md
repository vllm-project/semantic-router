# Decoder Milestone 9 — state (keep current; newest first)

Assignment: COORDINATION 2026-10-01 10:30 (4B HR2 efficacy pilot, NOT releasable: full seeds from Nox + N4XF + an HR2
block vs a matched control, screened with HT-DEV v2). Budget 16 GPU-h. GPUs: node A GPU6–7 only (9B-owned, lent).
Preregistration: [`dec-m9-prereg-2026-10-01.md`](dec-m9-prereg-2026-10-01.md) (`f22a5797c`); data lock
[`dec-m9-datalock-2026-10-01.md`](dec-m9-datalock-2026-10-01.md) (`2f20c2223`, PASS). Gist 04 entry 11:15.

## Now

- 2026-10-01 ≈11:05 UTC+8 (03:05Z) — **Chains running.** Mirror `e9a357ed408cd569fa41b436c9e578354beaa99a-src_training_decision2`
  on node A (139 tests pass in the image). Chains launched 02:58Z: `m9/chains/chain-g6.pid` (4081925),
  `chain-g7.pid` (4081939); H9-s1 / C9-s1 passed zero-step, one-step running (containers `dec-m9-H9-s1-*`,
  `dec-m9-C9-s1-*`). Expected seed ≈ 1.45 h; E1 ≈ 04:35Z; soups ≈ 07:35Z.
  - `m9-lines.sh refs` on GPU6 (co-tenant, started 03:00Z, log `m9/logs/refs.log`): 4b-I HT-DEV v2 read; typed DEV
    reading; then CSS pilot, Score5-typed-DEV, HR2 DEV, hs1-dev, scoring and node-B parity.
  - Known: the chains' copy of `m9_gpuh.py` also counts the copied M7 parity receipts (+0.226 GPU-h, conservative for
    the stop rules); fixed in the next mirror for reporting.

## Plan

| Step | Where | Status |
| --- | --- | --- |
| Prereg | workstation | done `f22a5797c` |
| Tooling + tests, mirror | workstation → node A | done `e9a357ed4` |
| Inputs + caches over the direct link | node B → node A | done (content manifests equal; formal caches `f6d0f920…` / `65d7d38f…`) |
| Data build + lock | node A CPU | done 02:55Z, PASS |
| Chains: GPU6 H9-s1 → C9-s2 → H9-s3; GPU7 C9-s1 → H9-s2 → C9-s3; E1 after s1 | node A GPU6–7 | running since 02:58Z |
| References (`4b-I` on the M9 path + parity) | node A GPU6 (co-tenant) | running |
| Lines (α 1, ½), diagnostics, scoring, rules | node A | after the soups |
| Formal (H9 pick only; parity run first) | node A | tooling to be committed before use |

## Operations (node A, from the mirror `S=/data/dev2/src/<mirror>/src/training/decision2`)

- Liveness: `cat /data/dev2/runs/dec/m9/chains/chain-g{6,7}.pid` + `ps -p`; `docker ps | grep dec-m9`. Never `pgrep -f`.
- Logs: `m9/OPERATIONS.log`, `m9/logs/chain-g{6,7}.log`, `m9/arms/OPERATIONS.log`, `m9/lines/4b/OPERATIONS.log`.
- Early rule: written by the chains to `m9/early/E1.json` (`m9-lines.sh early <ARM>` then `e1`).
- After both soups (`m9/status/{H9,C9}.DONE`): `M9_GPU=6 m9-lines.sh line H9` and `M9_GPU=7 m9-lines.sh line C9`
  (parallel), then `m9-lines.sh score` and `m9-lines.sh rules` → `m9/select/4b-pick.json`.

## GPU-hours

| Item | GPU-h |
| --- | ---: |
| **Total (cap 16)** | running |
