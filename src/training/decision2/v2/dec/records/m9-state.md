# Decoder Milestone 9 — state (keep current; newest first)

Assignment: COORDINATION 2026-10-01 10:30 (4B HR2 efficacy pilot, NOT releasable: full seeds from Nox + N4XF + an HR2
block vs a matched control, screened with HT-DEV v2). Budget 16 GPU-h. GPUs: node A GPU6–7 only (9B-owned, lent).
Preregistration: [`dec-m9-prereg-2026-10-01.md`](dec-m9-prereg-2026-10-01.md).

## Now

- 2026-10-01 ≈11:15 UTC+8 (03:15Z) — **Preregistered; no GPU job yet.** Next: M9 tooling (`ops/m9/`), mirror on
  node A, direct-link copy of the 4B recipe inputs and the node-B readout / formal autotune caches, data build and
  lock, then the chains on node A GPU6–7.

## Plan

| Step | Where | Status |
| --- | --- | --- |
| Prereg | workstation | done (this commit) |
| Tooling + tests, mirror | workstation → node A | pending |
| Inputs + caches over the direct link | node B → node A | pending |
| Data build (HR2 fetch, compose, teachers, exposure, lock) | node A CPU | pending |
| Chains: GPU6 H9-s1 → C9-s2 → H9-s3; GPU7 C9-s1 → H9-s2 → C9-s3; early rule after s1 | node A GPU6–7 | pending |
| References (`4b-I` on the M9 path + parity) | node A GPU6 / 7 (co-tenant) | pending |
| Lines (α 1, ½), diagnostics, scoring, rules | node A | pending |
| Formal (H9 pick only; parity run first) | node A | pending |

## Liveness and operations

- Never `pgrep -f`; check PIDs and container names (`dec-m9-*`) with `docker ps`.
- Chain rule: mirror, verify, launch in a separate step, confirm the first log line.

## GPU-hours

| Item | GPU-h |
| --- | ---: |
| **Total (cap 16)** | **0** |
