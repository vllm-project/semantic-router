# Decoder Milestone 8 — state (keep current; newest first)

Assignment: coordinator note 2026-09-30 17:00 (user plan decision: 4B M8 = cross-size distillation from DEV2.0-27B
A20r, matched controls, starts now, HR2 only by amendment). Budget 24 GPU-h. GPUs: node A GPU5, node B GPU3–4
(decoder), node A GPU0–1 shared leases for the A20r labels. Preregistration:
[`dec-m8-prereg-2026-09-30.md`](dec-m8-prereg-2026-09-30.md).

## Now

- 2026-09-30 ≈17:55 UTC+8 — Preregistration and tooling committed (worktree re-created from the pushed branch at
  `e066e0505`, fast-forwarded to integration `d4903065d`). Next: mirror to both nodes; Score5-typed-DEV prompts to
  node B; `m8-prep.sh` on node B (CPU); data lock part 1 (slice + C teacher) → C chains; label prompts to node A →
  three label shards; lock part 2 (A20r teachers) → D chains.

## Plan

| Step | Where | Status |
| --- | --- | --- |
| Data build (slice, S split, C teacher, prompts, exposure) | node B CPU | pending |
| A20r labels, 3 shards (+ parity smoke) | node A GPU0 / GPU1 (shared) / GPU5 | pending |
| Chains b3: C:1 D1:1 D1:2 D2:2; b4: C:2 C:3 D2:1 D2:3 D1:3 | node B GPU3 / GPU4 | pending |
| Lines L-D1 / L-D2 / L-C (α 1, ⅔, ⅓) + refs + hs1-dev | node B (co-tenant readouts) | pending |
| HT-DEV v2 / Score5-typed-DEV scoring, rules → finalists | node A CPU | pending |
| Formal (≤ 3), successor items 1–7, item-8 hand-off | node B → node A | pending |

## Liveness and operations

- Never `pgrep -f`; check PIDs (`chains/chain-b3.pid`, `label/logs/shard-<k>.pid`) and container names
  (`dec-m8-*`) with `docker ps`.
- Chain rule: upload/mirror, verify size + SHA-256, launch in a separate step, confirm the first log line.

## GPU-hours

| Item | GPU-h |
| --- | ---: |
| (none yet) | 0 |
