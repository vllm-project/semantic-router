# Decoder M8-small — state (keep current; newest first)

Assignment: COORDINATION 2026-09-30 17:05 (decoder M8-small: DEV2.0-27B → DEV2.0-2B / DEV2.0-0.8B distillation).
Budget 24 GPU-h for both tiers. GPUs: node B GPU5–7 (27B-owned, shared lease entries `owner.dec-m8s`) and GPU2
(spare). Worktree `vllm-sr-dev2-dec-small`, branch `xunzhuo/decision-2-training-dec-small`, gist
`04b-decision-2-dec-small.md`. Preregistration: [`dec-m8s-prereg-2026-09-30.md`](dec-m8s-prereg-2026-09-30.md).

## Now

- 2026-09-30 ≈18:55 UTC+8 — Preregistration and tooling committed (`ops/m8s/`, 11 tests). Next: mirror to node B,
  `m8s-prep.sh` (starts, top-up files, 2B control teacher, exposure), lock part 1, READY, chains g2 / g5 / g6 / g7.
  No GPU job yet. The 4B M8 prereg was not pushed at freeze time (λ rule in the prereg).

## Plan

| Chain | GPU | Items |
| --- | --- | --- |
| g2 | node B GPU2 | C arms (2B, 0.8B), their seed-1 HT-DEV v2 collections, C soups |
| g5 | node B GPU5 | A20r parity gate, label shard 0/3, D1 arms with early rule, D1 soups |
| g6 | node B GPU6 | label shard 1/3, D2 arms with early rule, D2 soups |
| g7 | node B GPU7 | label shard 2/3; then references (I), lines, formal (manual, committed wrappers) |

## GPU-hours

Receipts: node B `/data/dev2/runs/dec/m8s/GPU-SECONDS.jsonl` (wall-clock × GPUs per job).

| Item | GPU-h |
| --- | ---: |
| **Total (cap 24)** | **0** |

## Hand-off checklist (for a continuation worker)

1. Read this file, the prereg and `/data/dev2/runs/dec/m8s/OPERATIONS.log` on node B.
2. Chain liveness: `cat /data/dev2/runs/dec/m8s/chains/chain-*.pid` then `ps -p <pid>`; running containers
   `docker ps --format '{{.Names}}' | grep '^dec-m8s-'` (never `pgrep -f`).
3. Markers: `/data/dev2/runs/dec/m8s/status/*.{DONE,FAILED,STOPPED}`, `early-*.{PASS,STOP}`, soups
   `m8s/soup/<tier>-<arm>/{DONE,FAILED}`.
