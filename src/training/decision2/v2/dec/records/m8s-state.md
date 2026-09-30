# Decoder M8-small — state (keep current; newest first)

Assignment: COORDINATION 2026-09-30 17:05 (decoder M8-small: DEV2.0-27B → DEV2.0-2B / DEV2.0-0.8B distillation),
GPUs per the 17:15 reclaim: **node B GPU6–7 plus spare GPU2** (never co-tenant a 27B job). Budget 24 GPU-h for both
tiers. Worktree `vllm-sr-dev2-dec-small`, branch `xunzhuo/decision-2-training-dec-small`, gist
`04b-decision-2-dec-small.md`. Records: [prereg](dec-m8s-prereg-2026-09-30.md) (`c99cf9b7f`),
[amendment 1](dec-m8s-amendment-1-2026-09-30.md) (`ada76c46e`), [data lock](dec-m8s-datalock-2026-09-30.md)
(part 1 `16044feab`).

## Now

- 2026-09-30 ≈09:58Z (17:58 UTC+8) — **Training live.** A20r parity gate PASS (698 / 698 CAL rows bit-exact). Label
  shards 0/2 (GPU6) and 1/2 (GPU7) running since 09:42Z (ETA ≈10:20Z). Control seeds 1 done: `2b-C-s1` (159 updates,
  0.124 GPU-h incl. preflights) and `08b-C-s1` (144 updates, 0.095); their early HT-DEV v2 collections done / running.
  References: `2b-I` and `08b-I` read at 16K; both equal the eval track's HT-DEV v2 reference collections exactly
  (Δ 0.0000, CI [0, 0]). The 0.8B node-B formal reference (`m8s-formal.sh ref08b`) is collecting on GPU2.
  - Next (manual): after both label shards, `m8s_lock.py teachers` + `check --part 2` → lock part 2 committed →
    READY for `teacher/{2b,08b}-{D1,D2}` → the D chains continue by themselves.
  - Erratum: the prereg header says "≈18:05 UTC+8" and amendment 1 "≈17:50"; the commits were at 09:30Z / 09:38Z
    (17:30 / 17:38 UTC+8). Both preceded every GPU job (first GPU job 09:39Z). Amendment 1 cites "09:44Z" for the
    L128 observation; it was ≈09:37Z.
- Incident (no GPU compute): the first `ref08b` attempt stopped at the eval adapter's argument check (`model_id`
  missing), 6.9 s on GPU2; moved to `m8s/formal/attempts/ref08b-1-model-id/`, fixed in `8e7a133be`, relaunched.

## Plan

| Chain | GPU | Items | Status |
| --- | --- | --- | --- |
| g2 | node B GPU2 | C arms (2B, 0.8B) + seed-1 HT-DEV v2 + C soups; co-tenants: references, `ref08b` | running |
| g6 | node B GPU6 | parity gate, label shard 0/2, D1 arms (early rule), D1 soups | shard 0 running |
| g7 | node B GPU7 | label shard 1/2, D2 arms (early rule), D2 soups | shard 1 running |

Mirrors on node B: `ada76c46e` (chains), `23e0b2e69` (lines), `8e7a133be` (formal).

## GPU-hours

Receipts: node B `/data/dev2/runs/dec/m8s/GPU-SECONDS.jsonl` (wall-clock × GPUs per job).

| Item | GPU-h |
| --- | ---: |
| **Total (cap 24)** | ≈0.5 so far |

## Hand-off checklist (for a continuation worker)

1. Read this file, the prereg, amendment 1 and `/data/dev2/runs/dec/m8s/OPERATIONS.log` on node B.
2. Chain liveness: `cat /data/dev2/runs/dec/m8s/chains/chain-*.pid` then `ps -p <pid>`; containers
   `docker ps --format '{{.Names}}' | grep '^dec-m8s-'` (never `pgrep -f`).
3. Markers: `/data/dev2/runs/dec/m8s/status/*.{DONE,FAILED,STOPPED}`, `early-*.{PASS,STOP}`, soups
   `m8s/soup/<tier>-<arm>/{DONE,FAILED}`; labels `m8s/labels/{parity.json,shard-*}`.
4. Lines: `M8S_GPU=<2|6|7> bash <mirror>/v2/dec/ops/m8s/m8s-lines.sh <tier> line <D1|D2|C>` after each soup, then
   `m8s_rules.py finalists --tier <tier> --lines-root /data/dev2/runs/dec/m8s/lines/<tier> --output
   /data/dev2/runs/dec/m8s/select/<tier>-finalists.json` (lines of stopped arms: `--dropped <ARM>=<reason>`).
5. Formal: `M8S_GPU=… m8s-formal.sh finalist <tier> <point>` and `mlx`; relay with `m8s-relay.sh pull <run>` (from
   the worktree); score on node A with `M6_FORMAL_ROOT=/data/dev2/runs/dec/formal/m8s ops/m6/m6-score.sh`.
