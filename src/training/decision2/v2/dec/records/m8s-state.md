# Decoder M8-small — state (keep current; newest first)

Assignment: COORDINATION 2026-09-30 17:05 (decoder M8-small: DEV2.0-27B → DEV2.0-2B / DEV2.0-0.8B distillation),
GPUs per the 17:15 reclaim: node B GPU6–7 plus spare GPU2. Budget 24 GPU-h. Worktree `vllm-sr-dev2-dec-small`, branch
`xunzhuo/decision-2-training-dec-small`, gist `04b-decision-2-dec-small.md`. Records: [prereg](dec-m8s-prereg-2026-09-30.md)
(`c99cf9b7f`), amendments [1](dec-m8s-amendment-1-2026-09-30.md) / [2](dec-m8s-amendment-2-2026-09-30.md) /
[3](dec-m8s-amendment-3-2026-09-30.md), [data lock](dec-m8s-datalock-2026-09-30.md), **[results](dec-m8s-results-2026-09-30.md)**.

## Now

- 2026-09-30 ≈11:55Z (19:55 UTC+8) — **M8-small DONE: no successor in either tier; DEV2.0-2B `a53cf66a` and
  DEV2.0-0.8B `bede7938` stand.** No GPU job running; the M8s lease entries on node B GPU2 / 6 / 7 are released (the
  GPUs go back to their owner, the ~27B track). Nothing uploaded; C1 not opened; no item-8 hand-off.
  - 2B: no finalist (every point fails the typed-DEV Score floor: `set_reconciliation` 172–216 vs 245, identical in
    D1 / D2 / C at equal α).
  - 0.8B: finalists `08b-D2-a1_2` (+0.18 [−0.52, +2.44] vs the node-B bar) and `08b-C-a1_3` (−0.09 [−0.89, +1.20])
    fail item 1; items 2–5 and 7 pass (mlx-diag card-eligible +.0075 / +.0057, significant).
  - ≈4.45 of 24 GPU-h.
- 2026-09-30 ≈10:20Z — D arms training; control arms done; lines being read (see the results record).
- 2026-09-30 ≈09:58Z — Training live; A20r parity gate PASS (698 / 698 CAL rows bit-exact).

## Where things are (node B `/data/dev2/runs/dec/m8s`, node A `/data/dev2/runs/dec/formal/m8s`)

- Starts (verified HF downloads): `start/dev2-2b-a53cf66a`, `start/dev2-0p8b-bede7938`.
- Top-up files `data/<tier>/topup`, teachers `teacher/<tier>-{C,D1,D2}`, A20r labels `labels/shard-{0,1}-of-2.jsonl`.
- Arm runs `arms/<tier>-<arm>-s<i>`, soups `soup/<tier>-<arm>/build/<tier>-<arm>-soup`, lines `lines/<tier>/`,
  selections `select/<tier>-finalists.json`.
- 0.8B formal: node-B reference `formal/m8s-ref-08b-I` (+ `-mlx`), frozen caches `formal/cache-frozen-08b(-mlx)`,
  finalist packages `formal/pkg/m8s-08b-*`; node A scoring and successor files under `formal/m8s/`.

## GPU-hours

| Item | GPU-h |
| --- | ---: |
| Labels (parity + 2 shards) | 1.014 |
| Arms (6 × 3 seeds) | 1.529 |
| Early collections | 0.104 |
| Lines, references, diagnostics | 1.42 |
| Formal | 0.381 |
| **Total (cap 24)** | **≈4.45** |

## Hand-off notes

- No release hand-off. If a later milestone retests 2B distillation, use a proportional top-up (the 4B M8 slice
  design): the 1:1 human / typed file shifted the shared Score head (the 2B Score loss appears in the control too).
- 0.8B node-B formal work: use `formal/cache-frozen-08b` with the node-B bar (the node-A binding differs in 30
  categories on this path).
