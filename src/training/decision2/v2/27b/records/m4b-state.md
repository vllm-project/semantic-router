# ~27B M4b state (resume file)

Updated: 2026-09-29 20:50 UTC+8 (resumed M4b worker; **A1-s1 done; A2-s1 full running**)
Branch: `xunzhuo/decision-2-training-27b-m4b` (worktree `/home/xunliu/code/vllm-sr-dev2-27b-m4b`; merge-only into
`xunzhuo/decision-2-training`). Gist file: `06b-decision-2-27b-m4b.md`. GPUs: node B GPU0–2 only (lent by 9B;
leases now `track=27b-m4b`, previous owners kept as `owner.prev-20260929T094419Z`). Budget 36 GPU-h.
Milestone 4 (other worker; node A GPU2–4 + node B GPU5–7; `m4-*` files/dirs) is not touched.

## Plan (prereg `m4b-prereg-2026-09-29.md`, amendments 1–4)

- A1 FF-gold (full-parameter FSDP ×3 on F1's exact file `de00df03…`, s1/s2), A2 FF-Lux (+1.0·KL own-Lux on S, s1/s2),
  A3 FF-AJ (+1.0·KL AutoJev on S, s1 only, budget permitting). S = 5,744 human-gold rows (`419fa355…`).
- Interpolation line θ(α) = α·S + (1 − α)·F1M, α ∈ {⅓, ½, ⅔}, development α rule vs F1.
- Finalists (priority): A2 artifact, A1 artifact, θ(α*) else A3-s1. Formal node B, 32K, image `dbe5f32b`, F1's scored
  cache `03b172f1…`. Successor rule vs F1 + "beats AutoJev" bar.

## Steps

| # | Step | Status |
| --- | --- | --- |
| 1 | Prereg `74b3bc3a0` | done |
| 2 | Code `d32bd7c88` (+ fixes `9c0876b0a`, `9732527cc`, `10f836f5c`, `2fe86bf3f`, `e46f553df`, `b99a87a22`, `909f8b748`) | done |
| 3 | Teacher files built twice, byte-identical; amendment 1 `e46f553df` | done |
| 3b | F1M = F1 merged in FP32 (`/data/dev2/runs/27b/m4b/F1M/checkpoint`, sha `3ed76a7b…`); merge check 496 projections, max rel 5.5e-8 | done |
| 4 | P0: parity PASS (gradient gate, amendment 2); probe 32K OOM → 16K + expandable segments PASS (174.8 GB, 1,721 tok/s); amendment 3 `909f8b748` (autocast, caps 7.5 / 8.3) | done |
| 5 | Chain `chain_m4b.sh` @ `9c7168698` (pid 2081589, `logs/chain.log`, `CHAIN.jsonl`): A1-s1 → A2-s1 → A1-s2 → A2-s2 → A3-s1 (budget guard; A3 expected skipped); ≈ 2.2 h per arm-seed. A1-s1 one-step PASS (0.610), reload PASS (0.026, Δp 0.0) | running (A1-s1 full) |
| 6 | Soups, F1M readout, α line, finalists | pending |
| 7 | CAL698 + formal ×≤3 + gates + mlx-diag | pending |
| 8 | Records, gist 06b, merge, report | pending |

## Launch pattern (chain rule)

Every script runs from a verified subtree mirror `/data/dev2/src/<sha>-src_training_decision2` (hash compared with the
local copy before launch); launch = `setsid nohup bash <script> … > /data/dev2/runs/27b/m4b/logs/<name>.log 2>&1 <
/dev/null &` in its own ssh call; liveness = PID (`kill -0`) + `docker ps` name `d2-27b-m4b-*` + first log line.

## GPU-hours (node B GPU0–2; receipts under `/data/dev2/runs/27b/m4b/**/receipts`)

| Item | GPU-h |
| --- | ---: |
| P0 parity attempt 1 (rendezvous hang, stopped) | 0.161 |
| P0 parity attempt 2 (runtime check) | 0.007 |
| P0 parity attempt 3 (Adam parameter gate) | 0.052 |
| P0 parity attempt 4 (PASS) | 0.025 |
| P0 probe 1 (32K, OOM) / probe 2 (16K, PASS) | 0.313 / 0.410 |
| A1-s1 one-step / reload | 0.610 / 0.026 |
| A1-s1 full (404 updates, 1.61 h wall; BEST = step 404, SELECT family-macro .914) | 4.828 |
| A2-s1 one-step / reload | 0.205 / 0.026 |
| **Total so far** | **6.66** |

## Research notes (read-only passes, 16:10–16:40)

- No multi-GPU trainer existed; `v2/dec/train_dec.py` has the token batching and the partial-teacher KL.
  `training.model.loss.per_example_loss`: total = CE + 0.5·Brier (gold, every row) + w·KL on teacher rows.
  The 27B kernel inference / CAL698 / `qwen-full` release paths accept full checkpoints unchanged.
- Model: 64 layers (48 gated-delta + 16 full attention), hidden 5,120, vocab 248,320; backbone 25,624,600,064 + head
  5,263,872 parameters (loaded 25,629,863,936 for an FF model).
- Human-gold rows in F1's file: 7,581 (A0s-strict 3,275; A6h 2,469; A7i 1,837 stays gold). No DEV2.0-9B targets exist.
- Node B paths: base `/data/decision20-20260926/models/Qwen3.8-27B`; F1 soup `/data/dev2/runs/27b/M3-A-soup/soup/checkpoint`;
  F1 scored cache `/data/dev2/runs/27b/M3-A-soup/formal/triton-cache` (`03b172f1…`); readout cache
  `/data/dev2/runs/27b/m3-warm-32768/triton-cache` (`583241fb…`); AutoJev comparator
  `/data/dev2/runs/27b/m2-peer-autojev27-nodeB-kernel`; teacher files `/data/dev2/private/27b/m4b-data/teachers-1/`.
