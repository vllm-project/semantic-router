# ~27B M4b state (resume file)

Updated: 2026-09-29 16:55 UTC+8 (M4b worker; **prereg written, no GPU job yet**)
Branch: `xunzhuo/decision-2-training-27b-m4b` (worktree `/home/xunliu/code/vllm-sr-dev2-27b-m4b`; merge-only into
`xunzhuo/decision-2-training`). Gist file: `06b-decision-2-27b-m4b.md`. GPUs: node B GPU0–2 only (lent by 9B;
eval may hold short shared leases for peer mlx-diag: wait for release). Budget 36 GPU-h.
Milestone 4 (other worker; node A GPU2–4 + node B GPU5–7; `m4-*` files/dirs) is not touched.

## Plan (prereg `m4b-prereg-2026-09-29.md`)

- A1 FF-gold (full-parameter FSDP ×3 on F1's exact file `de00df03…`, s1/s2), A2 FF-Lux (+1.0·KL own-Lux on the
  5,744 human rows, s1/s2), A3 FF-AJ (+1.0·KL AutoJev on the same rows, s1 only, budget permitting).
- Interpolation line θ(α) = α·S + (1 − α)·F1M, α ∈ {⅓, ½, ⅔}, development α rule vs F1.
- Finalists (priority): A2 artifact, A1 artifact, θ(α*) else A3-s1. Formal node B, 32K, image `dbe5f32b`, F1's scored
  cache `03b172f1…`. Successor rule vs F1 + "beats AutoJev" bar.

## Steps

| # | Step | Status |
| --- | --- | --- |
| 1 | Prereg committed + pushed | pending |
| 2 | Code (`v2/27b/m4b/`: trainer, launcher, teacher builder, soup/interp, drivers, tests) | pending |
| 3 | Mirror to node B; teacher files built twice (CPU); amendment 1 (freeze) | pending |
| 4 | P0 (tiny parity, 27B probe); amendment 2 (precision choice, caps) | pending |
| 5 | A1-s1, A2-s1, A1-s2, A2-s2, A3-s1 (+ readouts) | pending |
| 6 | Soups, F1M, α line, finalists | pending |
| 7 | CAL698 + formal ×≤3 + gates + mlx-diag | pending |
| 8 | Records, gist 06b, merge, report | pending |

## GPU-hours

| Item | GPU-h |
| --- | ---: |
| (none yet) | 0 |

## Research notes (read-only passes, 16:10–16:40)

- No multi-GPU trainer exists (all shared trainers are single-device); `v2/dec/train_dec.py` has the token batching and
  `--teacher --teacher-kl-weight --teacher-partial` (teacher.jsonl `{id, input_sha256, teacher_probs}` joined by id).
  The 27B kernel inference / CAL698 / `qwen-full` release paths accept full checkpoints unchanged
  (`run_formal_kernel.sh` needs `LOADED_PARAMETERS=25629863936`). Existing 27B drivers allow only node B GPU5–7:
  M4b uses its own wrappers under `v2/27b/m4b/`.
- Model: 64 layers (48 gated-delta + 16 full attention), hidden 5,120, intermediate 17,408, vocab 248,320, untied;
  backbone 25,624,600,064 + head 5,263,872 parameters. FSDP estimate ≈ 200–215 GB/GPU at 32K tokens/rank with
  autocast's weight cache off.
- Human-gold rows in F1's file: 7,581 (A0s-strict 3,275; A6h 2,469; A7i 1,837). Own-Lux covers all; AutoJev covers
  5,744 (not A7i); no DEV2.0-9B targets exist. Own-Lux agreement with gold: GoEmotions Choice .698 / Noul .824, SNLI
  .989; AutoJev .754 / .828 / .846; A6h .38–.67 both.
- Node B paths: base snapshot `/data/dev2/hf-cache/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2…` (M3 mounted
  `/data/decision20-20260926/models/Qwen3.8-27B`); F1 soup `/data/dev2/runs/27b/M3-A-soup/soup/checkpoint`; F1 scored
  cache `/data/dev2/runs/27b/M3-A-soup/formal/triton-cache` (`03b172f1…`); readout cache
  `/data/dev2/runs/27b/m3-warm-32768/triton-cache` (`583241fb…`); SELECT700
  `/data/decision20-20260926/data/rights_clean_goemotions_v2/select.jsonl`; M3-A-s1 training cache
  `/data/dev2/runs/27b/M3-A-s1/…/triton-cache`; AutoJev comparator `/data/dev2/runs/27b/m2-peer-autojev27-nodeB-kernel`.
  AJ-SL (177 A6h rows) is on node A only (`/data/dev2/runs/data/m3a2/upload-aj-sl/`) and HF `@780d2743`.
