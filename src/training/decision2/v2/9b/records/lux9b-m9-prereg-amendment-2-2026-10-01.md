# 9B Milestone 9, amendment 2: stage 2 (IB1-r3 + IB2 on the L9 recipe) starts as node-C GPUs free up

Written 2026-10-01 ≈16:00 UTC+8, before any stage-2 GPU job and before any stage-1 arm readout (the four stage-1
seeds are mid-training; only the references C0, Lux 1.0 and B0 have been read). It replaces the preregistration's
"Stage 2" section (`8570a5896`); everything else is unchanged.

## Why

- COORDINATION 15:55: **IB1-r3** (`decision-2.0-training-data@31b200a3`, `m6/ib1/`) and **IB2** (`@c5dbdd0a`,
  `m6/ib2/`) are both release-safe, and 27B M6 / 9B M9 start their IB stages themselves. The 4B decoder track runs
  one combined stage on its from-base LoRA recipe (M11: LH + IB1 + IB2, plus a transfer-only ablation without the
  in-distribution families).
- Waiting for the stage-1 verdict (≈18:30) before training would leave four node-C GPUs idle and push stage 2 past
  midnight. The 4B evidence already fixes the recipe a priori (the from-base LoRA, L9's analogue), so stage 2 trains
  on the L9 recipe without waiting. This is not result-driven: no stage-1 arm has been read.

## Stage 2 arms (two seeds each, merged FP32 soups as in stage 1)

| Arm | TRAIN | Rows |
| --- | --- | ---: |
| **L9IB** | x60 + IB1-r3 TRAIN (`1e1b08f3…`, 24,325 rows, 4.75M tokens) + IB2 TRAIN (`ee137efa…`, 24,518 rows, 5.42M tokens) | 171,494 |
| **L9IBX** (transfer-only ablation) | L9IB without the in-distribution families `isarc`, `w2c` (IB1) and `hover`, `gsm2` (IB2) | 160,543 |

- Recipe = L9's (LoRA r 128 / α 256 / dropout .05 / LR 1e-4 from Qwen3.5-9B-Base `68c46c4b…`, fresh candidate head with
  the shared init seed 20261001, head LR 1e-4, own-Lux KL 1.0, CE + 0.5·Brier, token batching ≤ 32,768 / ≤ 64 rows, 64
  rows per update, one epoch, `even8` + SELECT700 `matrix-v1`, max length 8,192), seeds 20260926 / 20260927. **IB rows
  train on gold only** (no teacher target; `--teacher-partial`); x60 rows keep their own-Lux targets.
- Builds: `lux9b/m9_data.py` (x60 lines byte for byte, then the kept IB1 and IB2 lines; teacher = x60's file byte for
  byte; refuses any IB row whose id, lineage group or canonical input repeats one already in the build) through
  `m9/stage2.sh build` on node C, after the IB files (host HF CLI, pinned revisions) match the release records'
  TRAIN / DEV hashes. `data/READY2.json` pins both builds; the chains re-hash against it before each seed.
- **Placement (node C, `m9/chains2.sh`):** GPU5 L9IB-s1 at once (the node's M9 cache is warm from stage 1); GPU2
  L9IB-s2 and GPU4 L9IBX-s1 when their stage-1 chains end; GPU1 L9IBX-s2 after node C's L9 merges. Seed cap 5.0 GPU-h;
  the 100 GPU-h start gate holds. Merges / soups `m9/post-c.sh` (L9IB on GPU5, L9IBX on GPU4).

## Readouts, gates and finalists

- Node A, the stage-1 path (`m9/post-a.sh` with `M9_STAGE=2`): the same eight panels, the same seven development gates
  against C0 (`m9_rules.py`, output `select/9b-finalists-s2.json`), plus report-only contrasts L9IB − L9, L9IBX − L9,
  L9IB − L9IBX and the **IB1 / IB2 DEV per-family accuracy** (`v2.dec.eval_rows`; C0, L9 and both soups; diagnostic,
  never a gate).
- **Formal:** at most two stage-2 passers (an HT-DEV v2 GAIN first, then the larger typed-DEV T), on the stage-1
  formal path, after the newest COORDINATION notes are re-read and a lock record is pushed. At most four formal
  candidates in M9 overall. The successor choice among every passer of items 1–7 keeps the preregistered order
  (highest v3 `ci95.low` vs the released T = 1 run; within 0.25, an HT-DEV v2 GAIN, then the higher formal ΔH).
- **Item 6:** before a stage-2 formal run, overlap exposure receipts (`v2.eval.overlap_effects exposure`, the K-a13
  payload `2194716a…` on node A) for the IB1 and IB2 TRAIN files must list 0 groups.
- **C1:** any IB-trained candidate needs the custodian's C1 content recheck before item 8.
- If stage 1 leaves L9L as the only passer and L9 fails, a later amendment may add L9L-based IB arms; nothing else
  changes.

## Budget

Stage 2 ≈ 4 seeds × ≈ 3.6 GPU-h (≈ 70M tokens each) + merges / readouts ≈ 2 → ≈ 16 GPU-h, inside the preregistered
stage-2 allowance (IB1 40 + IB2 30).
