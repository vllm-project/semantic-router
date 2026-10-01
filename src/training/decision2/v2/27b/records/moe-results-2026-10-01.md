# 27B MoE milestone (MoE-1): results — INTERIM (Stage A running)

Preregistration `moe-prereg-2026-10-01.md` (signed `bc82b771e`; amendments 1–3). Gate `moe-gate-2026-10-01.md`.
Development readouts are never release scores; post-key numbers are "post-key same-panel". Nothing was uploaded to
Hugging Face. Sections marked **pending** are filled from the screen chain and Stage B.

## Summary so far

- **Gate:** Gemma-4-26B-A4B(-it) and Qwen3.5-35B-A3B(-Base) are Apache-2.0 (Gemma 4's licence link is the verbatim
  Apache License 2.0), ungated, pinned and hash-verified on both nodes; the pinned ROCm image supports both; LoRA fits
  one MI325X.
- **Experts kernel:** eager experts are infeasible for training (≈ 49 s per update Gemma, ≈ 110 s Qwen-MoE);
  `grouped_mm` is as close to an FP32 reference as eager (hidden-state error ratio 1.05 / 0.85 at the median) and is
  pinned for training and inference (amendment 1, which replaced a mis-specified random-head parity rule before any
  training).
- **Cost:** at micro-batch 1 the MoE bases are no cheaper to train than the dense 27B: Gemma 9.9 s per update
  (A20r ≈ 10), Qwen-MoE 15.4–16.9 s. Per-token active compute (≈ 3.8B / 2.9B + head) does not translate into
  training speed on this path.
- **Trainer provenance finding (27B track, no result changes):** the 27B launcher's `python -m` from `/code` shadows
  the vendored BEST368 tar, so A20r and the other 27B arms trained with the repository trainer (their provenance
  hashes equal the mirror's files). The MoE cells use the same byte-frozen trainer through a wrapper (amendment 2).

## Stage P0 (node A GPU3 / GPU5)

| | Gemma-4-26B-A4B-it | Qwen3.5-35B-A3B |
| --- | --- | --- |
| Text parameters (loaded text decoder) | 25,233,141,760 | 34,152,051,328 |
| Active per token (text decoder; + LM head for Qwen: 508.6M) | 3,822,530,560 | 2,946,429,568 |
| LoRA targets / adapter / head (rank 32) | 205 / 37,171,200 / 2,895,360 | 310 / ≈ 42.3M / 2,105,856 |
| Training s per row, eager → grouped_mm | 3.05 → 0.66 | 6.91 → 1.07 |
| FP32-reference hidden-state error, eager / grouped_mm (median, max) | .185 / .193, .338 / .359 | .234 / .198, .349 / .419 |
| Peak allocated (training / T = 1 inference at 32K) | ≈ 99 GiB / ≈ 101 GB | ≈ 131 GiB / pending |

## Stage A cells (seed 1 = 20260926; `a20`; A20r contract)

| Cell | Node / GPU | Admission | One-step (r2) | Reload | s per update | Checkpoint 892 | Screen readout |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| MOE-Git-s1 (Gemma-4-26B-A4B-it) | A / 3 | 56,969 rows, 26,128,868 tokens, max 4,708 (limit 4,736) | pass, 0.078 GPU-h | 0 / 32, 6e-8 | 9.9 | pending (≈ 19:25Z) | pending |
| MOE-Qit-s1 (Qwen3.5-35B-A3B) | A / 5 | 56,969 rows, 25,043,392 tokens (= A20r) | pass, 0.139 | 0 / 32, 6e-8 | 15.4 | pending | pending |
| MOE-Qpt-s1 (Qwen3.5-35B-A3B-Base) | B / 6 | same as Q-it | pass, 0.146 | 0 / 32, 6e-8 | 16.9 | pending | pending |

- The first one-step of each cell failed on an argparse flag (≈ 0.0004 GPU-h each, no model code ran; amendment 2).
- **Dense matched reference** (M4-A20r-s1 checkpoint 892, node B, T = 1, 32K, kernel path, cache `583241fb`):
  P_dev **78.22** (T_dev .9244, H_pilot .6618), H_dev2 **.5634** (vs the A20r soup: −.002 [−.015, +.011], TIE).
  Readout `/data/dev2/runs/27b-moe/readouts/A20r-s1-c892`.

## Screen decision — pending (`/data/dev2/runs/27b-moe/screen/SCREEN.json`, node B)

## Stage B, formal runs, successor items 1–7, "beats AutoJev" — pending

## Budget (GPU-h; cap 60)

- At 17:20Z ≈ 1.3: probes 0.150, kernel checks 0.049, failed launches 0.001, preflights 0.49, dense reference
  readout ≈ 0.4, MoE path smokes ≈ 0.15, plus the three cells' running full attempts.
- Projection (amendment 1 plan with measured speeds): the screen decision lands ≈ 21:45Z with the three cells at
  ≈ 4.8 h each (≈ 14.4); a Gemma winner then needs ≈ 5 + 10 (seed 2) + ≈ 4 evaluation → ≈ 36 total; a Qwen winner
  ≈ 10 + 15.5 + 4 → ≈ 46 total.
