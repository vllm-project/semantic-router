# The fla-npu Ascend C chunk on Ascend 910B1

Issue: #6236 (accepted, Router Models & Inference Runtime). Precedents:
`npu-fla-chunk-ascend950dt.md` (the slot's first SoC round and the consent
verdict), `npu-parity-ascend910b1.md` (the device class).

## What was measured

The `chunk_gated_delta_rule` slot of the d3 decision models on Ascend 910B1
(Huawei Cloud EulerOS 2.0 aarch64, CANN 8.5.1, torch 2.10.0,
torch_npu 2.10.0.post4, fla-npu a2 wheel 26.9.1), at the served GDN shape
(v_heads=32, k_heads=16, head_k=head_v=128, chunk 64, bf16,
`use_qk_l2norm_in_kernel=True`, B=1):

| T | reference (`kernels.chunk_gated_delta_rule_ref`) | fla-npu Ascend C (`flash_gated_delta_rule`) | speedup | max \|fla − ref\| |
| --- | --- | --- | --- | --- |
| 256 | 28.87 ms | 0.921 ms | 31× | 4.9e-4 |
| 355 | 42.98 ms | 0.948 ms | 45× | 9.8e-4 |
| 512 | 56.75 ms | 0.884 ms | 64× | 7.3e-4 |
| 1343 | 147.58 ms | 0.903 ms | **163×** | 4.9e-4 |

The kernel's time is flat in T (one Ascend C launch regardless of length), so
the win grows with row length; the 950DT round measured the same slot at
107.1 ms → 0.42 ms there. The 910B1's reference path is ~1.4× slower per
token than the 950DT's. Every shape's divergence sits at or under 9.8e-4 —
an order under the 3.9e-3 BF16 GDN rounding floor the 950DT round reported
(arXiv:2609.04098) — consistent with both implementations being rounding
paths of the same math.

## Serving impact

Serving `vllm-sr/vela3-mini` (BF16, d3-mini base) with the slot active
against the exact-kernel serve, back-to-back on the same host:

| metric | reference kernels | fla-npu Ascend C |
| --- | --- | --- |
| sequential p50 | 1,635 ms | **272 ms (6×)** |
| 16-way throughput | 0.6 rps | **12.1 rps (20×)** |
| calibrated `max_num_batched_tokens` | 256 | **4,096** (0.3 s cap 5,376) |
| mean rows per forward (`/metrics` batch_rows) | 1.0 | **4.6** |
| label diff (60-leaf corpus, 8-way × 3 rounds) | — | **60/60 agreement, no flips**, max \|dp\| 0.0111 |

The batching recovery is the cascade the reference path's cost blocked: at
2.5 ms/token the calibration's 0.3 s forward cap shrank the token budget
below one question row (~180-360 tokens), so every forward carried exactly
one row (`/metrics`: batch_rows sum == count over a probe round); with the
chunk at ~0.06 ms/token the same calibration lands at 4,096 tokens and the
scheduler's merged batches carry 4.6 rows on average.

The serve-level round registered the slot through a measurement patch ahead
of this change (the consent lands in the same PR); the reference arm ran the
consent-off serve unchanged.

## Numerics

- Per-call divergence from the reference stays at or under 9.8e-4 across the
  served shapes — an order under the 950DT round's 3.9e-3 floor.
- Serve-level: 60/60 leaf agreement, no flips, max option-probability move
  0.0111 — under the npu readiness gate's `GPU_TOLERANCE` (0.02). The 950DT
  round flipped 4 of 60 (all on two-option gaps ≤ 0.02).
- Narrower than the 950DT evidence on purpose-noted grounds: this round's
  corpus covers the `domain` and `attack` router questions (60 leaves), and
  the 950DT round's per-layer injection and no-compounding-across-layers
  probes were not re-run on this SoC. The registration below leans on the
  opt-in, not on claiming the 950DT consent verdict transfers.

## Registration

- `npu_fla.MEASURED` gains `"Ascend910B1"` (the fla-npu a2 wheel's build);
  bf16-only, and the tree, branch and packed layouts keep the per-segment
  padded path (the bound kernel takes no `cu_seqlens`), as on the 950DT.
- The opt-in (`FLA_NPU_APPROXIMATE`) stays default-off everywhere.
- The vela3_d3 family consents for its own packages: a vela3_d3 package is
  validated as its own model (its own registry pin, tests and surfaces), so
  its spec carries `approximate_kernels=True` where the decision3 base's
  dtype consented only quantized packages. The activation chain is measured
  SoC + package consent + the explicit opt-in; without the opt-in, and for
  the released decision3/d3 BF16 packages (whose 950DT verdict stands), the
  serving path is unchanged.
- Readiness stays guarded: the npu gate is tolerance-based
  (`GPU_TOLERANCE` 0.02, category agreement) and this round's serve-level
  max |dp| is 0.0111. The vela3_d3 golden check with the slot active passes
  within the bound; npu references for vela3_d3 are collected where missing.
- Unit tests (`tests/test_npu_fla.py`) cover the second SoC in the MEASURED
  fixture and the vela3_d3 consent; no other code paths move.

## Reproduction

Measurement scripts (microbenchmark, serve rounds, answer capture and diff)
live with the fla-npu measurement notes; the round ran on one 910B1 of a
shared 8-card host under the canhazgpu scheduler (`npu run --gpus 1`). The
microbenchmark arms call the two slot implementations directly at the served
shape; the serving arms serve `vllm-sr/vela3-mini` at registry pin `bbc0206e`
(d3-mini base `566da0ea`) on main `83f8242e7` with and without the slot
registered, same card allocation, and diff the captured answers with
`docs/records/npu-parity-ascend910b1.md`'s tolerance framing.
