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
under the 3.9e-3 BF16 GDN rounding floor the 950DT round reported
(arXiv:2609.04098) — and on real in-model activations the per-call max
sits at that floor exactly (below); consistent with both implementations
being bounded rounding paths of the same math.

## Serving impact

Serving `vllm-sr/vela3-mini` (BF16, d3-mini base) with the slot active
against the exact-kernel serve, back-to-back on the same host:

| metric | reference kernels | fla-npu Ascend C |
| --- | --- | --- |
| sequential p50 | 1,635 ms | **272 ms (6×)** |
| 16-way throughput | 0.6 rps | **12.1 rps (20×)** |
| calibrated `max_num_batched_tokens` | 256 | **4,096** (0.3 s cap 5,376) |
| mean rows per forward (`/metrics` batch_rows) | 1.0 | **4.6** |
| label diff (60-leaf corpus, 8-way × 3 rounds) | — | **180/180 agreement, no flips**, max \|dp\| 0.0186 |

The batching recovery is the cascade the reference path's cost blocked: at
2.5 ms/token the calibration's 0.3 s forward cap shrank the token budget
below one question row (~180-360 tokens), so every forward carried exactly
one row (`/metrics`: batch_rows sum == count over a probe round); with the
chunk at ~0.06 ms/token the same calibration lands at 4,096 tokens and the
scheduler's merged batches carry 4.6 rows on average.

The performance arms registered the slot through a measurement patch ahead
of this change (the consent lands in the same PR); the label-diff arms in
this round ran the branch itself — measured SoC + the family's consent + the
explicit `FLA_NPU_APPROXIMATE` opt-in, no patch — against the consent-off
serve, back-to-back on the same card.

## Numerics

The 950DT round's numerics probes re-run on this SoC (a kernel-counter
wrapper computing both implementations at every GDN-layer call and forwarding
the reference output; 60 requests × 24 GDN layers = 1,440 interleaved calls):

- Per-call divergence from the reference stays at or under 9.8e-4 on the
  microbenchmark shapes; on real in-model activations the per-call max sits
  at 3.9e-3 (2^-8) — the reported rounding floor of BF16 GDN kernels
  (arXiv:2609.04098), exactly where the 950DT probes landed.
- Per-layer injection is sparse and bounded: mean |diff| 8.4e-7 per call
  (worst call 3.8e-6), 47% of elements bit-identical on average (min 37%);
  every per-call max sits on the bf16 ulp grid (2^-14..2^-8).
- No compounding: the injection curve does not grow across the 24 GDN
  layers (growth 0.50×, last over first; the 950DT round measured 0.96×),
  and single-call divergence does not grow with sequence length (T=140..345,
  the lengths the served corpus reaches, all capped at 2^-8).
- FP64 audit at the served shape, against the correctly-rounded bf16 of
  the FP64 answer: the reference kernel sits within half an ulp for 99.98%
  of elements, while the fla-npu chunk — a bf16 chunkwise accumulation in
  a different summation order — does not reproduce the 950DT round's
  "max 1 ulp from the FP64 reference" under this audit (61.8% of elements
  beyond half an ulp; the max ulp ratios sit on near-zero outputs where
  the ulp yardstick degenerates). The registration rests on the
  reference-relative bounds above and the serve-level label diff, not on
  a correctly-rounded verdict.
- Serve-level (the branch's own activation chain — consent + `MEASURED` +
  the opt-in, no patch): 180/180 leaf answers agree (60 leaves × 3 rounds),
  no flips, max option-probability move 0.0186 — under the npu readiness
  gate's `GPU_TOLERANCE` (0.02). The 950DT round flipped 4 of 60 (all on
  two-option gaps ≤ 0.02); this round's corpus covers the `domain` and
  `attack` router questions.

## Registration

- `npu_fla.MEASURED` gains `"Ascend910B1"` (the fla-npu a2 wheel's build);
  bf16-only, and the tree, branch and packed layouts keep the per-segment
  padded path (the bound kernel takes no `cu_seqlens`), as on the 950DT.
- The opt-in (`FLA_NPU_APPROXIMATE`) stays default-off everywhere.
- The vela3_d3 family consents for its own packages: a vela3_d3 package is
  validated as its own model (its own registry pin, tests and surfaces), so
  its spec carries `approximate_kernels=True` scoped to the npu
  (`DtypePolicy.approximate_accelerators`) where the decision3 base's dtype
  consented only quantized packages -- the CUDA and ROCm approximate kernels
  (`cuda-fused-approximate.md`) keep the base's consent, which is off; no GPU
  record covers this family. The activation chain is measured
  SoC + package consent + the explicit opt-in; without the opt-in, and for
  the released decision3/d3 BF16 packages (whose 950DT verdict stands), the
  serving path is unchanged.
- Readiness stays guarded: the npu gate is tolerance-based
  (`GPU_TOLERANCE` 0.02, category agreement) and this round's serve-level
  max |dp| is 0.0186. The vela3_d3 golden check with the slot active
  passes (8 of 8 golden answers match the recorded npu references,
  `reference: npu`); this round records vela3-mini's npu references (the
  generic `npu` key, torch 2.10.0 — the hardware-qualified `npu:Ascend910B1`
  entry answers as its baseline and is not kept).
- Unit tests (`tests/test_npu_fla.py`) cover the second SoC in the MEASURED
  fixture and the vela3_d3 consent; no other code paths move.

## Reproduction

Measurement scripts (microbenchmark, numerics probes, serve rounds, answer
capture and diff) live with the fla-npu measurement notes; the round ran on
one 910B1 of a shared 8-card host under the canhazgpu scheduler
(`npu run --gpus 1`). The microbenchmark arms call the two slot
implementations directly at the served shape; the numerics probes wrap the
engine's kernel selection so every GDN-layer call computes both
implementations (the reference output forwarded, per-call divergence
recorded per layer and length), plus an FP64 audit of both implementations
against the correctly-rounded bf16 of the FP64 answer. The performance arms
served `vllm-sr/vela3-mini` at registry pin `bbc0206e` (d3-mini base
`566da0ea`) on main `83f8242e7` with the slot registered through the
measurement patch; the label-diff arms serve the same package on this
branch with and without `FLA_NPU_APPROXIMATE`, same card allocation, and
diff the captured answers with `docs/records/npu-parity-ascend910b1.md`'s
tolerance framing. The npu golden references are recorded with
`tools/golden_answers.py --record vllm_srun/registry/golden_answers_vela3_d3.json`
on the consent-off serve, and the slot-active serve's readiness check
verifies against them.
