# ROCm runtime kernels: state record (paused 2026-10-02)

Research code only; nothing here touches the released packages or the release path.
Target: MI325X (gfx942, wave64), ROCm PyTorch, Transformers 5.17, FLA 0.5.2.
open-jev-fast was used as a design reference only; no code was copied.

Timing: GPU time per call, median of >=100 iterations, measured via HIP graph replay
(`timing.time_graph`, 20 calls per graph) unless marked "eager". GEMM weights rotate
past the 256 MB MALL.

## 1. Fused element-wise kernels (Triton, `triton_elementwise.py`)

Reference = the exact op sequence the runtime executes (`reference.py`).
27B backbone shapes (hidden 5120), M = tokens.

| Kernel | M512 ref -> Triton (us) | M2048 ref -> Triton (us) | 0.8B M512 ref -> Triton (us) |
|---|---|---|---|
| add + RMSNorm | 65.5 -> 15.1 (4.3x) | 126 -> 44 (2.9x) | 31.4 -> 5.8 (5.4x) |
| SiLU * up | 29.3 -> 15.2 (1.9x) | 88 -> 50.5 (1.7x) | |
| GDN prep (conv1d+SiLU, l2norm, gates) | 74.2 -> 19.1 (3.9x) | 132 -> 62 (2.1x) | 49 -> 13.8 (3.6x) |
| gated RMSNorm | 85.4 -> 11.2 (7.6x) | 221 -> 32 (6.9x) | |
| attention prep (head norm + RoPE) | 180 -> 29.9 (6.0x) | 330 -> 111 (3.0x) | |
| sigmoid output gate | 22.7 -> 7.1 (3.2x) | 48.5 -> 21 (2.3x) | |

Numbers are from the run before the final rsqrt / head-dim-128 exactness fixes; the
exact path costs more for attention prep at large M (non-exact 34 us vs exact 111 us
at M2048). Re-time with the final kernels on resume (`bench_elementwise.py`).

Alternatives:

- torch.compile: similar speed on simple kernels, but default mode changes rounding
  (SiLU*up 73% bit-match, attention prep 95%); with `emulate_precision_casts` it
  matches like the Triton kernels.
- AITER rmsnorm / silu_and_mul: 45% / 51% bit-match. Fails fidelity.
- HIP: not needed; Triton reaches bandwidth-bound timings.

### Bit-exactness findings

All six kernels are bit-identical to the reference at every tested shape. Required:

- Runtime numerics are bf16 autocast over fp32 parameters: residual, norms and gate
  parameters are fp32; each Linear input is cast to bf16 once.
- ATen ROCm row reductions were reproduced exactly: vec4 loads, 64 lanes with 4
  sequential accumulators combined as ((a0+a1)+a2)+a3, then a shfl_down tree
  (offsets 1..32); for row length 128, vec4 over 32 lanes with a 5-level tree
  (`aten_reduce.py`). Mean uses multiply by float32(1/H).
- `torch.rsqrt` is correctly rounded on this build; reproduced as fp64 rsqrt then
  round to fp32. This was the last mismatch source for gated RMSNorm and the head norm.
- causal_conv1d ROCm build uses no FMA, OCML exp and true division; the Triton conv
  uses mul+add and Triton is compiled with `enable_fp_fusion=False`.
- FLA chunk GDN with q/k at 16 heads (GVA) and pre-normalised q/k is bitwise equal to
  the repeat_interleave path with in-kernel l2norm.

### End to end (`integrate.py`, `capture.py`, public-231 prompts, released packages)

| Model | Requests bit-identical | Answer changes | Max drift | Eager median ref -> fused (ms) |
|---|---|---|---|---|
| 0.8B | 231 / 231 | 0 | 0 | 21.0 -> 13.4 (-36%) |
| 4B | 231 / 231 | 0 | 0 | 27.7 -> 18.1 (-35%) |

Most of the eager gain is fewer kernel launches (Python launch cost ~20-25 us per op).
The 27B capture was stopped by the pause.

## 2. Gated DeltaNet (FLA chunk)

FLA's default autotune space was widened with gfx942 options (waves_per_eu,
matrix_instr_nonkdim=16, more warps/stages; `bench_gdn.py --tune`), plus the
pre-normalised GVA path that drops the two l2norm kernels and the head repeat.
Output stays bitwise identical to the default configs.

| Shape (value heads, T) | default (us) | tuned + GVA (us) |
|---|---|---|
| 48, 512 | 121.5 | 81.3 |
| 48, 1024 | 177.6 | 117.8 |
| 48, 4096 | 649 | 437 |
| 16, 512 | 101.7 | 65.1 |

27B T512 breakdown (default): fwd_h 29.7, kkt+solve 27.6, fwd_o 22.4, w_u 22.1,
l2norm 20.7 us. Eager FLA is CPU-bound (~275-300 us per call), so graphs or
fewer launches matter more than kernel speed at small T. FLA autotune keys omit T;
pin configs per shape. A fused GDN kernel was not built.

## 3. Tree attention (`tree_attention.py`)

Triton kernel with a packed bitmask, block skipping and the output gate fused.
SDPA flash / mem-efficient reject explicit masks on this ROCm build, so masked
SDPA falls back to the math kernel.

| 27B case | SDPA (us) | Triton (us) |
|---|---|---|
| N384 tree mask (vs math) | 189 | 38.4 |
| N384 causal (vs flash) | 70.6 | 38.4 |
| N1024 tree mask (vs math) | 713 | 115 |
| N1024 causal (vs flash) | 165 | 146 |

Fidelity: 65% bit-match, rel L2 2.1e-3 vs math (88.7%, 1.1e-3 with fp32 PV); SDPA flash
vs math is itself 65% / 2.2e-3. Not bit-exact by construction.

## 4. GEMMs (27B, `bench_gemm.py`, `hipblaslt_search.cpp`, `triton_gemm.py`)

torch `F.linear` vs the best of ~1360 supported hipBLASLt algorithms per shape.

| GEMM | M | torch (us) | best hipBLASLt (us) |
|---|---|---|---|
| gate/up (N17408) | 64 | 67.5 | 48.3 |
| gate/up | 128 | 74.2 | 62.7 |
| gate/up | 512 | 170 | 167 |
| gate/up | 1024 | 321 | 276 |
| down | 512 | 208 | 183 |
| q_proj | 512 | 140 | 122 |

Merged weights (M512): GDN in_proj 242.7 separate -> 160.2 merged; gate+up
2x170 -> 305; q/k/v 188 -> 139. Triton GEMM is slower than hipBLASLt (2-3x at
K=17408). Tiny-N projections are launch-bound in eager (~17 us vs ~7 us GPU).
Not measured: split-K (fixed to `GemmTuning::setSplitK`, not yet run),
small-model GEMMs, rocBLAS/CK backends.

## 5. Go / no-go so far

- GO: Triton element-wise kernels (bit-exact, 0 answer changes, ~-35% eager latency).
- GO: FLA gfx942 retune + pre-normalised GVA path (bitwise identical, 0.64-0.84x).
- GO: hipBLASLt per-shape algorithm selection and merged projection GEMMs.
- CONDITIONAL: tree attention (4-6x vs masked SDPA, not bit-exact; needs a parity gate).
- NO-GO: AITER element-wise (fidelity), Triton GEMM (speed), HIP/MFMA rewrites (no gap
  that justifies them yet).
- UNTESTED: fused GDN kernel, split-K.

GPU use so far: about 0.5 GPU-h of the 10 GPU-h budget.

## 6. Resume

1. Lease one GPU with `track=rocm-kernels` on a node not doing release work; run via
   `run.sh` (network-less container, renderD isolation).
2. Re-time element-wise kernels with the final exact kernels (`bench_elementwise.py`).
3. 27B end-to-end capture (`capture.py`, modes fused/shadow).
4. Split-K and small-model GEMM sweeps (`bench_gemm.py`).
5. Build the per-size projection table, then decide on the fused GDN kernel.
