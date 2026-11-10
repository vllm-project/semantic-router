# NPU: the fused rotate-half op as an exact kernel (Ascend 950DT)

## Findings

On Ascend 950DT (CANN 9.1, torch_npu 2.10.0.post4), measured against the eager
reference kernels:

- **`npu_rotary_mul(rotary_mode="half")` is bit-exact as the `rotary_half`
  slot when routed through FP32.** The reference computes rotate-half in FP32
  and casts back (`kernels.rotary_half_ref`); the op's half mode computes the
  same algebra (`r1 * input + r2 * cat((-x2, x1))`). The op's own BF16 path
  rounds one ulp off the reference (max 0.0625); feeding the op FP32 inputs
  and casting back reproduces the reference **bit for bit at fp32 and bf16**,
  the dtypes tried. Registered as an exact kernel, so no gate applies and no
  answer changes by construction.
- **`npu_fusion_attention` equals eager SDPA bit for bit without a mask** —
  the same kernel underneath (torch's SDPA already dispatches to it on this
  device), so registering it buys nothing. With a mask it does not reproduce
  the additive-mask reference: the op takes bool masks only (1 = masked), and
  the float→bool conversion path measures fp32 max 6.0e-7 / bf16 max 2.0e-3.
  Not registered.
- **`npu_rms_norm` and `npu_swiglu` are not bit-exact** (fp32 max 1.9e-6 /
  9.5e-7, bf16 0.0625 — reduction-order and rounding differences). Not
  registered: nothing consumes those slots on NPU yet (the fast path's
  `fused_applies` gates on `is_cuda`), and no family consents to inexact
  kernels. The measurements stay in the probe data for the fast-path work.
- **Served end to end, the fused kernel does not move latency.** A/B on one
  950DT card, fused vs reference-only, same script and window:

  | concurrency | fused (rps) | reference (rps) |
  | --- | ---: | ---: |
  | 4 | 507.1 | 498.7 |
  | 8 | 1355.9 | 1360.2 |
  | 16 | 1241.5 | 1231.3 |

  Isolated, the op is 4.3x faster at fp32 (105 → 24 us per (q, k) call) and
  2.6x at bf16 (157 → 60 us); the encoder's batch forward is dominated by
  everything else, and a per-op gain this small does not surface. The value
  of the change is the registration channel and the exactness verdicts, not
  latency.

## The bit-exact evidence

- Per-op, per shape ([1,12,195,64], [2,16,473,64], [4,16,665,64]) and dtype
  (fp32, bf16): `torch.equal` against `rotary_half_ref` — all exact on the
  FP32-routed wrapper, and the wrapper matches the reference on CPU tensors
  too (unit-tested with a faked torch_npu).
- End to end: a served Vela 1.0 encoder (`/v1/classify`, the 30-prompt office
  corpus) produced **bitwise identical answers** on the fused and reference
  paths — 1,140 numeric leaves compared across the 30 requests, zero
  differences; readiness passed its golden check on both arms.

## Setup

- Host: 2-socket Kunpeng 950 7590, 8x Ascend 950DT (driver 25.6.rc1,
  CANN 9.1.0), torch 2.10.0+cpu / torch_npu 2.10.0.post4, runtime main
  `7313c646c` with this change; card 3 shared with a co-tenant container.
- Probe: synthetic V1.0/V2.0-shaped inputs, 200 timed reps after warmup for
  the isolated per-op numbers; serve A/B ran the frozen corpus (195 input
  tokens p50) at c4/8/16, 120 requests per point.
- `npu_rotary_mul` constraint check: head_dim 64 < 896, B*N < 1000, BNSD
  layout as the slot passes it; cos/sin as [1, 1, S, D] (11SD). The wrapper
  enforces the documented limits and falls back to `rotary_half_ref` beyond
  them, so the exact contract holds wherever the shapes land.
- Exactness was measured on this SoC only: the accelerator gates the exact
  registration on the SoC parsed from `torch.npu.get_device_properties().name`
  (`Ascend950DT_9572` → `Ascend950DT`, the rocm.py precedent of registering
  exact only on the measured gfx942). Other Ascend SoCs — the 910B1 included —
  run the reference kernels until measured there.
