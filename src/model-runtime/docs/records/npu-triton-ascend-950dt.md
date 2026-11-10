# NPU: the triton-ascend probe on Ascend 950DT

The fused decoder layers behind `engines/native/fast.py` are what make the
`triton_gfx942` kernels worth registering: on CUDA the approximate path yields
14-48% end to end on the decision families that consent to
`DtypePolicy.approximate_kernels`. None of that is reachable on NPU today -
`fast.py` gates on `is_cuda`, and no NPU kernels exist beyond the rotate-half
slot (#5188). Before scoping that work, this record answers the tooling
question: can triton-ascend compile, run, and reproduce eager numerics on the
950DT at all.

## Findings

Probed on an Ascend 950DT with the `triton` package this stack already ships
(3.2.0 - the triton-ascend backend, `triton.backends` listing only `ascend`),
against eager torch ops:

| op | dtype | bit-exact | triton us | eager us |
| --- | --- | --- | ---: | ---: |
| add (smoke) | fp32 | yes | 34.1 | 14.8 |
| silu_mul | fp32 | yes | 33.0 | 24.6 |
| silu_mul | bf16 | yes | 32.3 | 41.3 |
| rmsnorm | fp32 | no (1.9e-6) | 71.7 | 58.0 |
| rmsnorm | bf16 | no (0.0156) | 72.4 | 76.1 |
| rotary_half | fp32 | yes | 136.0 | 88.7 |
| rotary_half | bf16 | yes | 131.9 | 161.2 |

- **The tooling works end to end.** `@triton.jit` kernels compile and run on
  the 950DT, pass device tensors from torch_npu, and reproduce the eager ops
  **bit for bit** for element-wise math (add, silu_mul) and rotate-half at
  fp32 and bf16. A reduction op (RMSNorm) differs only in reduction order
  (fp32 max 1.9e-6, bf16 0.0156) - the same class of difference the
  `npu_rms_norm` probe measured in the fused-op record - and stays inside
  the tolerance the GPU parity records use.
- **First-cut kernels carry no launch-time win.** A triton launch costs
  ~33 us here against 15-25 us for eager torch ops, so small standalone
  kernels do not pay for themselves. A fused layer's value on this device
  has to come from fewer memory round trips across a whole block, not from
  per-op launch wins - and the A/B has to be served end to end, since the
  950DT's execution hides dispatch costs (the graph-capture record's
  finding).
- **The upstream probe path works as-is.** The package named `triton` in
  this environment is triton-ascend, so an accelerator's `_optional`
  style detection sees it without any NPU-specific indirection.

## Scoping the follow-up

The fused-layer path the probe enables decomposes as:

- **All seven slots or none.** `fast.py`'s `fused_unavailable` requires every
  `FUSED_SLOTS` entry (`add_rmsnorm`, `residual_add`, `silu_mul`, `attn_prep`,
  `gdn_prep`, `gated_rmsnorm`, `sigmoid_gate`) to have a kernel before the
  fused layers install; registering only the bit-exact ones buys nothing.
  The RMSNorm-class slots land as approximate kernels, the prep slots
  (`attn_prep`, `gdn_prep`) need torch_npu equivalents or triton-ascend
  implementations of their own.
- **`fast.py` gates need an NPU-aware form**: `fused_applies`' `is_cuda`
  check, the `cuda`-specific autocast probes, and the stream-capture guard
  in `run_fused`.
- **The consumers are the decision families, not vela.** The fused layers
  serve `qwen3` / `qwen3_5_text` backbones - decision 2.0 (Kai/Eos/Sol) and
  decision 3.0 (d3). Both families already consent to
  `DtypePolicy.approximate_kernels`; vela does not, and does not consume
  these layers either. The precondition is therefore decision packages
  served on NPU, which has not been attempted yet.

## Setup

- Host: 2-socket Kunpeng 950 7590, 8x Ascend 950DT (driver 25.6.rc1,
  CANN 9.1.0), torch 2.10.0+cpu / torch_npu 2.10.0.post4, triton 3.2.0
  (triton-ascend), one card under a co-tenant container.
- Probe: synthetic shapes over the ops above, warmup then timed reps per
  op and dtype, `torch.equal` for exactness against the eager reference;
  the rotate-half kernel needed one semantics fix (the sign of the first
  half) before the bit-exact verdicts hold.
- Data: `a4_kernel_probe.py` output on the measurement host.
