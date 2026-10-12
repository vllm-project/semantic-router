# CUDA fused kernels and FLA pins on NVIDIA H200

Issue [#4621](https://github.com/vllm-project/semantic-router/issues/4621)
item 4: bit-exactness of each fused gfx942 slot on CUDA before enabling it,
and pinned FLA kernel choices for the Qwen3.5 decoders. The fused slots are
**not all bit-exact**, so they stay approximate (`exact=False`).
`CUDAAccelerator.validated` stays **false**. This does not close #4621.
Machine readable summary:
[`cuda-fused-kernels.json`](cuda-fused-kernels.json).

The 4070 Ti (`sm_89`) measurement that first marked these kernels approximate
is [`cuda-fused-approximate.md`](cuda-fused-approximate.md). Golden answers
from the published image are [`cuda-golden-answers.md`](cuda-golden-answers.md).

- **Date:** 2026-10-12.
- **Hardware:** one NVIDIA H200 (sm_90, 143771 MiB total). Driver 580.178.04.
  Native BF16. Kernel probe and Decision 2.0 Eos / Sol / Nox / Lux FLA pins:
  Secure Cloud pod with a 100 GB workspace disk. Vega-27B FLA pin: Secure
  Cloud pod `mr8uxlkhub0n6l` in US-CA-2, volume `sr-models-us-ca-2`.
- **Image:** not the published `vllm-sr-cuda` image. RunPod
  `pytorch:1.0.2-cu1281-torch280-ubuntu2404` plus a venv with torch
  2.10.0+cu128, CUDA 12.8, Triton 3.6.0, fla-core 0.5.2, matching the CUDA
  image pins. `causal-conv1d` was not installed. The image has no
  git-revision label.
- **Head:** `83c1ddced3cbe4b9d82b2f9bd2ad0dad23a34356`, the checkout used for
  these measurements.
- **Profile:** `exact`, device `cuda:0`. Fused gfx942 kernels are not
  selected on this profile.

## Fused slots are not all bit-exact

Shapes: 4096 BF16 rows, hidden 1024 / 2560 / 4096 where the slot takes a
hidden size. A slot that is not bit-exact stays approximate. None of these
slots is enabled as exact on CUDA.

| Slot | Result on H200 `sm_90` |
| --- | --- |
| `residual_add` | Exact on 1024 / 2560 / 4096 (0 mismatches). |
| `silu_mul` | Exact against the kernel's documented rounding (SiLU to BF16, then multiply). Not exact against ATen `silu` in FP32 then multiply: 1149524 / 4194304 mismatches on 4096×1024 (~27%), max abs 0.0625. |
| `rotary_half` | Exact on query and key (`[2, 8, 64, 128]` views). |
| `attn_prep` | Deterministic on repeat (0 mismatches vs a second launch). Not compared to an eager reference. |
| `add_rmsnorm` | Hidden stream and normed BF16 output differ from ATen CUDA. The kernel sums in ATen's ROCm reduction order (64-lane wavefront, four accumulators), not CUDA's. On 4096×4096: 9019529 / 16777216 hidden mismatches, max abs 0.015625; 3357157 / 16777216 normed mismatches, max abs 0.125. |
| `gated_rmsnorm` | Not exact: 136416 / 524288 mismatches on 4096×128, max abs 0.125. |
| `sigmoid_gate` | Not exact: 34092 / 131072 mismatches, max abs 0.03125. The kernel rounds `libdevice.exp` sigmoid to BF16 before the multiply. |
| `gdn_prep` | Launched and deterministic. No `causal-conv1d` reference, so not scored exact. |

```text
residual_add:        exact on all three shapes
silu_mul.kernel_round: exact; silu_mul.aten: not exact
rotary_half.q/k:     exact
add_rmsnorm / gated_rmsnorm / sigmoid_gate: not exact
gdn_prep:            deterministic; no causal-conv1d reference
CUDAAccelerator.validated: false
fused slots remain exact=False
```

## FLA kernel choices (`cuda:sm_90`)

`tools/golden_answers.py` with `--autotune-cache`, then
`tools/kernel_choices.py --device-class cuda:sm_90 --fla 0.5.2 --triton 3.6.0`.
Entries land next to `rocm:gfx942` in `registry/kernel_choices.json`.

Kai-0.6B has no autotuned FLA kernels. Decision 1.0 Qwen3.5 decoders copy the
Decision 2.0 table in Python (`registry/tables/decision1.py`). Vela 2.0 0.8B /
4B / 9B reuse Eos / Nox / Lux.

| Model | Autotune + dump |
| --- | --- |
| Decision 2.0 Eos-0.8B | recorded |
| Decision 2.0 Sol-2B | recorded |
| Decision 2.0 Nox-4B | recorded |
| Decision 2.0 Lux-9B | recorded |
| Decision 2.0 Vega-27B | recorded on US-CA-2 volume `sr-models-us-ca-2` (pod `mr8uxlkhub0n6l`) |
| Decision 1.0 Eos / Sol / Nox / Lux | same architecture as Decision 2.0 (Python copy) |
| Vela 2.0 0.8B / 4B / 9B | Eos / Nox / Lux |

Each dump has the six forward kernels the ROCm pins use:
`chunk_fwd_kernel_o`, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`,
`chunk_gated_delta_rule_fwd_kkt_solve_kernel`,
`chunk_local_cumsum_scalar_kernel`, `l2norm_fwd_kernel`,
`recompute_w_u_fwd_kernel`. The golden request records one tuning key per
kernel; unrecorded keys take that first entry, as on ROCm. FLA **0.5.2**,
Triton **3.6.0**.

Vega-27B golden check on that capture: `checked: 3, matched: 3, reference: cuda`.

## Coverage and checks

- Registry: every Qwen3.5 decoder used by Decision 1.0, Decision 2.0 and
  Vela 2.0 now resolves `cuda:sm_90` through `builtin.kernel_choices`.
- CPU tests: `tests/test_vela2_registry.py`, `tests/test_readiness.py`,
  `tests/test_kernel_choices.py`, `tests/test_decision1_package.py`
  (`not gpu and not prod_weights`).
- Fused slots stay off on `exact`. `max_speed` may still select them only
  where a family consents (`cuda-fused-approximate.md`).

## Direct reproduction

Dump FLA choices from a warm autotune cache on `cuda:0`:

```bash
python tools/golden_answers.py vllm-sr/Decision-2.0-Eos-0.8B \
  --device cuda:0 --autotune-cache RELEASE_CACHE
python tools/kernel_choices.py vllm-sr/Decision-2.0-Eos-0.8B \
  --autotune-cache RELEASE_CACHE \
  --device-class cuda:sm_90 --fla 0.5.2 --triton 3.6.0
```

Merge the printed `choices` object under `choices["cuda:sm_90"]` next to
`rocm:gfx942`. Repeat for Sol, Nox, Lux and Vega. Vela 2.0 0.8B / 4B / 9B
entries are copies of Eos / Nox / Lux.

## Scope of the evidence

This is item 4 of #4621: a fused-slot bit-exactness record on one H200, plus
pinned FLA choices for the Qwen3.5 decoders on `cuda:sm_90`. It is not
completion of the issue.

Managed CUDA container serving, Decision 1.0 / 2.0 scored panels, Vela 2.0's
scored request panel, the Vela 1.0 legacy path, performance vs the legacy or
released runtime, and user-guide/design status remain outside this record.
`gdn_prep` has no `causal-conv1d` reference compare. Other NVIDIA
architectures (`sm_89`, Ampere) are not re-pinned here. The global CUDA
`validated` flag remains false.
