# CUDA: the gfx942 fused kernels as approximate kernels

The fused element-wise Triton kernels of `accel/triton_gfx942.py` run on
NVIDIA GPUs but do not reproduce the eager ops there, so the CUDA accelerator
registers them as approximate kernels. Only `max_speed` selects approximate
kernels, and only for a family that consents (`DtypePolicy.approximate_kernels`):
Decision 2.0 does, Vela 2.0 does not.

- **Date:** 2026-10-09.
- **Device:** one NVIDIA GeForce RTX 4070 Ti SUPER (sm_89, 16 GB), driver
  591.86, under WSL2.
- **Stack:** PyTorch 2.10.0+cu128 (the CUDA router image's pin), Triton 3.6.0,
  FLA 0.5.2, causal-conv1d 1.7.0.
- **Code:** main of 2026-10-08 with the change that adds this record.

## Not bit-exact on CUDA

- `silu_mul` and `residual_add` equal the eager ops on every shape tried.
- `add_rmsnorm` differs by one BF16 ulp on 15 to 102 of the outputs of a
  4,096-row batch (hidden sizes 1,024, 2,560 and 4,096): it sums each row in
  the order of ATen's ROCm reduction (64-lane wavefront, four accumulators,
  lane tree), and CUDA's reduction sums in another order. Emulated orders with
  8 to 32 lanes, 1 to 32 warps per row and 1 to 8 accumulators matched about
  80% of the rows of ATen CUDA's `mean` at best, so CUDA's order is not one of
  these simple layouts.
- The random backbones of `tests/test_gpu_fast_path.py` (Kai-0.6B, Eos-0.8B
  and Nox-4B dimensions, and Qwen3.5 with LoRA) differ from their eager copies
  on every one of the test's 12 batch shapes, by at most 0.038 to 0.088.

## Decision 2.0: no decision changed

The public many-question request (`tools/many_questions.py`, a ~300-token
ticket) at 1, 4, 16 and 64 questions, and the ticket repeated six times at 1
and 16 questions, loaded through `tools/gpu_parity.py`'s `load` with fused
kernels and graphs, once with `EngineOptions(exact_kernels_only=False)`.
Exact profile batching in both runs, so only the kernels differ. Five untimed
runs, then the p50 of 30.

| Model | Shape | Exact kernels (ms) | Approximate kernels (ms) | Faster |
| --- | --- | ---: | ---: | ---: |
| Kai-0.6B | 1 question | 11.5 | 8.6 | 25% |
| Kai-0.6B | 64 questions | 754.0 | 389.1 | 48% |
| Kai-0.6B | long, 16 questions | 944.5 | 555.1 | 41% |
| Eos-0.8B | 1 question | 11.8 | 9.8 | 17% |
| Eos-0.8B | 64 questions | 717.3 | 471.0 | 34% |
| Eos-0.8B | long, 16 questions | 761.9 | 514.3 | 32% |
| Sol-2B | 1 question | 21.4 | 18.4 | 14% |
| Sol-2B | 64 questions | 1,321.7 | 989.6 | 25% |
| Sol-2B | long, 16 questions | 1,384.8 | 1,048.0 | 24% |

Across the six shapes each model answered 102 questions (Choice, Noul and
Score). None changed its decision. The largest probability difference was
4.8e-3 (Kai), 1.5e-2 (Eos) and 6.3e-3 (Sol). Over all six shapes the
approximate kernels were 11% to 48% faster. Nox-4B outgrew the 16 GB card at
64 questions and spilled to host memory, and Lux-9B and Vega-27B do not fit;
none of the three is measured.

## Vela 2.0: span answers change

`tools/vela2_parity.py --generate 300` against the packages' own engine on
the same GPU, with approximate kernels allowed:

- **0.3B:** 300 of 300 requests identical. The encoder uses only
  `rotary_half`, an FP32 element-wise kernel that is exact on CUDA too.
- **0.8B:** every Choice (316), Score (198), Noul (224), Set (109) and
  relevance (25) decision unchanged, but 23 of 336 span answers changed
  (93.2% unchanged), with a largest probability difference of 0.022. That is
  below the 99% per question type a family needs to consent, so Vela 2.0
  keeps the exact kernels under `max_speed`.

Without approximate kernels the same 300 requests are identical on both sizes,
as before this change.
