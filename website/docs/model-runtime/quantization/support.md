---
title: Quantization support by hardware and model
sidebar_label: Support by hardware and model
sidebar_position: 1
description: Which devices run FP8, int4 and NVFP4 packages, which kernels they take there, and which models have quantized packages.
---

# Quantization support by hardware and model

Each format needs matrix units the device may not have, and the runtime checks
that before any weight loads: a package the device cannot run is refused with
the reason, never loaded in BF16 instead. Only Decision 3.0 packages may hold
quantized weights.

## By hardware

| Device | FP8 (W8A8) | int4 (W4A16) | NVFP4 (W4A4) |
| --- | --- | --- | --- |
| AMD Instinct MI300X, MI325X (gfx942) | Yes (E4M3 FNUZ) | Yes | No |
| AMD Instinct MI350 series (gfx950) | Yes (E4M3) | Yes | No |
| Other AMD GPUs with BF16 | No | Yes, not validated | No |
| NVIDIA Ampere (compute capability 8.0, 8.6) | No | Yes, with the fused kernel | No |
| NVIDIA Ada, Hopper (compute capability 8.9, 9.0) | Yes | Yes, with the fused kernel | No |
| NVIDIA Blackwell (compute capability 10.x, 12.x) | Yes | Yes, with the fused kernel | Yes, with Triton |
| CPU | No | Not measured | No |

Decision 3.0 holds its weights in BF16 on a GPU, so a GPU without BF16 support
runs none of its packages.

- **FP8** multiplies in FP8 where PyTorch offers a scaled FP8 product: on ROCm
  for gfx942 and gfx950, and on CUDA from compute capability 8.9. Any other
  device refuses an FP8 package with "FP8 weights need an NVIDIA GPU of compute
  capability 8.9 or later or an AMD Instinct MI300 or MI350 series GPU".
- **int4** needs no special units: its layers unpack the weights to BF16 and
  multiply in BF16. On NVIDIA GPUs, `--dtype auto` multiplies inputs of up to
  256 rows straight from the packed weights with a fused Triton kernel; ROCm
  keeps the unpacked product, which is faster there.
- **NVFP4** needs FP4 tensor cores, which only NVIDIA Blackwell GPUs have, a
  PyTorch with FP4 tensors, and Triton to quantize the activations. Any other
  device refuses the package before its weights load, naming which of these is
  missing.
- The ROCm image's PyTorch has no CPU LAPACK, which Decision 3.0's Qwen3.5
  backbone needs on a CPU, so no Decision 3.0 package, quantized or not, runs
  on the ROCm image's CPU.

<details>
<summary>Kernel details from the record</summary>

- On Ada, `torch._scaled_mm` with per-row scales takes a slow kernel from 256
  rows on. There `--dtype auto` multiplies FP8 products of 256 rows or more
  unscaled into FP32 and applies the scales after; an explicit `--dtype` keeps
  the per-row product.
- On compute capability 12.x, graph captures of up to 512 rows multiply NVFP4 in
  FlashInfer's FP4 GEMM, which needs a CUDA toolkit of 12.9 or later with Ninja
  to build its timing kernel; without one, the runtime keeps cuBLAS and says
  why. The CUDA image keeps cuBLAS.
- With an autotune cache (`--autotune-cache`), the first process records each
  shape's tiles and later processes load them untimed, so their answers agree.

The [quantization record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/decision3-quantized.md)
has the measurements.

</details>

## By model

| Family | Quantized packages | Notes |
| --- | --- | --- |
| [Decision 3.0](../supported-models/decision-3/overview.md) | Yes: FP8, int4 (GPTQ, AWQ) and NVFP4 in the compressed-tensors format | Every size's family loads them; packages are published for `vllm-sr/d3` |
| [Vela 3.0](../supported-models/vela-3/overview.md) | No | The larger sizes run a pinned BF16 Decision 3.0 package with adapters |
| [Decision 2.0](../supported-models/decision-2/overview.md), [Decision 1.0](../supported-models/decision-1/overview.md) | No | Run at their published precision |
| [Vela 2.0](../supported-models/vela-2/overview.md) | No | Vela 2.0 0.3B runs a pre-packed FP32 copy on x86 CPUs |

The published packages of `vllm-sr/d3` (27B), at the revisions the record
measured:

| Package | Format | Revision |
| --- | --- | --- |
| [`vllm-sr/d3-FP8`](https://huggingface.co/vllm-sr/d3-FP8) | FP8 | `2fb271adbdaf6dc969ca1ca746b61a6a5127ed62` |
| [`vllm-sr/d3-W4A16-GPTQ`](https://huggingface.co/vllm-sr/d3-W4A16-GPTQ) | int4, GPTQ (symmetric, group 128) | `93aa927c5bbf184770ecd144972bed9a1450a4bd` |
| [`vllm-sr/d3-W4A16-AWQ`](https://huggingface.co/vllm-sr/d3-W4A16-AWQ) | int4, AWQ (asymmetric, group 128) | `39ffacd3644dad729b6885ac986e740e8db7e22d` |
| [`vllm-sr/d3-NVFP4`](https://huggingface.co/vllm-sr/d3-NVFP4) | NVFP4 | `07fa5a04859155eb99a034e2e424d423dd50ce1d` |
| [`vllm-sr/d3-NVFP4-AWQ`](https://huggingface.co/vllm-sr/d3-NVFP4-AWQ) | NVFP4 with AWQ smoothing | `27dbfdc48957f9cbc3b83ac49a58d3434451a0a9` |

:::note
These packages are not built in, so they have no recorded reference answers:
their self-check reports `unverified`. Each package's identity is checked
against its own `MODEL_MANIFEST.json`.
:::
