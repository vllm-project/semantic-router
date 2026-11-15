---
title: Quantization
sidebar_label: Quantization
description: The quantized package formats the model runtime loads, what each costs in agreement and saves in memory and latency, and how to choose one.
---

# Quantization

A quantized package holds a model's weights in fewer bits, so the model fits in
less device memory and, on GPUs with the matching matrix units, answers faster.
The model runtime loads Decision 3.0 packages in three compressed-tensors
formats. A quantized package is its own model: its answers are the ones its
weights give, close to but not the same as the BF16 package's.

```bash title="Serve the FP8 package of d3 on an AMD Instinct MI300X or MI325X"
vllm-sr serve vllm-sr/d3-FP8 --revision 2fb271adbdaf6dc969ca1ca746b61a6a5127ed62 --engine --platform rocm
```

## The formats

| Format | Weights | Activations | Device memory of d3 | Keeps of the BF16 answers | Needs |
| --- | --- | --- | ---: | ---: | --- |
| BF16 (the published package) | BF16 | BF16 | 52.2 GB | 100% | Any GPU, or a CPU |
| **FP8** (W8A8) | FP8 E4M3, one scale per output channel | FP8 per token, quantized as each call runs | 31.9 GB | 99.08% | FP8 matrix units: AMD Instinct MI300 or MI350 series, or an NVIDIA GPU of compute capability 8.9 or later |
| **int4** (W4A16), GPTQ or AWQ | 4-bit integers in groups of 128 input columns | BF16 | 16.4 GB (GPTQ), 16.5 GB (AWQ) | 97.98% (GPTQ), 97.76% (AWQ) | Any GPU the model runs on |
| **NVFP4** (W4A4), plain or AWQ | FP4 E2M1 in groups of 16, with FP8 block scales | FP4, quantized per block as each call runs | 17.2 GB | 96.64% (NVFP4), 96.81% (NVFP4 AWQ) | FP4 tensor cores: an NVIDIA Blackwell GPU (compute capability 10.x or 12.x) |

Agreement is measured on the 14,031 test requests of the
[quantization record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/decision3-quantized.md):
the share of answers whose top choice, or Noul side, is the BF16 package's.
Accuracy moved from 0.8325 (BF16) to 0.8324 (FP8), 0.8310 (GPTQ) and 0.8309
(AWQ) on an MI300X, and from 0.8323 to 0.8271 (NVFP4) and 0.8292 (NVFP4 AWQ) on
a Blackwell GPU.

## The trade-offs

- **FP8 keeps the most answers** and saves 39% of the memory. With graphs it is
  faster than BF16 (36.8 ms against 46.4 ms per request on an MI300X); without
  graphs its per-layer launches cost more host time than they save.
- **int4 halves FP8's memory again** but saves memory, not time: its layers
  unpack the weights to BF16 on every call, so a request takes a little longer
  than BF16 (62.0 ms against 48.2 ms with graphs). On NVIDIA GPUs, `--dtype auto`
  runs a fused int4 kernel for up to 256 rows that narrows the gap.
- **NVFP4 is the fastest and smallest**, on the GPUs that have FP4 tensor cores:
  23.2 ms against BF16's 62.6 ms per request on an RTX PRO 6000 Blackwell, in
  17.2 GB. It changes the most answers of the three.

:::note
A quantized package also runs the approximate fused kernels its records allow
(norms, residual adds and gates) at `--dtype auto`; a BF16 package keeps exact
kernels at every dtype. An explicit `--dtype` runs a quantized package with
exact kernels only.
:::

## How to choose

1. **The BF16 package fits and is fast enough?** Keep it: it gives the released
   answers.
2. **It doesn't fit, or you want more headroom on an AMD Instinct MI300 or
   MI350, or an NVIDIA Ada, Hopper or Blackwell GPU?** Use **FP8**.
3. **You need the smallest footprint on another GPU?** Use **int4**, GPTQ or
   AWQ; their agreement is within a quarter of a point of each other.
4. **You serve on NVIDIA Blackwell and latency matters most?** Use **NVFP4**,
   and measure its agreement on your own questions first.

Whatever you pick, compare the package with the BF16 one on representative
requests before rollout, and re-check the predicates of your rules: a package's
probabilities move a little, and near a threshold a decision can flip.

## What quantization is not

- **`--dtype`** runs the same weights in another precision (`float32`,
  `bfloat16` or `float16`) with the reference kernels; it does not load a
  quantized package. See [Engine arguments](../deploy/engine-args.md#precision).
- **Reduced copies** are faster copies of a model's linear layers that batches
  merging requests run, where the records show no decision changed: pre-packed
  FP32 layers on x86 CPUs for Vela 2.0 0.3B and the Decision 1.0 encoders. They
  are on by default and need no package.

## Next

- [Support by hardware and model](./support.md): which device runs which
  format, and which models have quantized packages.
- [Serve a quantized package](./enable.md): the configuration, the checks and
  the errors you may see.
