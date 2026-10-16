# Decision 3.0 quantized packages

The native engine loads d3 packages whose projections are held as
compressed-tensors FP8 (W8A8, `float-quantized`) or int4 (W4A16,
`pack-quantized`) weights (`quantization.py`, `engines/native/quantized.py`).
A quantized package is its own model: its answers are the ones its weights
give, and this record compares them with the BF16 package's.

- **Date:** 2026-10-14.
- **Device:** one AMD Instinct MI300X (gfx942), `vllm/vllm-openai-rocm` image
  (PyTorch 2.12.0, Triton 3.7.1, FLA 0.5.2), exact kernels only
  (`EngineOptions.exact_kernels_only`).
- **Kernel tables:** also an NVIDIA GeForce RTX 4070 Ti SUPER (Ada, compute
  capability 8.9; PyTorch 2.10.0+cu128, Triton 3.6.0).
- **Baseline:** `vllm-sr/d3` (27B) at the built-in revision `cbff3a3c`.

## Packages

| Package | Made by | Projections | Kept in BF16 | Weights file |
| --- | --- | --- | --- | --- |
| FP8 | per-output-channel E4M3, scale `max / 448` rounded to BF16; FP8 activations per token at run time | 304 | GDN `in_proj_qkv` / `in_proj_z` / `in_proj_a` / `in_proj_b`, the vision tower | 31.9 GB |
| AWQ int4 | AWQ (per-channel scales folded into the norms and `up_proj`, clipping searched per row and group) on 128 rows of the first train shard; asymmetric, group 128 | 400 | `in_proj_a` / `in_proj_b`, the vision tower | 16.2 GB |
| GPTQ int4 | `TheoHsiung/d3-W4A16-G128` (from d3 revision `093cffba`), with a `MODEL_MANIFEST.json` written for it; symmetric, group 128 | 400 | as AWQ | 16.1 GB |

The BF16 package's weights take 52.2 GB. The packages were built with the
scripts in `bench/d3-quant-eval` (`fp8_package.py`, `awq.py`).

## Agreement

The whole test split of `tasksource/tasksource-jev-typed-decisions` at
revision `28c39125`: 14,031 requests of one question each (the rows of
`docs/records/decision3-float8.md`). "Changed" counts answers whose top choice
(or Noul side) differs from the BF16 package's.

| Package | Changed | Agreement | Accuracy (BF16 0.8325) | Device memory (BF16 52.2 GB) |
| --- | --- | --- | --- | --- |
| FP8 | 129 | 99.08% | 0.8324 | 31.9 GB |
| AWQ int4 | 314 | 97.76% | 0.8309 | 16.5 GB |
| GPTQ int4 | 284 | 97.98% | 0.8310 | 16.4 GB |

The AWQ calibration rows (128 of the first train shard) share no state with
the test split. Of the 1,000 train rows used below for the AWQ layer
experiment and the latency, 8 share their state text with a test request.

On 1,000 rows of the train shard, keeping the last one, two or four decoder
layers of the AWQ package in BF16 changed 24, 24 and 25 answers against 24
without: the flips do not come from those layers.

## Latency

Median of a request answered a third time, so that its shape's graph replays,
over the first 500 of those 1,000 train rows, each package against BF16 in the
same process. Decision 3.0 forwards are captured as graphs only with the
change of #4999, so the table also gives main as it is (the first 1,000 rows,
answered a second time):

| Package | BF16 with graphs | Package with graphs | BF16, main | Package, main |
| --- | --- | --- | --- | --- |
| FP8 | 45.2 ms | 36.5 ms | 66.1 ms | 83.3 ms |
| AWQ int4 | 45.1 ms | 60.2 ms | 67.5 ms | 86.1 ms |

FP8 pays per layer for one activation quantization and one scaled GEMM launch,
which costs more host time than it saves without graphs. Triton's launcher binds
and specializes the quantization kernel's arguments on every call (about 17 us
of host time per layer on the MI300X), so after a shape's first call the
compiled kernel is launched directly, once a direct launch has written the same
bytes: about 15 us instead of 21.5 per call. Binding the layer's transposed
views at load saves another 6 us; one gate projection of 64 rows then takes
40 us of host time against about 20 us for the BF16 GEMM, and writes the same
bytes.
Without graphs the FP8 package's median falls from 84.0 and 85.4 ms to 77.5 ms
in alternating runs (d3 68.3 to 71.3 ms). After loading, the
packages hold 52.3 GB (BF16), 32.2 GB (FP8) and 16.9 GB (AWQ) of device memory,
and 52.9, 32.9 and 17.5 GB at the peak of the 20 longest of those 1,000 rows
(up to 5,928 characters of state), each in its own process. Int4 layers unpack
their weights to BF16 on every call and then multiply in BF16: they save
memory, not time.

## Kernels

Microseconds at 64, 256 and 1,024 rows, summed over one of each of d3's
projection shapes (`in_proj_qkv`, `in_proj_z`, `out_proj`, `q_proj`, `k_proj`,
`o_proj`, `gate_proj` twice for `up_proj`, `down_proj`), random weights,
`triton.testing.do_bench`:

| Device | Rows | BF16 GEMM | FP8 (quantize + scaled GEMM) | int4 (unpack + GEMM) |
| --- | --- | --- | --- | --- |
| MI300X | 64 | 585 | 442 | 816 |
| MI300X | 256 | 881 | 609 | 1,223 |
| MI300X | 1,024 | 2,197 | 1,370 | 2,414 |
| RTX 4070 Ti SUPER | 64 | 2,225 | 1,289 | 4,102 |
| RTX 4070 Ti SUPER | 256 | 3,264 | 4,857 | 5,286 |
| RTX 4070 Ti SUPER | 1,024 | 11,585 | 17,737 | 13,655 |

On Ada, `torch._scaled_mm` with per-row scales takes a slow kernel from 256
rows on, while one scale per operand runs at about half the BF16 GEMM's time
(the gate projection at 256 rows: 824 us per row-scaled, 288 us per-tensor,
591 us BF16). There approximate kernels multiply products of 256 rows or more
unscaled into FP32 and apply the scales after (348 us for the same product),
which differs from the per-row result by about one BF16 rounding step (see
below); exact kernels keep the per-row product. On MI300X both scalings run
alike and the per-row product is kept.

A Triton kernel that multiplies straight from the packed int4 weights takes
0.59x the BF16 GEMM's time at 8 to 16 rows on the RTX 4070 Ti SUPER, 0.83x at
64 and 1.55x at 256, against 1.6x to 1.9x for unpacking first. On MI300X it at
best matches the BF16 GEMM at 64 rows and takes 1.5x it at 256, so ROCm keeps
the unpacked product. torch's `_weight_int4pack_mm` took 2.2x to 6.9x the BF16
GEMM's time on the RTX 4070 Ti SUPER at these sizes.

## The fused int4 kernel in a model

The decision3 family allows approximate kernels for a quantized package
(`DtypePolicy.approximate_kernels`), so at `--dtype auto` its int4 layers take
the fused kernel for inputs of up to 256 rows on NVIDIA GPUs; with exact kernels
only (an explicit `--dtype`) they keep the unpacked product. An AWQ int4
`vllm-sr/d3-mini` (made by `awq.py` as above) on the RTX 4070 Ti SUPER, the first
500 of the 1,000 train rows, each answered a third time:

| Package and kernels | Changed against the exact-kernel int4 answers | Median latency | Device memory |
| --- | --- | --- | --- |
| BF16 `d3-mini`, exact kernels | | 40.7 ms | 9.1 GB |
| AWQ int4, exact kernels | | 52.0 ms | 3.9 GB |
| AWQ int4, approximate kernels | 0 of 500 (largest probability change 0.028) | 46.9 ms | 3.9 GB |

Against the BF16 package, the int4 one changes 17 of the 500 answers (96.6%;
accuracy 0.812 against 0.820).

## The unscaled FP8 product in a model

On Ada, approximate kernels (`--dtype auto` for a quantized package) take the
FP8 product of 256 rows or more unscaled; exact kernels only
(`EngineOptions.exact_kernels_only`) keep the per-row scaled product. One
`k_proj`-shaped layer (5,120 to 1,024, 256 rows, activations with outlier
channels) gives bit-identical outputs for 91.5% of its elements unscaled, and
every output above 1% of the largest stays within 2.53 BF16 ulps of the scaled
one. An FP8 `vllm-sr/d3-mini` (made as the FP8 package above) on the RTX 4070
Ti SUPER, the first 500 of the 1,000 train rows (210 of them 256 tokens or
longer), each answered a third time, each setting loaded alone:

| Kernels | Changed against exact kernels | Largest probability change | Accuracy | Median latency |
| --- | --- | --- | --- | --- |
| Exact kernels only | | | 0.826 | 49.5 ms |
| Approximate, scaled product | 8 of 500 | 0.176 | 0.824 | 39.6 ms |
| Approximate (the default) | 7 of 500 | 0.176 | 0.826 | 40.7 ms |

The unscaled product itself changes 1 of the 500 answers against the
approximate kernels with the scaled product (largest probability change
0.087); the other changes come from the fused element-wise kernels. One
request at a time it saves no latency at these lengths; the gain is the
kernel table's, at 256 rows and more, as batches of concurrent requests reach.
