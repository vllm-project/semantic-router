# Decision 3.0 quantized packages

The native engine loads d3 packages whose projections are held as
compressed-tensors FP8 (W8A8, `float-quantized`), int4 (W4A16,
`pack-quantized`) or NVFP4 (W4A4, `nvfp4-pack-quantized`, below) weights
(`quantization.py`, `engines/native/quantized.py`).
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

| Package | Repository | Made by | Projections | Kept in BF16 | Weights file |
| --- | --- | --- | --- | --- | --- |
| FP8 | `vllm-sr/d3-FP8` `2fb271ad` | per-output-channel E4M3, scale `max / 448` rounded to BF16; FP8 activations per token at run time | 304 | GDN `in_proj_qkv` / `in_proj_z` / `in_proj_a` / `in_proj_b`, the vision tower | 31.9 GB |
| AWQ int4 | `vllm-sr/d3-W4A16-AWQ` `39ffacd3` | AWQ (per-channel scales folded into the norms and `up_proj`, clipping searched per row and group) on 128 rows of the first train shard; asymmetric, group 128 | 400 | `in_proj_a` / `in_proj_b`, the vision tower | 16.2 GB |
| GPTQ int4 | `vllm-sr/d3-W4A16-GPTQ` `93aa927c` | GPTQ by Theo Hsiung (from d3 revision `093cffba`, whose weights equal the built-in revision's), with a `MODEL_MANIFEST.json` written for it; symmetric, group 128 | 400 | as AWQ | 16.1 GB |

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
change of #4999; without graphs, the first 1,000 rows are answered a second
time. Measured on 2026-10-16 with this runtime as of 2026-10-14 and the graph
capture of #4999:

| Package | BF16 with graphs | Package with graphs | BF16 without graphs | Package without graphs |
| --- | --- | --- | --- | --- |
| FP8 | 46.4 ms | 36.8 ms | 67.1 ms | 80.9 ms |
| AWQ int4 | 48.2 ms | 62.0 ms | 72.5 ms | 86.9 ms |

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
packages hold 52.2 GB (BF16), 31.9 GB (FP8) and 16.5 GB (AWQ) of weights,
and 52.9, 32.8 and 17.5 GB at the peak of the 20 longest of those 1,000 rows
(up to 5,928 characters of state), each in its own process. Int4 layers unpack
their weights to BF16 on every call and then multiply in BF16: they save
memory, not time.

## Kernels

Microseconds at 64, 256 and 1,024 rows, summed over one of each of d3's
projection shapes (`in_proj_qkv`, `in_proj_z`, `out_proj`, `q_proj`, `k_proj`,
`o_proj`, `gate_proj` twice for `up_proj`, `down_proj`), random weights,
`triton.testing.do_bench` (the MI300X on 2026-10-15):

| Device | Rows | BF16 GEMM | FP8 (quantize + scaled GEMM) | int4 (unpack + GEMM) |
| --- | --- | --- | --- | --- |
| MI300X | 64 | 577 | 429 | 809 |
| MI300X | 256 | 877 | 611 | 1,211 |
| MI300X | 1,024 | 2,220 | 1,364 | 2,408 |
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
(`EngineOptions.exact_kernels_only`) keep the per-row scaled product. Before
PyTorch 2.12 the per-row kernel cannot run in a graph (its stream-K workspace
is zeroed on the default stream, pytorch/pytorch#177573), so there approximate
kernels take every product unscaled and exact kernels run without graphs. One
`k_proj`-shaped layer (5,120 to 1,024, 256 rows, activations with outlier
channels) gives bit-identical outputs for 91.5% of its elements unscaled, and
every output above 1% of the largest stays within 2.53 BF16 ulps of the scaled
one. An FP8 `vllm-sr/d3-mini` (made as the FP8 package above) on the RTX 4070
Ti SUPER with PyTorch 2.10, the first 500 of the 1,000 train rows (210 of them
256 tokens or longer), each answered a third time, each setting loaded alone
(the scaled product forced under approximate kernels runs without graphs):

| Kernels | Changed against exact kernels | Largest probability change | Accuracy | Median latency |
| --- | --- | --- | --- | --- |
| Exact kernels only | | | 0.826 | 50.5 ms |
| Approximate, scaled product | 8 of 500 | 0.176 | 0.824 | 39.6 ms |
| Approximate (the default) | 5 of 500 | 0.142 | 0.822 | 38.4 ms |

The unscaled product itself changes 3 of the 500 answers against the
approximate kernels with the scaled product (largest probability change
0.109); the other changes come from the fused element-wise kernels. One
request at a time it saves little latency at these lengths; the gain is the
kernel table's, at 256 rows and more, as batches of concurrent requests reach.

## NVFP4 on Blackwell

NVFP4 packages multiply on FP4 tensor cores, which only Blackwell GPUs have, so
they were measured on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition
(compute capability 12.0; PyTorch 2.11.0+cu128), against the BF16 package on
the same GPU, with the default engine options (graphs, fused kernels).

| Package | Repository | Made by | Projections | Kept in BF16 | Weights file |
| --- | --- | --- | --- | --- | --- |
| NVFP4 | `vllm-sr/d3-NVFP4` `07fa5a04` | FP4 E2M1 weights rounded to nearest, groups of 16 with E4M3 block scales under an FP32 global scale per tensor; activations the same per request, under a global scale calibrated per layer on 512 prompts of the validation split | 400 | as AWQ int4 | 17.2 GB |
| NVFP4 AWQ | `vllm-sr/d3-NVFP4-AWQ` `27dbfdc4` | AWQ smoothing (scales folded into the norms and `up_proj`) on 512 prompts of the validation split, then as NVFP4 | 400 | as AWQ int4 | 17.2 GB |

The same 14,031 test requests as above; the BF16 package answers 0.8323 of
them on this GPU:

| Package | Changed | Agreement | Accuracy (BF16 0.8323) | Device memory (BF16 52.2 GB) |
| --- | --- | --- | --- | --- |
| NVFP4 | 472 | 96.64% | 0.8271 | 17.2 GB |
| NVFP4 AWQ | 448 | 96.81% | 0.8292 | 17.2 GB |

Two calibration states are also the states of 2 of these requests.

Median latency as above (the first 500 of the 1,000 train rows answered a
third time with graphs; all 1,000 answered a second time without), with exact
kernels only and at `--dtype auto`, the default of `vllm-srun serve`. At
`--dtype auto` a quantized package also runs the fused element-wise kernels the
decision3 family allows it (norms, residual adds and gates; approximate on
CUDA, `docs/records/cuda-fused-approximate.md`); the agreement above was
answered so. A BF16 package keeps exact kernels at every dtype.

| Package | Exact, graphs | Exact, without graphs | `--dtype auto`, graphs | `--dtype auto`, without graphs |
| --- | --- | --- | --- | --- |
| BF16 | 62.6 ms | 63.1 ms | as exact | as exact |
| NVFP4, the JIT launch and torch scale layout | 30.2 ms | 60.0 ms | | |
| NVFP4 | 28.5 ms | 48.0 ms | 23.2 ms | 37.4 ms |
| NVFP4 AWQ | 28.5 ms | 47.7 ms | 23.2 ms | 37.8 ms |

Without graphs an NVFP4 forward is host-bound: it launched 7,196 kernels, and
the GPU was busy 46% of the time. Each layer now launches two of them. The
Triton kernel writes the activation scales in cuBLAS's 32x4x4 layout, which
replaces three torch kernels per call, and the layer skips copies of inputs
that are already BF16 and contiguous. After a shape's first call, the compiled
kernel is launched directly, as the FP8 quantization is. After loading, the
NVFP4 packages hold 17.2 GB of device memory, and 17.9 GB at the peak of the 20
longest rows (52.2 and 52.8 GB for BF16).

In both NVFP4 packages, q / k / v and gate / up share one input global scale
and one weight global scale in every layer (the Gated DeltaNet `in_proj_qkv` /
`in_proj_z` share only the input scale, in all but one layer). Their weights
are merged at load into one layer per group, so one quantization and one
block-scaled product replace two or three. Each column comes out the same as
the separate products from 1 to 9,000 rows, and the parts keep views of the
merged weight, so device memory is unchanged. q and k reach `attn_prep`, and
gate and up reach `silu_mul`, as column-slice views of the merged product, so a
layer's projections peak no higher than the separate ones (89.1 MB both at
2,048 rows of d3's q / k / v widths; copying q and k out took 142.6 MB). The
`in_proj_qkv` / `in_proj_z` pair shares its input quantization. On the 1,000 test requests of the
agreement table, every answer is the same byte for byte. At `--dtype auto`, two
alternating runs of each:

| NVFP4 | Graphs | Without graphs | Device memory after loading |
| --- | --- | --- | --- |
| Separate projections | 23.3 / 23.4 ms | 37.5 / 37.9 ms | 17.2 GB |
| Merged projections | 21.5 / 21.5 ms | 36.5 / 36.5 ms | 17.2 GB |

### FlashInfer's FP4 GEMM in graphs

For the few hundred rows of a d3 request, cuBLASLt's block-scaled FP4 product
reads the NVFP4 weights below the GPU's memory bandwidth. With approximate
kernels on compute capability 12.x, graph captures of up to 512 rows multiply
in FlashInfer's CUTLASS FP4 GEMM instead (`accel/flashinfer_nvfp4.py`, #5290),
its tile tuned per weight shape and power of two of rows at load. A FlashInfer
call takes 46 us of host time against 17 for `scaled_mm`, so eager forwards keep
cuBLAS. Of the NVFP4 products of 200 train requests (each answered twice),
16.6% have up to 128 rows, 58.1% up to 256, 80.4% up to 512 and 95.1% up to
1,024.

The forward's 400 products (merged as above, a distinct weight per product,
13.7 GB) captured in one graph per kernel and replayed alternately, PyTorch
2.13.0+cu130, FlashInfer 0.7.0.post1:

| Rows | cuBLAS | FlashInfer | Ratio |
| --- | --- | --- | --- |
| 16 | 13.64 ms | 10.49 ms | 0.77 |
| 64 | 12.65 ms | 10.92 ms | 0.86 |
| 128 | 12.91 ms | 10.76 ms | 0.83 |
| 192 | 14.49 ms | 12.87 ms | 0.89 |
| 256 | 14.99 ms | 14.49 ms | 0.97 |
| 320 | 19.47 ms | 19.45 ms | 1.00 |
| 384 | 20.16 ms | 20.01 ms | 0.99 |
| 512 | 25.83 ms | 23.90 ms | 0.93 |

Each of FlashInfer's 32 tiles writes the bytes CUDA 12.8's cuBLAS writes for
d3's widths at the row counts tried with random inputs (1 to 512), but not for
every row count of real activations: tiles round some sums differently. FlashInfer's tuner
times each tile after a delay kernel that FlashInfer 0.7 compiles ahead of
time for other GPUs only, so on 12.x it compiles it at first use and needs a
CUDA toolkit of 12.9 or later with Ninja; without one, every product would take
FlashInfer's default tile, which is no faster than cuBLAS and fails in a graph
after a few captures, so the runtime keeps cuBLAS and the receipt's
`nvfp4_product` says why. The CUDA image (PyTorch 2.10.0+cu128, no compiler)
keeps cuBLAS.

`vllm-sr/d3-NVFP4` at `--dtype auto`, as the latency tables above (the first
500 of the 1,000 train rows answered a third time with graphs, two alternating
runs of each), and the first 1,000 test requests of the agreement table
answered a second time (the graphs the first pass captured replaying), on two
stacks: PyTorch 2.11.0+cu128 (cuBLAS 12.8) with a CUDA 12.9 compiler for
FlashInfer, and the `vllm/vllm-openai:v0.32.0` image (PyTorch 2.13.0+cu130):

| Stack | Product | Median latency | Mean | Load | Device memory after loading | Train answers changed | Test answers changed | Test accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CUDA 12.8 | cuBLAS | 21.3 / 21.5 ms | 30.9 / 31.4 ms | 12 s (26 s cold) | 16.04 GB | | | 0.805 |
| CUDA 12.8 | FlashInfer | 20.5 / 20.6 ms | 29.7 / 30.1 ms | 23 s (28 s cold) | 16.08 GB | 8 and 16 of 500; 0 and 1 decisions | 5 of 1,000; no decision | 0.805 |
| CUDA 13 | cuBLAS | 21.3 / 21.4 ms | 31.1 / 31.2 ms | 14 to 17 s | 16.04 GB | | | 0.799 |
| CUDA 13 | FlashInfer | 20.3 / 20.5 ms | 29.9 / 30.1 ms | 22 to 24 s | 16.11 GB | 2 and 2 of 500; no decision | 2 of 1,000; no decision | 0.799 |

The median falls 3.8% (CUDA 12.8) and 4.5% (CUDA 13), short of the 5% #5290
asked for. Loading takes about 10 s longer for the tuning and the delay
kernel's first build. The changed decision is a `corr2cause` choice whose top
two options stood 0.495 and 0.447; the largest probability change is 0.061.
A request's answer can differ between its first forward (eager, cuBLAS) and
later ones (graphs, FlashInfer): 3 and 5 of 150 train requests did. Processes
that time their own tiles answered 144 of 150 alike; with an autotune cache
(`--autotune-cache`) the first process records each shape's tiles there and
later ones load them untimed, and three processes answered all 150 alike.
