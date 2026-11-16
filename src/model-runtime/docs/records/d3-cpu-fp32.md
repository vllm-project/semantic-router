# Decision 3.0 computes in FP32 on CPUs

Decision 3.0's BF16 answers on a CPU depend on the CPU's kernels by more than
the golden check's CPU tolerance of 1e-3: an Intel Xeon with AMX answers up to
0.0033 from an AMD EPYC with AVX-512 BF16 (#5273), an AVX2 CPU up to 0.027
([PyTorch 2.14 record](d3-cpu-goldens-pytorch-2.14.md)), and both Xeon and
EPYC select `cpu:x86_64:avx512:bf16_native`, so one of them always failed to
load. Its FP32 answers agree on every CPU class tried to within 6.9e-6. So
`--dtype auto` computes every Decision 3.0 size in FP32 on a CPU, with one
`cpu` reference for every CPU class, and keeps BF16 on GPUs and NPUs.
`--dtype bfloat16` still runs the published BF16 on a CPU and is checked
against the BF16 references of its CPU class. No golden or panel decision
changes between the two.

- **Date:** 2026-11-10.
- **Hosts:** dev-mi325x8-03 and -04, each an AMD EPYC 9575F (Zen 5), two
  sockets in a virtual machine, AVX-512 with BF16, no AMX.
- **Image:** Python 3.12 with PyTorch 2.14.1+cpu, the version the CPU images
  pin, and the runtime's lock; the runtime of this change.
- **Packages:** the eight revisions `registry/tables/decision3.py` pins.
- **Threads:** 16 pinned cores per run unless stated.

## The FP32 references

`tools/golden_answers.py vllm-sr/<size> --device cpu` on -04 recorded each
size's `cpu` reference at `--dtype auto`, now FP32. The recording host's
hardware entry equals the baseline it seeds, so the recorder keeps the one
`cpu` entry. The BF16 references of the [PyTorch 2.14 record](d3-cpu-goldens-pytorch-2.14.md)
keep their values and PyTorch versions under `@bfloat16` keys: `cpu@bfloat16`,
`cpu:x86_64:avx512:bf16_native@bfloat16` and
`cpu:x86_64:avx2:bf16_emulated@bfloat16`.

## Across CPU classes

`vllm-srun`'s loading path (the golden check against the recorded references)
on each class, and the largest difference of any compared value (Choice and
Score probabilities, Noul values) from the `cpu` reference:

| Class (how it runs) | Key it selects | Sizes | Golden check | Largest difference |
| --- | --- | --- | --- | ---: |
| AVX-512 BF16, -04 and -03 | `cpu:x86_64:avx512:bf16_native` | all 8 | 5 of 5 each | 0 |
| AVX2 (`ATEN_CPU_CAPABILITY=avx2 ONEDNN_MAX_CPU_ISA=AVX2`), -04 and -03 | `cpu:x86_64:avx2:bf16_emulated` | all 8 | 5 of 5 each | 1.3e-6 |
| AVX-512 without BF16 (`ONEDNN_MAX_CPU_ISA=AVX512_CORE`) | `cpu:x86_64:avx512:bf16_emulated` | edge to flash | 5 of 5 each | 0 |
| MKL's other SGEMM path (`MKL_CBWR=COMPATIBLE`) | `cpu:x86_64:avx512:bf16_native` | all 8 | 5 of 5 each | 2.9e-6 |
| AVX2 and that SGEMM path, -03 | `cpu:x86_64:avx2:bf16_emulated` | edge to flash | 5 of 5 each | 6.9e-6 |
| No vector ISA (`ATEN_CPU_CAPABILITY=default ONEDNN_MAX_CPU_ISA=SSE41 MKL_CBWR=COMPATIBLE`) | `cpu:x86_64:default:bf16_emulated` | edge to flash | 5 of 5 each | 4.2e-6 |
| 1, 8 and 32 threads | `cpu:x86_64:avx512:bf16_native` | edge, lite | 5 of 5 each | 0, 0, 1.1e-6 |

A class run on both hosts answers the same on both. The largest difference
per size over all classes is 5.5e-8 (d3-ultra) to 6.9e-6 (d3-nano), at least
145 times below the tolerance. In BF16, CPU classes differ by 0.0033 (Xeon
against EPYC) to 0.027 (AVX2 against AVX-512).

### Why FP32 differences stay far below 1e-3

Two CPUs' kernels compute the same products and sums in another order (GEMM
blocking, vector width, how threads split a reduction) and so round them
differently. A rounding moves a value by up to 2^-8 of it in BF16 (8
significand bits) and 2^-24 in FP32 (24 bits). D3 rounds every linear,
attention and normalisation output to BF16, so two kernels' BF16 outputs
differ by whole BF16 steps (0.4 to 0.8% of a value), and the backbone carries
these to answers 0.003 to 0.027 apart. In FP32 the same disagreements start
2^16 = 65,536 times smaller; scaled by that, the BF16 differences become 5e-8
to 4e-7. The measured FP32 differences, 5.5e-8 to 6.9e-6, are of that order,
and every one is far below 1e-3.

Intel's AMX tiles multiply BF16 and int8 only, so a Xeon computes FP32 matrix
products on its AVX-512 FMA units: PyTorch's float32 matmul precision is
`highest` by default and the runtime never lowers it, so oneDNN keeps strict
FP32, and MKL's SGEMM keeps FP32 unless asked for a reduced-precision compute
mode, which nothing here does. MKL picks its SGEMM kernels by CPU (on these
EPYC hosts it reports a generic Intel 64 processor), so a Xeon's order of
operations differs from AMD's; the compatible SGEMM path above stands in for
another kernel's order.

## FP32 against BF16

The FP32 golden answers against every other recorded reference, on -04:

| Size | BF16 AVX-512 | BF16 AVX2 | ROCm BF16 | Decisions changed |
| --- | ---: | ---: | ---: | --- |
| d3-edge | 0.0175 | 0.0249 | 0.0358 (CUDA sm_89: 0.0322) | none |
| d3-lite | 0.0072 | 0.0126 | 0.0087 | none |
| d3-nano | 0.0440 | 0.0174 | 0.0346 | none |
| d3-mini | 0.0238 | 0.0108 | 0.0199 (NPU: 0.0124) | none |
| d3-flash | 0.0037 | 0.0022 | 0.0045 | none |
| d3 | 0.0056 | 0.0035 | 0.0032 | none |
| d3-turbo | 0.0050 | 0.0071 | 0.0063 | none |
| d3-ultra | 0.0006 | 0.0011 | 0.0018 | none |

A decision is a Choice's choice, a Score's most probable level or a Noul's
side of 0.5. `tools/decision3_parity.py`'s panel (19 requests: text, Unicode,
JSON, long text, 1 to 12 questions, images and videos) served at `--dtype auto`
against `--dtype bfloat16` on -04:

| Size | Requests | Largest difference | Decisions changed |
| --- | ---: | ---: | --- |
| d3-edge | 19 | 0.0141 | none |
| d3-lite | 19 | 0.0090 | none |
| d3-nano | 19 | 0.0408 | none |
| d3-mini | 19 | 0.0194 | none |
| d3-flash | 19 | 0.0382 | none |

## Cost on a CPU

The weights take twice the bytes in FP32. Peak resident memory after loading
and the golden check, and the median of 15 runs of two requests with the same
Choice, Noul and Score questions, one with a one-sentence state and one with
an 80-sentence state (about 700 tokens), on -03 with 16 threads:

| Size | Precision | Peak memory | Short request | Long request | AVX2: short | AVX2: long |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| d3-edge | FP32 | 3.1 GiB | 167 ms | 1,013 ms | 167 ms | 1,025 ms |
| d3-edge | BF16 | 2.2 GiB | 91 ms | 573 ms | 910 ms | 3,328 ms |
| d3-lite | FP32 | 4.5 GiB | 379 ms | 2,221 ms | 386 ms | 2,275 ms |
| d3-lite | BF16 | 3.1 GiB | 313 ms | 1,392 ms | 2,233 ms | 7,977 ms |
| d3-mini | FP32 | 19.8 GiB | 2,072 ms | 11,916 ms | 2,174 ms | 11,760 ms |
| d3-mini | BF16 | 12.0 GiB | 818 ms | 4,482 ms | 7,733 ms | 35,259 ms |
| d3-flash | FP32 | 32.5 GiB | 3,982 ms | 20,574 ms | — | — |
| d3-flash | BF16 | 18.0 GiB | 1,141 ms | 7,660 ms | — | — |

The 26B sizes peak at 98.5 GiB in FP32 and 50.6 GiB in BF16. Where the CPU
computes BF16 natively, FP32 takes 1.2 to 3.5 times as long, the larger sizes
the most; where BF16 is emulated (AVX2), FP32 is 3.0 to 5.8 times faster than
BF16.

## The BF16 opt-in

`--dtype bfloat16` on a CPU computes as `--dtype auto` did before this change:
a CPU registers none of D3's kernel variants, and BF16 packages consent to no
approximate kernels and no reduced copies. It matched its references with no
difference at all: all eight sizes on -04 natively and with AVX2 forced, d3-edge
to d3-mini on -03 natively and with AVX2 forced. A CPU class without a BF16
reference compares with `cpu@bfloat16`, recorded on AVX-512 BF16, as it
compared with `cpu` before. An Intel Xeon with AMX selects the same
`cpu:x86_64:avx512:bf16_native@bfloat16` as the EPYC and still fails it at
BF16 (#5273): there `--dtype bfloat16` needs a reference keyed by vendor,
and `--dtype auto` serves.

## Vela 3.0's d3 sizes

vela3-lite, -mini, -flash and -turbo run a pinned Decision 3.0 package with
gated adapters and take its `DtypePolicy`, so they compute in FP32 on a CPU
too: their backbone is D3's and so are its BF16 rounding differences, the ones
that kept [their parity record](vela3-d3-parity.md) from recording CPU
references. They define no golden requests yet; FP32 lets one CPU reference
serve each of them once they do. vela3-lite serves on a CPU at both
precisions: its card says `fp32` at `--dtype auto`, and on -04 the short
request above takes 402 ms in FP32 and 245 ms at `--dtype bfloat16` (median of
5; peak memory 4.7 and 3.2 GiB).

## Quantized packages

A quantized Decision 3.0 package keeps BF16 on a CPU: its format names BF16
activations (int4 W4A16; FP8 and NVFP4 need GPUs), its memory estimate counts
its files, and no CPU reference is recorded for one.

## Not measured

- An Intel Xeon (AMX) and ARM CPUs (NEON, SVE): no such host was reachable.
  The classes above vary the instruction sets, the SGEMM path and the thread
  count on AMD.
