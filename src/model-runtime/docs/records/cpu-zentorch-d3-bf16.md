# CPU: zentorch's BF16 linear for Decision 3.0, checked against its own references

Where a CPU class computes BF16 natively and has its own BF16 references,
`--dtype auto` computes Decision 3.0 in BF16
([record](d3-cpu-bf16-native.md)), with exact kernels. zentorch's FP32 linear
already runs for Decision 3.0 at FP32 on AMD EPYC Turin onwards
([record](cpu-zentorch-d3.md)). This record measures zentorch's linear at BF16.
At BF16 its answers miss the BF16 references recorded with oneDNN, though they
stay as close to FP32 as oneDNN's answers do. This record finds why: a BF16
reference holds one GEMM summation order. It also records references for
zentorch's order under a key that names the kernel,
`cpu:x86_64:avx512:bf16_native:amd:zentorch@bfloat16`.

With those references, BF16 with zentorch's linear serves 1.04 to 1.16 times
faster than BF16 with oneDNN's, in the same memory. That is 1.20 to 1.95 times
faster than FP32 with zentorch's linear, and 1.9 to 3.7 times faster than FP32
alone. On the typed-decisions TEST set it is as far from FP32 as oneDNN's BF16:
its mean |Δp| is the same at every size, and no accuracy difference survives a
correction for the comparisons made.

- **Date:** 2026-11-20.
- **Hosts:** AMD EPYC 9755 (Zen 5, Turin), AVX-512 with BF16; the kernel
  checks also on an AMD EPYC 9454 (Zen 4, Genoa).
- **Image:** Python 3.12 with PyTorch 2.14.1+cpu and zentorch 2.14.1.0 (the
  release built for PyTorch 2.14), the runtime's lock; tcmalloc preloaded.
- **Runtime:** this change at `--dtype auto` with `VLLM_SRUN_ZENTORCH=1`
  (BF16 with zentorch's linear), against #7096's BF16 at `auto` (oneDNN) and
  FP32 with and without zentorch's linear.
- **Packages:** the eight revisions `registry/tables/decision3.py` pins.
- **Threads:** 32 pinned cores per run unless stated.

## Why the oneDNN references miss

Both kernels take BF16 inputs, sum their products in FP32 and round the sum
to BF16 once. Against that sum computed in FP64 and rounded once, over the d3
sizes' linear shapes (162 cases: 9 shapes of `K` from 1,024 to 12,288 and `N`
from 1,024 to 18,432, 1, 17 and 233 rows, with and without a bias; on the
EPYC 9454, and on the EPYC 9755 with the same largest errors):

| Kernel | Outputs off the rounded sum | Mean signed error | RMS error, typical / largest |
| --- | ---: | ---: | --- |
| oneDNN (`F.linear`) | at most 0.1% | within ±0.02 ulp | 0.29 / 0.71 ulp |
| zentorch, prepacked | at most 0.1% | within ±0.02 ulp | 0.29 / 0.42 ulp |

The largest RMS errors come from outputs near zero, where a sum of large
products cancels and one unit of the last place is small. Both kernels are as
exact as one BF16 rounding allows, and neither leans one way. They agree on at
least 99.8% of outputs and round the rest differently, because they add the
products in different orders. A BF16 rounding step is 0.4% to 0.8% of a value,
so through 12 to 32 layers these differences grow into answers that differ by
0.003 to 0.005 in probability on average (below), above the golden check's CPU
tolerance of 1e-3. A BF16 reference therefore checks one kernel's summation
order, as the vendor key in the native BF16 record already reflects for
Intel's and AMD's kernels (#5273).

zentorch's BF16 linear is deterministic, and its outputs are bit for bit the
same at 1, 8, 16 and 32 threads, including with one prepacked weight reused
across thread counts, on both hosts. So one reference per class can check it,
as one checks oneDNN's.

zentorch's weight cache rewrites a weight in place and keys it by address. A
benchmark that frees a weight and allocates another at the same address
gets wrong answers. The runtime keeps each laid-out weight for the model's
lifetime, so it doesn't reuse addresses this way.

## The key names the kernel

At `auto` on a native-BF16 CPU class, `dtypes.on_host` now keeps the CPU
kernels the family consents to only where the class key that also names them
(`with_kernels`: `cpu:x86_64:avx512:bf16_native:amd:zentorch`) has a BF16
reference for every golden request. With references only under the plain key,
the model runs exact kernels against them, as before. The golden check tries
the key naming the kernels, then the same key without the vendor, then
`cpu@bfloat16`.

| Host, `auto` | Linear | Checked against |
| --- | --- | --- |
| AMD, AVX-512 BF16, Turin onwards, zentorch opted in | zentorch | `…:amd:zentorch@bfloat16` |
| AMD, AVX-512 BF16, otherwise | oneDNN | `…:amd@bfloat16` |
| Any other CPU | as before | as before |

If a CPU's BF16 answers fail that key's check, the model loads again in FP32,
as with the plain key.

`tools/golden_answers.py --cpu-bf16` records these references. It records the
zentorch version with them: a new release can sum in another order, and then
its references are recorded again. Answers computed with zentorch never
replace a generic baseline.

## Recorded references

Recorded on the EPYC 9755 at 32 threads for all eight sizes. Each answers its
oneDNN reference to within 1e-3 on 1 to 4 of 5 golden requests: d3-edge 2,
d3-lite 4, d3-nano 3, d3-mini 1, d3-flash 3, d3 2, d3-turbo 4, d3-ultra 4.

Serving with them, from a separate process:

| Sizes | Threads | Golden check |
| --- | --- | --- |
| d3-edge | 1, 8, 32 | matched |
| d3-lite, d3-nano, d3-mini | 32 | matched |

Without the opt-in, d3-edge still computes BF16 with oneDNN, matches
`…:amd@bfloat16`, and answers the TEST set one request at a time bit for bit
as BF16 with oneDNN does.

Each size's TEST answers, one request at a time, are bit for bit the same as
those of the day before's run of the same kernels, whose timings are below.

## Decisions

The typed-decisions TEST set (2,000 decisions: 600 Choice, 600 Noul, 800
Score), one request at a time on the EPYC 9755. A decision is a question's
most probable answer; accuracy counts decisions that match the labelled one.

| Size | Against | Changed | Accuracy before | Accuracy BF16 + zentorch | Right → wrong | Wrong → right | McNemar p | max \|Δp\| | p99 \|Δp\| |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d3-edge | FP32 | 9 | 0.4210 | 0.4195 | 5 | 2 | 0.45 | 0.061 | 0.020 |
| d3-edge | BF16 | 8 | 0.4230 | 0.4195 | 7 | 0 | 0.02 | 0.045 | 0.022 |
| d3-lite | FP32 | 18 | 0.5225 | 0.5220 | 8 | 7 | 1.00 | 0.027 | 0.015 |
| d3-lite | BF16 | 19 | 0.5210 | 0.5220 | 8 | 10 | 0.81 | 0.035 | 0.019 |
| d3-nano | FP32 | 9 | 0.6185 | 0.6200 | 2 | 5 | 0.45 | 0.043 | 0.016 |
| d3-nano | BF16 | 12 | 0.6225 | 0.6200 | 8 | 3 | 0.23 | 0.061 | 0.020 |
| d3-mini | FP32 | 4 | 0.7320 | 0.7305 | 3 | 0 | 0.25 | 0.027 | 0.009 |
| d3-mini | BF16 | 7 | 0.7315 | 0.7305 | 4 | 2 | 0.69 | 0.033 | 0.013 |

Changed decisions against FP32 by question type (Choice / Noul / Score):
d3-edge 4 / 2 / 3, d3-lite 3 / 7 / 8, d3-nano 1 / 2 / 6, d3-mini 3 / 0 / 1.

Mean |Δp| against FP32, per decision:

| Size | oneDNN BF16 | zentorch BF16 | zentorch against oneDNN |
| --- | ---: | ---: | ---: |
| d3-edge | 0.0036 | 0.0037 | 0.0041 |
| d3-lite | 0.0037 | 0.0037 | 0.0045 |
| d3-nano | 0.0034 | 0.0034 | 0.0039 |
| d3-mini | 0.0023 | 0.0022 | 0.0027 |

The two BF16 kernels are the same distance from FP32, and further from each
other than either is from FP32: two independent roundings of one computation.

Every accuracy difference is within ±0.022, the 95% interval of one accuracy
over 2,000 decisions. Two of the 16 comparisons here and in the batched table
below have p below 0.05: d3-edge against oneDNN's BF16 (p = 0.02; oneDNN's
BF16 itself gained four decisions against FP32) and d3-mini against FP32 at
concurrency 8 (p = 0.03). Neither stays below 0.05 after a Holm correction.
On d3-mini, the probability of the labelled answer moves by +0.00000 (c8) and
−0.00003 (c1) on average against FP32, with a standard error of 0.00006, and
by +0.00003 and −0.00005 with oneDNN's BF16: zentorch doesn't move answers
towards wrong labels. Every decision it changes on d3-mini, at either
concurrency, was within 0.0062 of a tie in FP32.

## Batched as served

The same set at concurrency 8:

| Size | Against its one-at-a-time answers: changed | Accuracy c1 / c8 | McNemar p | Against FP32 at c8: right → wrong / wrong → right | McNemar p |
| --- | ---: | --- | ---: | --- | ---: |
| d3-edge | 4 | 0.4195 / 0.4195 | 1.00 | 7 / 4 | 0.55 |
| d3-lite | 8 | 0.5220 / 0.5220 | 1.00 | 7 / 6 | 1.00 |
| d3-nano | 4 | 0.6200 / 0.6210 | 0.62 | 2 / 7 | 0.18 |
| d3-mini | 4 | 0.7305 / 0.7290 | 0.25 | 6 / 0 | 0.03 |

As with oneDNN's BF16 (3 to 13 decisions), batching changes a few decisions.
Batches form by arrival, so batched answers also vary from run to run: the
day before, d3-mini at c8 against FP32 was 7 / 0 (p = 0.02). All of these
changes are at near-ties (above). The golden check plans each request alone.

## Speed and memory

The native BF16 record's protocol on the EPYC 9755: peak resident memory after
loading and the golden check, and the median of 15 runs of a short request and
a long one (about 700 tokens). The four configurations ran interleaved in one
job, two rounds each in opposite orders, averaged; rounds agree within 7%.

| Size | FP32 short / long | FP32 + zentorch | BF16 | BF16 + zentorch | Over BF16 | Over FP32 + zentorch | Peak memory BF16 + zentorch |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| d3-edge | 182 / 988 ms | 111 / 626 ms | 97 / 514 ms | 89 / 494 ms | 1.09x / 1.04x | 1.24x / 1.27x | 2.1 GiB |
| d3-lite | 405 / 2,231 ms | 252 / 1,452 ms | 224 / 1,203 ms | 211 / 1,154 ms | 1.06x / 1.04x | 1.20x / 1.26x | 3.1 GiB |
| d3-nano | 766 / 4,145 ms | 388 / 2,243 ms | 310 / 1,717 ms | 287 / 1,610 ms | 1.08x / 1.07x | 1.35x / 1.39x | 7.3 GiB |
| d3-mini | 1,949 / 11,371 ms | 1,011 / 6,884 ms | 685 / 3,998 ms | 617 / 3,718 ms | 1.11x / 1.08x | 1.64x / 1.85x | 12.0 GiB |
| d3-flash | 3,346 / 18,900 ms | 1,517 / 10,543 ms | 985 / 6,270 ms | 911 / 5,410 ms | 1.08x / 1.16x | 1.66x / 1.95x | 17.9 GiB |

BF16 with zentorch holds each linear's weight once, in zentorch's layout, so
its peak memory is BF16's. The gain over BF16 is bounded by the linears' share
of a BF16 forward, 24% to 46% across these sizes.

## Not measured

- The TEST set for d3-flash and the 26B sizes; their references are recorded
  and checked by their own golden requests only.
- Zen 4 (EPYC 9654, 9454) at model level. zentorch runs on Turin onwards (CPU
  family 1Ah) only; its BF16 linear matches Zen 5's kernel checks above.
- zentorch's SDPA at BF16: Decision 3.0 consents to the linear only.
- The Vela 3.0 d3 sizes: they have no BF16 references and stay FP32 with
  zentorch's linear.
