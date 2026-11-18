# CPU: zentorch's FP32 linear and SDPA for Decision 3.0 and the Vela 3.0 d3 sizes

[cpu-zentorch-linear-vela3.md](cpu-zentorch-linear-vela3.md) added zentorch's
FP32 linear as the CPU accelerator's approximate `linear` kernel, run where
`VLLM_SRUN_ZENTORCH` is set on AMD EPYC Turin onwards, at `--dtype auto`, for a
family that consents on CPUs. This record adds zentorch's `zentorch_sdpa` in the
`sdpa` slot, and is the evidence for Decision 3.0's consent on CPUs to both
slots. The Vela 3.0 d3 sizes (vela3-lite, -mini, -flash, -turbo) share that
policy. Consent is now per slot (`DtypePolicy.approximate_cpu_kernels` names
the slots a record covers), so the vela3 family keeps the linear only, as its
records measured: vela3-fast in its own record, vela3.1-fast below. Quantized
Decision 3.0 packages keep exact kernels. Issue: #6022.

- **Date:** 2026-11-17 and 2026-11-18.
- **Device:** a 32-core slice of an AMD EPYC 9755 (Turin): four whole CCDs in
  one NUMA node, SMT off, threads bound one per core
  (`OMP_PROC_BIND=close OMP_PLACES=cores`), tcmalloc preloaded.
- **Stack:** PyTorch 2.14.1+cpu, zentorch 2.14.1.0.
- **Precision:** FP32, Decision 3.0's `--dtype auto` on CPUs
  ([d3-cpu-fp32.md](d3-cpu-fp32.md)).
- **Code:** #6753's head (`b99e46794`) with this change; main at #6753's merge
  (`bbe39b3f7`) differs from it only in `requirements-lock.txt`'s `filelock`
  pin. vela3.1-fast ran on this change rebased on main `fcee51341`, which
  registers it. The linear-only column narrowed the consented slots through a
  measurement-only switch that is not part of the change.
- **Packages:** the revisions `registry/tables/decision3.py` and
  `registry/tables/vela3_d3.py` pin (d3-edge `18d61df7`, d3-lite `a8d25a91`,
  d3-nano `3aeb1c22`, d3-mini `a475367d`, vela3-lite `c789afd6`, vela3-mini
  `bbc0206e`; vela3-fast `7cbefd4b`, vela3.1-fast `0ed6ed30`).

## Kernels compared

| Column | `linear` slot | `sdpa` slot |
| --- | --- | --- |
| Current | `F.linear` (the reference) | `F.scaled_dot_product_attention` (the reference) |
| zentorch linear | `zentorch_linear_unary` on a weight laid out once | the reference |
| zentorch linear + SDPA (this change) | `zentorch_linear_unary` | `zentorch_sdpa` |

`zentorch_sdpa` takes grouped key and value heads and additive masks of any
broadcast shape. It rejects boolean masks, so the slot turns a boolean mask
additive (0 where allowed, the dtype's minimum elsewhere); an input the op
rejects runs the reference, logged once per reason. No run below fell back.

## The typed-decisions TEST set, served

All 400 TEST cases (`LocalLLaMA/typed-decisions`: 2,000 Choice, Noul and Score
questions), each request alone at concurrency 1 through `/v1/systemone`, three
warm-up requests first. The three columns ran back to back on one host,
interleaved, twice for the smaller sizes and once for d3-mini and vela3-mini;
decisions are compared with the current kernels' first round. A decision is a
Choice's choice, a Score's most probable level or a Noul's side of 0.5.

| Size | Kernels | p50 per request | Speed-up | Accuracy | Decisions changed | max \|Δp\| | Golden check |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| d3-edge | Current | 806, 796 ms | 1.00x | 0.421 | 0 of 2,000 (round 2) | 0 | matched 5/5 |
| | zentorch linear | 508, 511 ms | 1.57x | 0.421 | 0 of 2,000 | 9.1e-6 | matched 5/5 |
| | **linear + SDPA** | 497, 505 ms | **1.60x** | 0.421 | **0 of 2,000** | 1.0e-5 | matched 5/5 |
| d3-lite | Current | 1,786, 1,801 ms | 1.00x | 0.522 | 0 of 2,000 (round 2) | 0 | matched 5/5 |
| | zentorch linear | 1,179, 1,185 ms | 1.52x | 0.522 | 0 of 2,000 | 5.5e-6 | matched 5/5 |
| | **linear + SDPA** | 1,198, 1,196 ms | **1.50x** | 0.522 | **0 of 2,000** | 7.6e-6 | matched 5/5 |
| d3-nano | Current | 3,511, 3,520 ms | 1.00x | 0.619 | 0 of 2,000 (round 2) | 0 | matched 5/5 |
| | zentorch linear | 1,963, 1,953 ms | 1.79x | 0.619 | 0 of 2,000 | 6.3e-6 | matched 5/5 |
| | **linear + SDPA** | 1,975, 1,949 ms | **1.79x** | 0.619 | **0 of 2,000** | 7.3e-6 | matched 5/5 |
| d3-mini | Current | 9,877 ms | 1.00x | 0.732 | | 0 | matched 5/5 |
| | zentorch linear | 5,854 ms | 1.69x | 0.732 | 0 of 2,000 | 4.4e-6 | matched 5/5 |
| | **linear + SDPA** | 5,811 ms | **1.70x** | 0.732 | **0 of 2,000** | 3.7e-6 | matched 5/5 |
| vela3-lite | Current | 1,780, 1,814 ms | 1.00x | 0.522 | 0 of 2,000 (round 2) | 0 | no references |
| | zentorch linear | 1,183, 1,203 ms | 1.51x | 0.522 | 0 of 2,000 | 5.5e-6 | no references |
| | **linear + SDPA** | 1,163, 1,183 ms | **1.53x** | 0.522 | **0 of 2,000** | 7.6e-6 | no references |
| vela3-mini | Current | 9,748 ms | 1.00x | 0.714 | | 0 | no references |
| | zentorch linear | 5,964 ms | 1.63x | 0.714 | 0 of 2,000 | 3.7e-6 | no references |
| | **linear + SDPA** | 5,907 ms | **1.65x** | 0.714 | **0 of 2,000** | 3.5e-6 | no references |

- **Speed:** zentorch's linear and SDPA together answer 1.50 to 1.79 times as
  fast as the reference kernels; round to round, every configuration moved by
  2% or less. Decision 3.0 runs full attention in one layer of four (the
  others are Gated DeltaNet), so at these sizes the linear carries the gain and
  SDPA holds it: linear and SDPA together are within 2% of the linear alone.
- **Accuracy:** no decision changed and TEST accuracy is identical in every
  size. The largest probability shift is 1.0e-5, 100 times below the golden
  check's 1e-3 CPU tolerance, and every golden check matched. The current
  kernels' second round was bit-identical to their first.
- **The Vela 3.0 d3 sizes** answer these questions with their adapters off,
  through the d3 backbone they pin: vela3-lite's answers equal d3-lite's
  bit for bit. Their adapter and span paths are measured next. They define no
  golden requests yet.

## The Vela 3.0 d3 sizes' routed paths

`tools/vela3_d3_parity.py`'s generated requests (seed 0) cover every path the
family routes: ordinary Choice, Noul and Score questions (adapters off), every
trained router question (the decision adapter and the task adapters),
`hazard:` Nouls, and span questions of every route (PII, unsupported claims,
toxic, evidence, relations, named and described open labels), over text,
multi-part states, CJK text and long targets. Each request alone, served at
`--dtype auto` in the runtime; compared with the current kernels:

| Size | Kernels | Requests | Questions changed | Span sets changed | max \|Δp\| | Span max \|Δp\| | p50 per request |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vela3-lite | zentorch linear | 200 | 0 of 1,498 | 0 of 391 | 7.3e-6 | 3.7e-6 | 2,952 ms (1.42x) |
| | **linear + SDPA** | 200 | **0 of 1,498** | **0 of 391** | 1.0e-5 | 3.2e-6 | 2,887 ms (1.45x) |
| vela3-mini | zentorch linear | 100 | 0 of 693 | 0 of 199 | 2.8e-6 | 2.4e-6 | 9,889 ms (1.70x) |
| | **linear + SDPA** | 100 | **0 of 693** | **0 of 199** | 2.7e-6 | 3.1e-6 | 10,133 ms (1.66x) |

The questions are 618 Choice, 738 Noul and 142 Score for vela3-lite, 333, 286
and 74 for vela3-mini. Every applied span threshold and every input-token count
is identical, and the largest difference is a tenth of the tool's CPU bar of
1e-4.

## Batched as served

The runtime batches requests by default, and a row's padding and GEMM shapes
depend on its neighbours. The TEST set again, at concurrency 1 and at
concurrency 8 with the default batching window, current kernels against
zentorch at each model's consented slots (vela3-fast: the linear; Decision
3.0: the linear and SDPA):

| Model | Batching, current (c1 → c8) | Batching, zentorch (c1 → c8) | zentorch vs current at c8 | Accuracy at c1 and c8, both kernels | zentorch speed-up at c8 |
| --- | --- | --- | --- | ---: | ---: |
| vela3-fast | 0 of 1,200; 0 | 0 of 1,200; 0 | 0 of 1,200; 4.1e-6 | 0.474 | 1.88x |
| d3-edge | 0 of 2,000; 4.6e-6 | 0 of 2,000; 9.6e-6 | 0 of 2,000; 9.2e-6 | 0.421 | 1.60x |
| d3-lite | 0 of 2,000; 3.4e-6 | 0 of 2,000; 5.6e-6 | 0 of 2,000; 7.6e-6 | 0.522 | 1.54x |
| d3-nano | 0 of 2,000; 3.2e-6 | 0 of 2,000; 5.5e-6 | 0 of 2,000; 7.3e-6 | 0.619 | 1.89x |

Each cell gives the decisions changed and the largest probability shift.
Batching moves Decision 3.0's FP32 answers by a few millionths with either
kernel set and changes none; vela3-fast answers bit for bit the same alone or
batched. The speed-up holds under batching (throughput, requests per second).

## vela3.1-fast

The vela3 family also serves vela3.1-fast (main registers it, #6844), so it
takes the family's consent to the linear. The TEST set as above (alone,
current kernels and zentorch's linear interleaved, twice), then batched at
concurrency 8:

| Kernels | p50 alone | Speed-up | Batched (c8) | Speed-up | Accuracy | Decisions changed | max \|Δp\| | Golden check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Current | 453, 460 ms | 1.00x | 2.30 req/s | 1.00x | 0.528 | 0 of 1,200 (round 2, c8) | 0 | matched 8/8 |
| **zentorch linear** | 242, 262 ms | **1.81x** | 3.96 req/s | **1.72x** | 0.528 | **0 of 1,200** (both rounds, c8) | 4.2e-6 | matched 8/8 |

The 1,200 decisions are 600 Choice and 600 Noul. Batching changes no answer
with either kernel set: each answers bit for bit the same alone and at
concurrency 8. zentorch's two rounds differ by 8% in p50.

Its routed paths: `tools/vela3_parity.py`'s generated requests (seed 0, 200
requests: router adapters, Choice, Set and Noul questions, and spans of every
route, over text, multi-part and JSON states, CJK text and parts long enough to
be read in windows), each alone through the runtime at `--dtype auto`. The
package carries no Vela 3.0 reference runtime, so the current kernels' answers
are the baseline:

| Kernels | Requests | Questions changed | Span sets changed | max \|Δp\| | Span max \|Δp\| | p50 per request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current | 200 | | | 0 | 0 | 1,220 ms (1.00x) |
| **zentorch linear** | 200 | **0 of 664** | **0 of 251** | 2.9e-6 | 3.2e-6 | 720 ms (**1.69x**) |

The questions are 483 Choice, 143 Noul and 38 Set. Every applied threshold and
every input-token count is identical. On 11 requests longer than the model
accepts, both kernel sets refuse the same 17 questions with
`max_length_exceeded`.

## The larger sizes

d3-flash, d3, d3-turbo, d3-ultra, vela3-flash and vela3-turbo, current
kernels against linear and SDPA. Decision 3.0 gets the golden check at load and
the 11 text requests of `tools/decision3_parity.py`'s panel (Unicode, JSON,
long text, 1 to 12 questions; each request twice). The panel's image and video
requests are in the next section. vela3-flash and vela3-turbo get 40 of
`vela3_d3_parity.py`'s routed requests (seed 0) each. Packages: d3-flash
`521525ef`, d3 `cbff3a3c`, d3-turbo `689b3af1`, d3-ultra `52ba9e7a`,
vela3-flash `af48828d`, vela3-turbo `ce95e6d9`.

| Size | Requests | Answers changed | max \|Δp\| | Golden check | p50 per request, current → linear + SDPA |
| --- | --- | ---: | ---: | --- | ---: |
| d3-flash | panel, 11 | **0** | 3.8e-6 | matched 5/5 | 3,455 → 1,566 ms (2.21x) |
| d3 | panel, 11 | **0** | 8.2e-6 | matched 5/5 | 11,561 → 4,279 ms (2.70x) |
| d3-turbo | panel, 11 | **0** | 2.6e-6 | matched 5/5 | 8,930 → 3,169 ms (2.82x) |
| d3-ultra | panel, 11 | **0** | 1.3e-5 | matched 5/5 | 10,697 → 4,090 ms (2.62x) |
| vela3-flash | routed, 40 | **0 of 290 questions, 0 of 79 span sets** | 2.4e-6 | no references | 28,407 → 14,922 ms (1.90x) |
| vela3-turbo | routed, 40 | **0 of 290 questions, 0 of 79 span sets** | 3.0e-6 | no references | 57,862 → 24,464 ms (2.37x) |

A panel answer counts as changed if any field differs: a choice, a level, or a
Noul's side of 0.5. Both kernel sets answered every request identically on its
repeat. Every applied span threshold and input-token count is identical on
vela3-flash and vela3-turbo. The gain grows with size: at these sizes zentorch
answers 1.9 to 2.8 times as fast. vela3-turbo ran with a one-hour forward
timeout (`forward_timeout_seconds`): with current kernels its longest routed
requests exceed the default 300 s, and the runtime then marks the model
degraded.

## Images and video

`tools/decision3_parity.py`'s eight image and video requests (a PNG, a JPEG
and a WebP image, several images, a wide image, a video, two videos, and an
image with a video), sent as `assets` placed before the state, at every size.
These run the vision tower's attention through the `sdpa` slot.

| Size | Answers changed | max \|Δp\| | p50 per request, current → linear + SDPA |
| --- | ---: | ---: | ---: |
| d3-edge | **0 of 8** | 4.6e-6 | 204 → 124 ms (1.64x) |
| d3-lite | **0 of 8** | 2.3e-6 | 445 → 286 ms (1.56x) |
| d3-nano | **0 of 8** | 6.8e-6 | 840 → 431 ms (1.95x) |
| d3-mini | **0 of 8** | 7.6e-6 | 2,116 → 1,064 ms (1.99x) |
| d3-flash | **0 of 8** | 7.4e-6 | 3,522 → 1,596 ms (2.21x) |
| d3 | **0 of 8** | 5.4e-6 | 11,785 → 4,588 ms (2.57x) |
| d3-turbo | **0 of 8** | 5.0e-6 | 8,957 → 3,206 ms (2.79x) |
| d3-ultra | **0 of 8** | 1.2e-5 | 8,921 → 3,258 ms (2.74x) |
| vela3-lite | **0 of 8** | 2.3e-6 | 443 → 288 ms (1.54x) |
| vela3-mini | **0 of 8** | 7.7e-6 | 2,123 → 1,106 ms (1.92x) |
| vela3-flash | **0 of 8** | 7.4e-6 | 3,637 → 1,653 ms (2.20x) |
| vela3-turbo | **0 of 8** | 5.4e-6 | 8,901 → 3,229 ms (2.76x) |

Every image and video request was answered under both kernel sets, identically
on its repeat.

## Why consent per slot

A family's consent on CPUs was one flag for every approximate kernel the CPU
accelerator registers. With a second slot, that flag would have switched
zentorch's SDPA on for vela3-fast, whose record measured the linear only.
`approximate_cpu_kernels` now names the slots a record covers (`True` still
means every slot), so a kernel added to a slot later changes no family's
answers until a record covers it.
