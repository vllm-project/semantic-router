# CPU: zentorch's FP32 linear for vela3-fast

zentorch (AMD's PyTorch plugin for EPYC, Apache-2.0, on PyPI) lays a linear
layer's FP32 weight out once in ZenDNN's layout and multiplies with it. Its
GEMM sums in another order than `F.linear`, so the CPU accelerator registers it
as an approximate `linear` kernel (`accel/zentorch.py`). It runs only where
`VLLM_SRUN_ZENTORCH` is set, on AMD EPYC Turin onwards (CPU family 1Ah), at `--dtype auto`,
for a family that consents on CPUs (`DtypePolicy.approximate_cpu_kernels`).
vela3-fast consents on the strength of this record; its GPU path stays exact.
Issue: #6022.

- **Date:** 2026-11-16.
- **Device:** a 32-core slice of an AMD EPYC 9755 (Turin): four whole
  CCDs in one NUMA node, SMT off, threads bound one per core
  (`OMP_PROC_BIND=close OMP_PLACES=cores`), tcmalloc preloaded.
- **Stack:** PyTorch 2.14.1+cpu, zentorch 2.14.1.0; the package side of the
  parity tool ran Transformers 5.17.0, the release's pin.
- **Package:** `vllm-sr/vela3-fast` at `7cbefd4b`.
- **Code:** main of 2026-11-16 (`584786913`) with the change that adds this
  record. The oneDNN column forced the `float32-packed` variant through a
  measurement-only switch that is not part of the change.

## Kernels compared

| Column | `linear` slot | Exact |
| --- | --- | --- |
| Current | `F.linear` (the reference) | yes |
| zentorch | `zentorch_linear_unary` on a weight laid out once | no |
| oneDNN | `float32-packed` (`onednn.PackedLinear`), forced as a variant | variant |

vela3-fast's backbone and its nine adapter networks hold their own linear
modules over shared weight tensors; the layout keeps that sharing, so the
model loads with the same parameter count (1,487,053,824 in the engine) and
holds each laid-out weight once.

## Decisions and spans against the package

`tools/vela3_parity.py --generate 200 --seed 0`, every request alone, compares
each answer with the package's own runtime (`Vela3Encoder.system_one`):
Choice arg-max, Set selections, Span sets of (label, start, end), the applied
thresholds and every probability. zentorch ran with `--consented-kernels`, as
`--dtype auto` serves it.

| Kernels | Questions identical | Span sets identical | max \|dp\| | Requests not identical |
| --- | ---: | ---: | ---: | ---: |
| Current | 898 of 915 | 248 of 251 | 7.6e-8 | 10 |
| zentorch | 898 of 915 | 248 of 251 | 5.3e-6 | the same 10 |
| oneDNN `float32-packed` | 898 of 915 | 248 of 251 | 6.0e-6 | the same 10 |

The 915 questions are 483 Choice, 143 Noul, 38 Set and 251 Span, over every
span route the tool generates (PII, unsupported claims, harm and evidence label
sets, relations, named entities, open labels).

- **Against the current kernels:** zentorch leaves each of the 915 questions
  exactly as identical to the package, or not, as the current kernels do; so
  does oneDNN's packed linear.
- **Spans:** 248 span sets have the package's labels and offsets under all three
  kernels. The other three (two named-entity, one PII question) differ from the
  package with every probability equal, under all three kernels: the runtime
  now leaves separators out of its spans (#6559) and reads the PII buckets only
  for PII questions (#6047), where the package does neither.
- **Decisions:** the 14 other questions that differ from the package (11 Choice,
  3 Set) are the same under all three kernels. They are seven requests whose
  context part is long enough that a router row does not fit one input: the
  package then fails every router question of the request, and the family only
  the one that does not fit, as `vela3-parity.md` documents for `g2-0017`.
- **Probabilities:** the largest difference from the package is 5.3e-6, against
  the tool's CPU bar of 1e-4.

## The typed-decisions TEST set, served

All 400 TEST cases (`LocalLLaMA/typed-decisions`, 600 Choice and 600 Noul
questions; vela3-fast answers no Score questions), each request alone at
concurrency 1 through `/v1/systemone`, three warm-up requests first. The three
kernels ran back to back on one host, twice, interleaved; decisions are
compared with the current kernels' first round.

| Kernels | p50 round 1 | p50 round 2 | Speed-up | Decisions changed | Agreement | max \|Δp\| | Golden check |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Current | 451 ms | 450 ms | 1.00x | 0 of 1,200 (round 2) | 100% | 0 | matched 8/8 |
| zentorch | 240 ms | 237 ms | 1.89x | 0 of 1,200 | Choice 600/600, Noul 600/600 | 4.1e-6 | matched 8/8 |
| oneDNN `float32-packed` | 264 ms | 268 ms | 1.69x | 0 of 1,200 | Choice 600/600, Noul 600/600 | 3.0e-6 | matched 8/8 |

- **Speed:** zentorch's linear is 1.89x faster than the reference per request,
  and 1.12x faster than oneDNN's packed linear (median of the two rounds).
  Round to round, every configuration moved by 1.5% or less.
- **Accuracy:** neither layout changed a decision. The largest probability
  shift was 4.1e-6, and the golden check matched all eight answers within the
  1e-3 CPU tolerance. The current kernels' second round was bit-identical to
  their first.

## Why consent, not a variant

oneDNN's packed linear is also within float noise here, but as a variant it
would become vela3-fast's reference forward and change its golden answers.
As an approximate kernel, zentorch's leaves the reference forward and the
goldens as they are, runs only where opted in, and stays off on GPUs, which
keep their own consent (`approximate_kernels`).

## Found on the way

Laying vela3-fast out exposed three things any packed layout meets on a LoRA
model, oneDNN's included; the change fixes them:

- `LoRALinear` took its dtype from `lora_A.weight`, which a packed factor does
  not have; it now takes it from the factor's first tensor.
- The load check's parameter count and the memory estimate counted only
  oneDNN's packed layers.
- Linears of different trees that share a weight were laid out once per tree;
  they are now laid out once per weight.
