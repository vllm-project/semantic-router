# CUDA gated-delta fusion without causal-conv1d

The CUDA image ships FLA but not causal-conv1d. Requiring both to register
`gdn_prep` left consenting Qwen3.5 models on the eager path. The Triton prep
implements its convolution itself; on CUDA it is already approximate and
requires family consent. Registration therefore needs FLA, not an installed
causal-conv1d. An optional causal-conv1d installation is also approximate on
CUDA, so non-consenting models keep the reference convolution.

This record compares the two proposals in
[issue #5616](https://github.com/vllm-project/semantic-router/issues/5616):

- **A:** install causal-conv1d 1.7.0, mark it approximate on CUDA and retain
  the existing registration requirement.
- **B:** leave causal-conv1d uninstalled and register `gdn_prep` when FLA is
  available. Retain the approximate classification for optional installations.

## Environment and method

- **Date:** 2026-10-31 (Asia/Taipei).
- **Device:** NVIDIA GB10, sm_121, aarch64, driver 580.173.02.
- **Stack:** PyTorch 2.10.0+cu128, Triton 3.6.0, FLA 0.5.2.
- **Source:** A at `7f3b80438235e9dd8e2c91ccec7b51cfa720cea1`; B is the same
  source with the causal-conv1d registration requirement removed and the
  package uninstalled. A's wheel capability filter has no effect on sm_121;
  the final B implementation omits that filter because it ships no wheel.
- **Shared prerequisites:** both variants use cuBLAS 12.9.1.4 for
  [#5640](https://github.com/vllm-project/semantic-router/issues/5640) and the
  same test-only MemAvailable placement workaround for GB10 unified memory
  ([#5298](https://github.com/vllm-project/semantic-router/issues/5298)). These
  changes are separate from this fix.

Each variant runs in a separate process in the same local CUDA runtime image,
using verified, pinned packages already downloaded. No package code runs.
`Runtime(ServeConfig(...))` loads models as `vllm-srun serve` does, with default
`dtype: auto`. Golden requests run twice for determinism and are compared with
all available recorded references at the GPU tolerance of 0.02.

Latency uses `tools/many_questions.py`'s public ticket at 1, 16 and 64
questions, answered alone through `ServedModel.answer_alone` in the released
forwards. Each shape has 5 warmups and 30 timed runs, with CUDA synchronization
before and after each timed request. The table reports the median. Variants
run sequentially; small differences in one paired run do not establish a
speedup or statistical equivalence.

The [evidence JSON](cuda-gdn-prep-without-causal-conv1d.json) preserves model
revisions and hashes, both variants' golden outputs, reference verdicts,
comparison summaries and every timing sample.

## Answers and kernel selection

| Model | A / B fused layers | A vs B golden outputs | B vs recorded references at 0.02 |
| --- | ---: | --- | --- |
| Kai-0.6B | 28 / 28 | bitwise identical | 3/3 for CPU, sm_90 and ROCm |
| Eos-0.8B | 24 / 24 | bitwise identical | 3/3 for CPU, sm_90 and ROCm |
| Sol-2B | 24 / 24 | bitwise identical | 3/3 for CPU, sm_90 and ROCm |
| Nox-4B | 32 / 32 | bitwise identical | 3/3 for CPU, sm_90 and ROCm |
| Lux-9B | 32 / 32 | bitwise identical | 3/3 for CPU, sm_90 and ROCm |
| Vela-2.0-0.8B | 0 / 0 | bitwise identical | 7/7 for CPU, sm_90 and ROCm |
| d3-edge | 0 / 0 | bitwise identical | 5/5 for CPU, CPU AVX2, sm_89 and ROCm |
| d3-NVFP4 | 64 / 64 | bitwise identical | no reference for this revision |

All golden outputs are deterministic. Vela 2.0 and d3-edge retain exact kernel
selection, including the reference convolution. The d3 golden requests include
both text and image inputs. d3-NVFP4 is compared directly with A; its revision
has no recorded golden reference. GB10 has no qualified sm_121 reference, so
runtime readiness remains `unverified` even where the explicit comparisons
against other device references pass.

## Latency

| Model | Questions | A (ms) | B (ms) | B change |
| --- | ---: | ---: | ---: | ---: |
| Kai-0.6B | 1 | 10.67 | 10.46 | -1.91% |
| Kai-0.6B | 16 | 153.80 | 152.33 | -0.96% |
| Kai-0.6B | 64 | 599.02 | 596.34 | -0.45% |
| Eos-0.8B | 1 | 15.24 | 15.48 | +1.57% |
| Eos-0.8B | 16 | 202.90 | 197.01 | -2.90% |
| Eos-0.8B | 64 | 807.02 | 781.50 | -3.16% |
| Sol-2B | 1 | 27.77 | 27.99 | +0.80% |
| Sol-2B | 16 | 363.26 | 357.84 | -1.49% |
| Sol-2B | 64 | 1422.28 | 1391.16 | -2.19% |
| d3-NVFP4 | 1 | 149.01 | 148.75 | -0.18% |
| d3-NVFP4 | 16 | 2409.45 | 2420.82 | +0.47% |
| d3-NVFP4 | 64 | 9642.27 | 9687.41 | +0.47% |

All 324 answers (four models, 81 questions each) are bitwise identical between
A and B. Both variants retain the same fused layers, with no clear latency
regression for B on this device. FlashInfer is not installed; NVFP4 products
use cuBLAS in both variants.

This is a GB10 golden and ticket comparison, not a full accuracy evaluation or
qualification of other CUDA architectures. The first d3-NVFP4 B attempt was
aborted because the probe lacked a multiprocessing main guard; only the
corrected, completed retry contributes to these results.
