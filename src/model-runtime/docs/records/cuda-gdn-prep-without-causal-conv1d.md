# CUDA gated-delta fusion without causal-conv1d

This GB10 experiment compares the proposals in
[issue #5616](https://github.com/vllm-project/semantic-router/issues/5616):

- **A:** install causal-conv1d 1.7.0, mark it approximate and retain the
  registration requirement.
- **B:** leave it uninstalled and register `gdn_prep` when FLA is available.

Both already classify optional causal-conv1d as approximate, so non-consenting
models compare reference against reference. Classification evidence and the
selection change for installed hosts are in the
[direct convolution record](cuda-causal-conv1d-exactness.md).

## Environment and method

- **2026-10-31 (Asia/Taipei):** GB10, sm_121, aarch64, driver 580.173.02;
  PyTorch 2.10.0+cu128, Triton 3.6.0, FLA 0.5.2.
- **Source:** A at `7f3b80438235e9dd8e2c91ccec7b51cfa720cea1`; B removes the
  causal-conv1d registration requirement and uninstalls the package. A's wheel
  capability filter has no effect on sm_121; final B ships no wheel/filter.
- **Both variants:** cuBLAS 12.9.1.4 for
  [#5640](https://github.com/vllm-project/semantic-router/issues/5640), and the
  same test-only MemAvailable placement workaround for GB10 unified memory
  ([#5298](https://github.com/vllm-project/semantic-router/issues/5298)).

Each variant runs in a separate process in the same local CUDA image, using
verified, pinned downloads without executing model-package code.
`Runtime(ServeConfig(...))` uses default `dtype: auto`, as `vllm-srun serve`
does. Golden requests run twice for determinism; every available reference is
compared at GPU tolerance 0.02.

Latency uses `tools/many_questions.py`'s ticket at 1/16/64 questions, answered
alone through `ServedModel.answer_alone` in the released forwards: 5 warmups,
then the median of 30 requests synchronized before/after each call. This is
one sequential A/B pair; small differences do not establish a speedup or
statistical equivalence. The [JSON](cuda-gdn-prep-without-causal-conv1d.json)
preserves model identities, golden outputs, reference verdicts and timing samples.

## Answers and kernel selection

All eight A/B golden outputs are deterministic and bitwise identical:

| Model | A / B fused layers | B vs recorded references at 0.02 |
| --- | ---: | --- |
| Kai-0.6B | 28 / 28 | 3/3 for CPU, sm_90 and ROCm |
| Eos-0.8B | 24 / 24 | 3/3 for CPU, sm_90 and ROCm |
| Sol-2B | 24 / 24 | 3/3 for CPU, sm_90 and ROCm |
| Nox-4B | 32 / 32 | 3/3 for CPU, sm_90 and ROCm |
| Lux-9B | 32 / 32 | 3/3 for CPU, sm_90 and ROCm |
| Vela-2.0-0.8B | 0 / 0 | 7/7 for CPU, sm_90 and ROCm |
| d3-edge | 0 / 0 | 5/5 for CPU, CPU AVX2, sm_89 and ROCm |
| d3-NVFP4 | 64 / 64 | no reference for this revision |

The d3 requests include text and images. NVFP4 is compared directly with A.
GB10 has no qualified sm_121 reference, so readiness remains `unverified`
even where explicit comparisons against other device references pass.

## Latency

Cells show **A → B milliseconds (B change)**:

| Model | 1 question | 16 questions | 64 questions |
| --- | ---: | ---: | ---: |
| Kai-0.6B | 10.67 → 10.46 (-1.91%) | 153.80 → 152.33 (-0.96%) | 599.02 → 596.34 (-0.45%) |
| Eos-0.8B | 15.24 → 15.48 (+1.57%) | 202.90 → 197.01 (-2.90%) | 807.02 → 781.50 (-3.16%) |
| Sol-2B | 27.77 → 27.99 (+0.80%) | 363.26 → 357.84 (-1.49%) | 1422.28 → 1391.16 (-2.19%) |
| d3-NVFP4 | 149.01 → 148.75 (-0.18%) | 2409.45 → 2420.82 (+0.47%) | 9642.27 → 9687.41 (+0.47%) |

All 324 answers are bitwise identical. Both variants retain the same fused
layers, with no clear latency regression for B. FlashInfer is absent; NVFP4
products use cuBLAS. The aborted first NVFP4 B attempt (missing multiprocessing
main guard) is excluded; only its completed retry is used.

These are GB10 golden/ticket results, not a full accuracy evaluation or
qualification of other CUDA architectures.
