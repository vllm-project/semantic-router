# Decision 1.0 parity on NVIDIA H200

Native CUDA answers vs each Decision 1.0 package's bundled runtime on the
saved MMLU-Pro panel (400 prompts). Machine readable:
[`cuda-decision1-parity.json`](cuda-decision1-parity.json).

- **Date:** 2026-10-16.
- **Hardware:** one NVIDIA H200 (`sm_90`, 143771 MiB). Driver 580.159.04.
- **Image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. Python 3.12.3,
  torch 2.8.0+cu128, Triton 3.4.0, FLA 0.5.2.
- **Head:** `ec307ad6ed081dd220b5dbb3571a1f9987d06914`.
- **Device / profile:** `cuda:0`, `exact`.
- **Panel:** 400 MMLU-Pro requests. This is not the unpublished ROCm panel.

| Model | Identical | Decision changes | Max \|Δ\| |
| --- | ---: | ---: | ---: |
| Kai-0.6B | 400 / 400 | 0 | 0.0 |
| Lex-0.6B | 400 / 400 | 0 | 0.0 |
| Route-0.6B | 400 / 400 | 0 | 0.0 |
| Eos-0.8B | 400 / 400 | 0 | 0.0 |
| Sol-2B | 400 / 400 | 0 | 0.0 |
| Nox-4B | 400 / 400 | 0 | 0.0 |
| Lux-9B | 400 / 400 | 0 | 0.0 |
