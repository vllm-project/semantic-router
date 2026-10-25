# Vela 1.0 legacy path on NVIDIA H200

CUDA runtime vs the legacy router path (candle CPU facade) on the 11 text
jobs `tools/legacy_parity.py` compares. Thresholds: probability 0.02, near
tie 0.001, label 0.995, spans 0.98. Machine readable:
[`cuda-vela1-parity.json`](cuda-vela1-parity.json).

- **Date:** 2026-10-16.
- **Hardware:** NVIDIA H200 (`sm_90`, 143771 MiB). Driver 580.159.04.
- **Image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. Python 3.12.3,
  torch 2.8.0+cu128, Triton 3.4.0, FLA 0.5.2.
- **Head:** `ec307ad6ed081dd220b5dbb3571a1f9987d06914`.
- **Legacy:** native facade at `61aa7eb2d`, CPU recipe (candle).
- **Device / profile:** runtime `cuda:0`, `exact`.
- **Inputs sha256:** `568ff59b76ef3b6c61cefb5148c73ecb176bd62c3820c93adfc1fbab860e9872`.

All 11 jobs **passed** with agreement **1.0**. This is answer parity, not a
repeated-round performance A/B.

| Job | Compared | Both rejected | Max \|Δp\| | CUDA p50 (ms) | Legacy p50 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| domain | 553 | 0 | 3.34e-4 | 9.29 | 132.60 |
| guard | 550 | 3 | 3.02e-5 | 3.77 | 86.86 |
| safety | 553 | 0 | 3.42e-5 | 3.05 | 109.13 |
| shield | 553 | 0 | 4.64e-5 | 3.08 | 160.66 |
| factcheck | 553 | 0 | 3.54e-5 | 3.09 | 108.00 |
| feedback | 553 | 0 | 8.76e-5 | 4.18 | 91.76 |
| modality | 553 | 0 | 9.95e-6 | 4.61 | 87.23 |
| hazard | 553 | 0 | 9.00e-6 | 2.94 | 87.29 |
| pii | 553 | 0 | 1.33e-4 | 7.83 | 154.90 |
| pii_truncate | 553 | 0 | 6.34e-5 | 5.32 | 204.50 |
| halu | 64 | 0 | 3.77e-5 | 32.60 | 4268.24 |

Per-job latency and runtime-only load windows are in the JSON. Legacy
throughput is `null`. This H200 file is answer parity only. The later
five-round candle-vs-CUDA performance A/B is
[`cuda-vela1-performance.md`](cuda-vela1-performance.md) on NVIDIA A100.

Embedding / rerank Vela 1.0 jobs are outside this record.
