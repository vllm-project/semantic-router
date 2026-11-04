# Vela 1.0 performance vs CPU legacy on NVIDIA A100

Native CUDA vs the in-tree CPU Candle / ONNX binary (`legacy_parity.py
ab`), five pair rounds then five 20 s load windows (4 callers,
`--profile batching`, 16 threads). Latency ratios are legacy / runtime
(above 1 is faster). Throughput ratios are runtime / legacy calls per
second, geometric mean with a 95% t-interval over the five windows. Every
interval is wholly above 1 (**Better**). Machine readable:
[`cuda-vela1-performance.json`](cuda-vela1-performance.json).

- **Date:** 2026-10-21 (run ended 2026-10-21T03:38:46Z).
- **Hardware:** NVIDIA A100.
- **Image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` plus the pod
  venv. Not the published `vllm-sr-cuda` image.
- **Head:** `ec307ad6ed081dd220b5dbb3571a1f9987d06914`. Legacy binary from
  `61aa7eb2db0cfd9f00bfa06abd4944b3864cf38d`.
- **Device / threads:** `cuda:0`; Candle/ONNX `OMP_NUM_THREADS=16` (also
  MKL, OPENBLAS, RAYON).

| Job | Pairs | p50 legacy → runtime (ms) [legacy / runtime, 95% CI] | p95 [legacy / runtime, 95% CI] | Median per-input speedup (95% CI) | 4 callers, calls/s: legacy → `exact` | `exact` / legacy (95% CI) |
| --- | ---: | --- | --- | --- | --- | --- |
| domain | 2,765 | 98.19 → 5.93 [16.33, 16.92] | 655.31 → 10.64 [55.73, 66.10] | 17.67× [17.31, 18.16] | 19.5 → 230.8 | 11.80× [11.25, 12.37] |
| guard | 2,750 | 103.18 → 5.94 [17.05, 17.74] | 650.18 → 10.53 [58.32, 66.45] | 18.31× [17.91, 18.68] | 16.5 → 191.3 | 11.62× [11.01, 12.27] |
| safety | 2,765 | 98.86 → 5.26 [18.48, 19.07] | 653.49 → 9.86 [63.37, 69.22] | 19.94× [19.68, 20.45] | 18.8 → 235.6 | 12.16× [11.67, 12.67] |
| shield | 2,765 | 97.86 → 5.12 [18.79, 19.49] | 635.35 → 9.57 [62.74, 70.78] | 20.18× [19.74, 20.58] | 19.0 → 238.8 | 12.52× [11.96, 13.11] |
| factcheck | 2,765 | 98.98 → 5.30 [18.39, 19.14] | 629.92 → 9.69 [61.34, 70.00] | 20.06× [19.73, 20.39] | 18.7 → 235.2 | 12.74× [11.95, 13.58] |
| feedback | 2,765 | 99.49 → 5.81 [16.82, 17.48] | 658.25 → 10.30 [61.53, 67.81] | 18.12× [17.78, 18.49] | 19.7 → 231.6 | 11.67× [11.32, 12.03] |
| modality | 2,765 | 98.05 → 5.33 [17.92, 18.71] | 626.66 → 9.95 [59.88, 66.75] | 19.14× [18.75, 19.51] | 18.9 → 235.5 | 12.51× [11.92, 13.12] |
| hazard | 2,765 | 98.35 → 5.35 [17.98, 18.80] | 656.11 → 10.05 [62.40, 70.90] | 19.58× [19.26, 20.03] | 17.3 → 142.7 | 8.19× [7.89, 8.50] |
| pii | 2,765 | 98.85 → 6.23 [15.56, 16.24] | 640.50 → 11.00 [54.64, 61.04] | 16.79× [16.46, 17.13] | 16.4 → 136.7 | 8.20× [7.73, 8.70] |
| pii_truncate | 2,765 | 198.28 → 6.13 [32.05, 32.92] | 713.51 → 11.03 [60.64, 70.27] | 33.93× [33.48, 34.44] | 12.6 → 218.9 | 17.47× [16.41, 18.59] |
| halu | 320 | 4674.19 → 31.15 [129.46, 165.43] | 114904.40 → 251.96 [172.91, 471.62] | 157.35× [147.64, 163.53] | 0.6 → 26.0 | 44.40× [41.81, 47.15] |

Batching windows are also Better on every job (geometric means 9.45× to
46.82×). Source: `cuda-vs-legacy.ab.json`, `pair_rounds` 5 and
`window_rounds` 5, exit 0. The earlier H200 file
[`cuda-vela1-parity.md`](cuda-vela1-parity.md) is answer parity, not this
A/B.
