# Decision 1.0 performance on NVIDIA H200

Native CUDA vs each Decision 1.0 package's bundled runtime on the same 400
MMLU-Pro prompts, 5 rounds (Lex 10). Difference is native minus bundled,
95% t-interval over rounds (`tools/bench_stats.py`). **Better** / **Worse**
means the whole interval is on one side of zero. Latency negative is
faster. Machine readable: [`cuda-decision1-performance.json`](cuda-decision1-performance.json).

- **Date:** 2026-10-20.
- **Hardware:** NVIDIA H200.
- **Image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` plus the pod
  venv. Not the published `vllm-sr-cuda` image.
- **Head:** `ec307ad6ed081dd220b5dbb3571a1f9987d06914`.
- **Device / profile:** `cuda:0`, `exact`. Metric: end-to-end `Runtime.call`.
- **FLA:** Eos / Sol / Nox / Lux `fla_pinned: true`. Kai / Lex / Route are
  encoders and correctly have no CUDA FLA pin.

All seven end-to-end `call` intervals are **Level or Better**. No row is
wholly worse.

| Model | Rounds | p50 bundled → native (ms) | p50 95% CI (ms) | Verdict |
| --- | ---: | --- | --- | --- |
| Kai-0.6B | 5 | 23.96 → 23.22 | [−1.22, −0.27] | Better |
| Lex-0.6B | 10 | 21.53 → 21.47 | [−1.26, +1.13] | Level |
| Route-0.6B | 5 | 19.33 → 18.48 | [−1.82, +0.11] | Level |
| Eos-0.8B | 5 | 33.11 → 8.05 | [−25.59, −24.53] | Better |
| Sol-2B | 5 | 33.92 → 9.16 | [−25.77, −23.75] | Better |
| Nox-4B | 5 | 45.94 → 14.04 | [−33.27, −30.53] | Better |
| Lux-9B | 5 | 45.12 → 17.66 | [−28.69, −26.22] | Better |

p95, mean and throughput for `call` are Better on every model. This JSON
contains **`call` intervals only**. `direct` and `scheduler` round
summaries remain in the source `.rounds.json` files. Throughput labels in
two older Kai/Lex files had inverted direction; this table uses recomputed
intervals, not those labels.
