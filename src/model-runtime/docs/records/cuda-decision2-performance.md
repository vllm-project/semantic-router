# Decision 2.0 vs released runtime on NVIDIA H200

Native CUDA vs the released Decision 2.0 runtime on 400 MMLU-Pro prompts,
5 rounds, two full-panel warmups before timing. Difference is native minus
released, 95% t-interval. Machine readable:
[`cuda-decision2-performance.json`](cuda-decision2-performance.json).

- **Date:** 2026-10-20.
- **Hardware:** NVIDIA H200.
- **Image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` plus the pod
  venv. Not the published `vllm-sr-cuda` image.
- **Head:** `ec307ad6ed081dd220b5dbb3571a1f9987d06914`.
- **Device / profile:** `cuda:0`, `exact`.
- **Kai, Eos, Sol, Nox, Lux:** simultaneous-residency paired rounds.
- **Vega-27B:** sequential alternating loads (one side resident at a time;
  order flips). Timed native graphs: 0 captures, 400 replays, 0 eager.
- **FLA:** Eos / Sol / Nox / Lux / Vega `fla_pinned: true`. Kai is Qwen3
  (no Qwen3.5 FLA pin).

First answers identical on every model. Every p50 / p95 / mean / throughput
interval is **Better**.

| Model | p50 released → native (ms) | p50 95% CI (ms) | FLA pin |
| --- | --- | --- | --- |
| Kai-0.6B | 15.49 → 6.22 | [−9.29, −9.26] | no |
| Eos-0.8B | 29.55 → 6.25 | [−23.31, −23.29] | yes |
| Sol-2B | 29.26 → 7.29 | [−21.99, −21.95] | yes |
| Nox-4B | 39.02 → 12.03 | [−27.05, −26.93] | yes |
| Lux-9B | 40.00 → 15.43 | [−24.75, −24.39] | yes |
| Vega-27B | 116.92 → 54.15 | [−65.90, −59.64] | yes |
