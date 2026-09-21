# Complete benchmark rankings

Scores are 0–100. Mean(TaskType) weights task types equally; Mean(Task) weights tasks equally. Rank and gaps use unrounded values.

| Benchmark | Mean(TaskType) | Global rank | Rank at ≤ size | Gap to best at ≤ size | Mean(Task) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTEB English v2 · 41 tasks | 60.78 | 66/188 | 5/75 | 0.61 pp | 64.88 |
| MAEB audio-only · 19 tasks | 52.34 | 19/64 | 6/27 | 3.51 pp | 43.59 |

Nano uses default shared text; Mini uses fixed official per-task MTEB instructions in the English comparison. Both use default shared audio. Mini's historical default English result (58.79 Mean(TaskType), 63.49 Mean(Task)) remains available separately and is not a matched causal control. One declared mode per Vela model is counted in each ranking population.

The September 17, 2026 registry snapshots contain 186 complete English models and 62 complete audio models; both Vela models are added. The 13 English peers without a known parameter count remain in global ranks only. Vela size counts all modalities; peer sizes are reported totals. Protocols may differ. These panels are not combined into a multimodal rank.

[Data and sources](general-rank-data.json) · [Methodology](pareto-methodology.md) · [Mini text-mode protocol](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/main/benchmarks/instruction-mode.md)
