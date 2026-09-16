# Reference workload CDFs

Fleet Simulator uses an empirical cumulative distribution function (CDF) of
total request tokens to represent a workload. Each CDF point is
`[token_length, cumulative_fraction]`.

| File | Workload represented |
| --- | --- |
| `azure_cdf.json` | Reference distribution derived from the Azure LLM Inference Trace 2023. |
| `lmsys_cdf.json` | Single-turn LMSYS-Chat-1M conversations. |
| `lmsys_multiturn_cdf.json` | LMSYS-Chat-1M context accumulated across turns. |
| `agent_heavy_cdf.json` | Seeded synthetic mix of software-agent, tool-use, and RAG-style request lengths. |

The synthetic file records its sample count, seed, and provenance in the JSON
object. The trace-derived files contain the CDF array directly.

## Use a custom workload

The loader accepts either a CDF array:

```json
[
  [128, 0.10],
  [512, 0.70],
  [2048, 1.00]
]
```

or an object with the array under `cdf` and optional provenance fields:

```json
{
  "cdf": [[128, 0.10], [512, 0.70], [2048, 1.00]],
  "source": "internal-sample",
  "n_samples": 10000
}
```

Token lengths must be sorted in ascending order. Cumulative fractions must be
nondecreasing, stay between `0` and `1`, and finish at `1`.

Save the file and pass it to a CLI workflow:

```bash
vllm-sr-sim optimize --cdf data/my-workload-cdf.json
```

Run `vllm-sr-sim optimize --help` to override the default arrival rate, SLO,
GPU profiles, context boundary, or simulation settings.
