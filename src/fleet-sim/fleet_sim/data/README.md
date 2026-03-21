# data/

Pre-processed workload CDF files used by the examples and CLI.

Each file is a JSON array of `[token_length, cumulative_fraction]` pairs
representing the empirical CDF of total token counts (input + output) for
that trace.

| File | Source | Description |
|---|---|---|
| `azure_cdf.json` | Azure LLM Inference Trace 2023 | 28K prod requests; p90=4.2K tokens |
| `lmsys_cdf.json` | LMSYS-Chat-1M | Single-turn conversations |
| `lmsys_multiturn_cdf.json` | LMSYS-Chat-1M (multi-turn) | Accumulated context per turn |
| `agent_heavy_cdf.json` | Synthetic agent-heavy | SWE-bench 40% + BFCL 25% + RAG 35% |

## Bring your own CDF

A CDF file is a JSON array of `[token_length, cumulative_fraction]` pairs,
sorted by token length, with cumulative fractions in `[0, 1]`.

```python
import json, numpy as np

# lengths = list of (input_tokens + output_tokens) per request
lengths = sorted(lengths)
n = len(lengths)
cdf = [[int(lengths[i]), (i + 1) / n] for i in range(0, n, max(1, n // 200))]
json.dump(cdf, open("data/my_workload_cdf.json", "w"))
```

Then pass `--cdf data/my_workload_cdf.json` to any CLI command.
