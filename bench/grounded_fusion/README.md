# Grounding-Aware Fusion Benchmark

`fusioneval` compares the Fusion looper's grounding policies on a cached panel.
It generates the panel responses once per item and synthesizes every arm from
the same panel bytes, so differences between arms come from the policy rather
than from regenerated panel responses.

The router supports three grounding policies:

| Policy | Behavior | Use |
| --- | --- | --- |
| `weight` | Keep every response and ask the judge to weight it by groundedness. | Current default. |
| `annotate` | Keep every response and expose groundedness as a note. | Isolate the value of score visibility. |
| `filter` | Drop responses below `min_score`, subject to `min_keep`. | Opt-in hard filtering. |

Historical filter-policy findings are summarized in [FINDINGS.md](FINDINGS.md).
They do not establish whether `weight` or `annotate` improves on plain fusion.
The Python DRACO exporter and graders, and the router-path A/B scripts that
produced those findings, are no longer in the tree; FINDINGS.md names the
commit that holds them.

## Arms

| Arm | Configuration | Question answered |
| --- | --- | --- |
| `A` | Judge model alone | Does fusion improve on a single model? |
| `B` | Plain fusion | Does the panel improve on the judge alone? |
| `C` | Grounding with `weight` | Does grounded weighting improve plain fusion? |
| `D` | Seeded random weights | Is any improvement specific to the grounding score? |

Optional `annotate` and `filter` arms can be selected with `--arms`.

## Prerequisites

- The repository's router and Candle binding built locally.
- A local NLI model, such as `models/mom-halugate-explainer`.
- An OpenAI-compatible chat endpoint for the panel and judge models, such as
  Ollama behind `ollama_proxy.py`.
- An items file with one JSON object per line: `id`, `domain`, `question` and an
  optional `context`.

The default model set is `qwen3:8b`, `llama3.1:8b`, and `gemma3:12b` for the
panel and `qwen3:14b` for the judge.

## Run

Build the driver and start the Ollama proxy. The proxy forwards
OpenAI-compatible requests to Ollama's native chat endpoint with thinking
disabled, avoiding truncated Qwen3 answers.

```bash
make build-fusioneval
python3 -m bench.grounded_fusion.ollama_proxy --port 11435
```

In another shell, generate a four-item smoke run before increasing the sample
count:

```bash
LD_LIBRARY_PATH=candle-binding/target/release bin/fusioneval \
  --items bench/grounded_fusion/results/items.jsonl \
  --nli-model models/mom-halugate-explainer \
  --endpoint http://localhost:11435/v1/chat/completions \
  --judge qwen3:14b \
  --panel qwen3:8b,llama3.1:8b,gemma3:12b \
  --arms A,B,C,D \
  --out-dir bench/grounded_fusion/results \
  --max-items 4
```

The driver writes `panel_cache.jsonl` and one ungraded `answers_{arm}.jsonl`
per arm. Runs are resumable; keep the results directory to reuse cached panels.

Before comparing arms, verify that every answer for an item has the same
`panel_sha256`. A mismatch means the arms did not use an identical panel.
Record model revisions, the dataset revision, source revision, policy
parameters, and hardware alongside any shared result.
