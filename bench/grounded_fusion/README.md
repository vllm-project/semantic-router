# Grounding-Aware Fusion Benchmark

This benchmark evaluates the Fusion looper on the
[DRACO](https://huggingface.co/datasets/perplexity-ai/draco) rubric-graded
deep-research dataset. It measures both the quality of the grounding score and
the effect of using that score during synthesis.

DRACO rewards correct, complete answers and applies weighted penalties to
incorrect, unsafe, or poorly sourced claims. Because DRACO does not include
source passages, this benchmark exercises cross-model `panel` grounding rather
than context-grounded factuality.

## What is measured

- **Intrinsic quality:** correlation between each panel response's grounding
  score and its rubric score. This indicates whether the scorer separates
  stronger and weaker responses.
- **Final-answer quality:** paired change in normalized DRACO score after the
  grounding policy is applied. Reports also include negative-criteria penalties
  and bootstrap confidence intervals.

The router supports three grounding policies:

| Policy | Behavior | Use |
| --- | --- | --- |
| `weight` | Keep every response and ask the judge to weight it by groundedness. | Current default. |
| `annotate` | Keep every response and expose groundedness as a note. | Isolate the value of score visibility. |
| `filter` | Drop responses below `min_score`, subject to `min_keep`. | Opt-in hard filtering. |

Historical filter-policy findings are summarized in [FINDINGS.md](FINDINGS.md).
They do not establish whether `weight` or `annotate` improves on plain fusion.

## Evaluation designs

Two runners are available:

1. `run_ab.sh` starts the router separately for grounding-on and grounding-off
   arms. It is useful for checking the deployed request path, but it regenerates
   the panel for each arm and therefore mixes sampling variation with the policy
   effect.
2. `fusioneval` generates one panel per item and reuses its exact bytes across
   every arm. Use this cached-panel design for policy comparisons.

## Prerequisites

- The repository's router and Candle binding built locally.
- A local NLI model, such as `models/mom-halugate-explainer`.
- A DRACO JSON export supplied through `DRACO_PATH` or `--draco-path`.
- Ollama with the panel and judge models used by the command.
- Python 3 with the benchmark dependencies.

The default local model set is:

- panel: `qwen3:8b`, `llama3.1:8b`, and `gemma3:12b`;
- fusion judge and rubric grader: `qwen3:14b`.

Create a Python environment and install the benchmark package:

```bash
python3 -m venv .venv-bench
.venv-bench/bin/python -m pip install -e 'bench[dev]' PyYAML
```

Pull the models required by the default commands:

```bash
ollama pull qwen3:8b
ollama pull llama3.1:8b
ollama pull gemma3:12b
ollama pull qwen3:14b
```

## Router-path smoke test

Start Ollama, then run a small two-arm evaluation:

```bash
DRACO_PATH=/path/to/draco.json \
  bench/grounded_fusion/run_ab.sh \
  --domains Medicine,Law \
  --max-samples 8 \
  --grade-panel
```

The script generates grounding-on and grounding-off configs, starts the
no-thinking Ollama proxy and Envoy when needed, restarts the router for each arm,
and writes reports under `bench/grounded_fusion/results/`. The generated on arm
uses the default `weight` policy. Because panel responses are regenerated per
arm, use this result for integration diagnosis rather than an efficacy claim.

## Cached-panel comparison

The recommended comparison uses four arms:

| Arm | Configuration | Question answered |
| --- | --- | --- |
| `A` | Judge model alone | Does fusion improve on a single model? |
| `B` | Plain fusion | Does the panel improve on the judge alone? |
| `C` | Grounding with `weight` | Does grounded weighting improve plain fusion? |
| `D` | Seeded random weights | Is any improvement specific to the grounding score? |

Optional `annotate` and `filter` arms can be selected with `--arms`.

Build the driver:

```bash
cd src/semantic-router
CGO_LDFLAGS="-L$PWD/../../candle-binding/target/release" \
  go build -o ../../bin/fusioneval ./cmd/fusioneval
cd ../..
```

Prepare items and start the Ollama proxy:

```bash
.venv-bench/bin/python -m bench.grounded_fusion.items \
  --draco-path /path/to/draco.json \
  --domains Medicine,Law \
  --max-samples 100 \
  --out bench/grounded_fusion/results/items.jsonl

.venv-bench/bin/python -m bench.grounded_fusion.ollama_proxy --port 11435
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

Grade each arm with the same rubric model:

```bash
for arm in A B C D; do
  .venv-bench/bin/python -m bench.grounded_fusion.grade_only \
    --answers "bench/grounded_fusion/results/answers_${arm}.jsonl" \
    --arm "$arm" \
    --draco-path /path/to/draco.json \
    --grader-model qwen3:14b \
    --resume
done
```

Then produce the paired verdict:

```bash
.venv-bench/bin/python -m bench.grounded_fusion.compare_multiarm \
  --results-dir bench/grounded_fusion/results \
  --arms A,B,C,D \
  --json-out bench/grounded_fusion/results/verdict.json
```

The comparison uses only sample IDs that are present and error-free in every
arm. Its decision rule evaluates normalized DRACO score with a paired bootstrap
confidence interval:

- `KEEP_GROUNDING`: C beats B and D, and B is no worse than A;
- `KILL_GROUNDING_ADDON`: C does not beat B or D;
- `KILL_FUSION`: A significantly beats B;
- `INCONCLUSIVE`: the available evidence does not satisfy another outcome.

Before interpreting the report, verify that every answer for an item has the
same `panel_sha256`. A mismatch means the arms did not use an identical panel.
Record model revisions, the DRACO revision, source revision, policy parameters,
and hardware alongside any shared result.

## Operational notes

- Panel grounding requires an enabled NLI model. `evaluate.py
  --assert-grounding` stops the router-path run if the on arm lacks a grounding
  trace.
- `make_configs.py` creates a minimal local routing configuration and disables
  external stores that the benchmark does not use.
- `ollama_proxy.py` forwards OpenAI-compatible requests to Ollama's native chat
  endpoint with thinking disabled, avoiding truncated Qwen3 answers.
- Runs are resumable. Keep the results directory when resuming so cached panels
  and previously graded samples can be reused.

## Key files

| File | Purpose |
| --- | --- |
| `datasets.py` | Load DRACO and generic rubric-graded JSONL data. |
| `rubric_judge.py` | Grade answers against weighted rubric criteria. |
| `evaluate.py` | Run and grade one router-path arm. |
| `compare.py` | Compare the two router-path arms. |
| `make_configs.py` | Generate minimal grounding-on and grounding-off configs. |
| `run_ab.sh` | Exercise the deployed router path. |
| `items.py` | Export items for the cached-panel driver. |
| `grade_only.py` | Grade cached-panel answer files. |
| `compare_multiarm.py` | Compare cached-panel arms and write a verdict. |
| `../../src/semantic-router/cmd/fusioneval` | Generate one panel and evaluate multiple arms in process. |
