---
title: Model Performance Evaluation
sidebar_label: Model Evaluation
---

# Model Performance Evaluation

Evaluate candidate generative models before assigning them to routing
decisions. The repository's `model_eval` scripts measure multiple-choice
accuracy through an OpenAI-compatible endpoint, plot per-category MMLU-Pro
results, and turn those results into a canonical configuration scaffold.

This workflow answers which evaluated model performed best on the selected
dataset and prompt mode. It does not prove that the same ranking will hold for
production traffic.

## What the workflow produces

| Step | Output | Use |
|------|--------|-----|
| MMLU-Pro evaluation | Per-question CSV plus `analysis.json` and `summary.json` | Compare models by category and overall accuracy |
| ARC Challenge evaluation | Per-question CSV plus overall analysis | Independent multiple-choice sanity check |
| Plotting | Bar chart or heatmap | Inspect category-level differences |
| Config generation | `config.eval.yaml` scaffold | Seed provider bindings, model cards, and domain scores for review |

Only MMLU-Pro results feed `result_to_config.py` because ARC output does not
contain the domain categories used by the generator.

## Prerequisites

- One or more OpenAI-compatible endpoints serving the models to compare
- Served model IDs that match the values passed through `--models`
- Network and dataset access for the Hugging Face datasets used by the scripts
- Enough provider capacity and budget to send every selected question to each
  candidate model

Create an isolated environment from the repository root:

```bash
cd src/training/model_eval
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The scripts save prompts, model responses, correctness labels, and timing data.
Choose an output directory with retention and access controls appropriate for
the evaluated content.

## Run MMLU-Pro

Start with a small sample to verify model IDs and response formatting:

```bash
python mmlu_pro_vllm_eval.py \
  --endpoint http://localhost:8000/v1 \
  --models phi4 qwen3-0.6B \
  --samples-per-category 10 \
  --output-dir results/mmlu-smoke
```

Important options:

- `--models` accepts space-separated model IDs or one comma-separated value.
  When omitted, the script queries the endpoint's `/models` API.
- `--categories` limits the MMLU-Pro categories.
- `--samples-per-category` controls the sample count in each selected category;
  its default is `5`, so set it explicitly for a formal run.
- `--use-cot` creates a separate `_cot` result directory for a
  chain-of-thought prompt variant.
- `--concurrent-requests` increases parallel requests. Begin at `1` so rate
  limits and queueing do not silently distort the comparison.
- `--temperature` defaults to `0.0` and `--seed` to `42`.

Each model and prompt mode gets a directory such as
`results/mmlu-smoke/phi4_direct/` containing:

- `detailed_results.csv`
- `analysis.json`
- `summary.json`

Accuracy is computed over successful requests. Always report
`successful_queries` and `failed_queries` with the accuracy; excluding failures
without disclosing them can make an unreliable endpoint look better.

## Run ARC Challenge

Use ARC as a second dataset, not as a source for domain scores:

```bash
python arc_challenge_vllm_eval.py \
  --endpoint http://localhost:8000/v1 \
  --models phi4 qwen3-0.6B \
  --samples 100 \
  --output-dir results/arc
```

`--samples` is a total sample count and defaults to `20`. The other generation,
concurrency, model, and prompt-mode options mirror the MMLU-Pro script.

## Plot category results

The plotter reads MMLU-Pro `analysis.json` files recursively:

```bash
python plot_category_accuracies.py \
  --results-dir results/mmlu-smoke \
  --plot-type heatmap \
  --output-file results/mmlu-smoke/category-accuracy.png
```

Use `--plot-type bar` for grouped bars. `--sample-data` renders synthetic data
only to preview the chart layout; never publish that output as an evaluation
result.

## Generate a configuration scaffold

```bash
python result_to_config.py \
  --results-dir results/mmlu-smoke \
  --output-file config.eval.yaml \
  --endpoint http://127.0.0.1:8000/v1 \
  --provider openai-compatible
```

The generator creates a human-readable v0.3 scaffold with:

- one provider binding, structured invocation control, and connection-free
  Model card per evaluated model
- one domain signal and decision per observed MMLU-Pro category
- each decision assigned to its best evaluated Model
- a default decision assigned to the best model overall
- sparse defaults for the classifier services used by the generated Recipe

The generated file has this top-level shape. Lists and nested bodies are
abbreviated here; inspect `config.eval.yaml` for the evaluated models, scores,
category signals, decisions, and assignments.

```yaml
version: v0.3
listeners: []
providers:
  models: []
routing:
  modelCards: []
recipes: []
entrypoints: []
global: {}
```

Direct and CoT result directories for the same base model are collapsed into
one logical Model. For each category, the generator keeps the higher observed
accuracy and assigns the best evaluated Model. Review those assignments before
publishing them as routing policy.

The generated backend address and invocation control are applied to every model
unless you override them. Review the endpoint topology, credentials, retry and
timeout policy, and pricing for each provider.

## Turn the scaffold into a routing policy

`config.eval.yaml` is intentionally incomplete:

- `listeners` is empty.
- Model connections use command-line defaults and conservative scaffold control
  values.
- evaluation categories may not match your user-facing decisions.
- the sparse `global` section may not match your runtime or security policy.

Review the generated Models, category decisions, and assignments, add a
listener, then validate the result:

```bash
vllm-sr validate --config config.yaml
```

Do not replace a production configuration wholesale. Preserve its listeners,
secrets, provider-specific endpoints, retry and health policies, pricing,
services, and storage settings.

## Evaluate the routing outcome

Use a held-out dataset or production-representative replay that was not used to
choose the models, prompt mode, category mapping, or thresholds. Report:

- quality by category and important workload slice
- request failures and excluded samples
- selected-model distribution after decisions are added
- end-to-end latency, token usage, and provider cost
- comparison with the default model and best single-model baseline
- dataset revision, model revisions, source commit, configuration, and command

The MMLU-Pro generator ranks evaluated answers; it does not test the complete
Router data path. Run an end-to-end benchmark after integrating the scaffold.
See [Benchmarking](../benchmarking/overview) for the available suites.

## Source

- [`src/training/model_eval`](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_eval)
- [Training Router Models](./training-overview)
- [ML-Based Model Selection](./ml-model-selection)
