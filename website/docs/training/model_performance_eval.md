# Model Performance Evaluation
## Why evaluate?
Evaluation makes routing data-driven. By measuring per-category accuracy on MMLU-Pro (and doing a quick sanity check with ARC), you can:

- Select the right model for each category and rank them into categories.model_scores
- Pick a sensible default_model based on overall performance
- Decide when CoT prompting is worth the latency/cost tradeoff
- Catch regressions when models, prompts, or parameters change
- Keep changes reproducible and auditable for CI and releases

In short, evaluation converts anecdotes into measurable signals that improve quality, cost efficiency, and reliability of the router.

---

This guide documents the automated workflow to evaluate models (MMLU-Pro and ARC Challenge) via a vLLM-compatible OpenAI endpoint, generate a performance-based routing config, and update categories.model_scores in config.

see code in [/src/training/model_eval](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_eval)

### What you'll run end-to-end
#### 1) Evaluate models: 

- per-category accuracies
- ARC Challenge: overall accuracy
  
#### 2) Visualize results

- bar/heatmap plot of per-category accuracies

**TODO**  a picture needed
#### 3) Generate an updated config.yaml:

- Rank models per category into categories.model_scores
- Set default_model to the best average performer
- Keep or apply category-level reasioning settings

## 1.Prerequisites

- A running vLLM-compatible OpenAI endpoint serving your models
  - Endpoint URL like http://localhost:8000/v1
  - Optional API key if your endpoint requires one
- Python packages for evaluation scripts:
  - From the repo root: matplotlib
  - From `/src/training/model_eval`: [requirements.txt](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_eval/requirements.txt)

  ```bash
  cd /src/training/model_eval
  pip install -r requirements.txt
  ```

**Optional tip:**

- Ensure your `config/config.yaml` includes your deployed model names under `vllm_endpoints[].models` and any pricing/policy under `model_config` if you plan to use the generated config directly.

## 2.Evaluate on MMLU-Pro
see script in [mmul_pro_vllm_eval.py](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_eval/mmlu_pro_vllm_eval.py)

### Example usage patterns:

```bash
# Evaluate a few models, few samples per category, direct prompting
python mmlu_pro_vllm_eval.py \
  --endpoint http://localhost:8000/v1 \
  --models gemma3:27b phi4 mistral-small3.1 \
  --samples-per-category 10

# Evaluate with CoT (results saved under *_cot)
python mmlu_pro_vllm_eval.py \
  --endpoint http://localhost:8000/v1 \
  --models gemma3:27b phi4 mistral-small3.1 \
  --samples-per-category 10
  --use-cot 
```

### Key flags:

- **--endpoint**: vLLM OpenAI URL (default http://localhost:8000/v1)
- **--models**: space-separated list OR a single comma-separated string; if omitted, the script queries /models from the endpoint
- **--categories**: restrict evaluation to specific categories; if omitted, uses all categories in the dataset
- **--samples-per-category**: limit questions per category (useful for quick runs)
- **--use-cot**: enables Chain-of-Thought prompting variant; results are saved in a separate subfolder suffix (_cot vs _direct)
- **--concurrent-requests**: concurrency for throughput
- **--output-dir**: where results are saved (default results)
- **--max-tokens**, **--temperature**, **--seed**: generation and reproducibility knobs

### What it outputs per model:

- **results/<model_name>_(direct|cot)/**
  - **detailed_results.csv**: one row per question with is_correct and category
  - **analysis.json**: overall_accuracy, category_accuracy map, avg_response_time, counts
  - **summary.json**: condensed metrics
- **mmlu_pro_vllm_eval.txt**: prompts and answers log (debug/inspection)

### Notes:

- Model naming: slashes are replaced with underscores for folder names; e.g., gemma3:27b -> gemma3:27b_direct directory.
- Category accuracy is computed on successful queries only; failed requests are excluded.

## 3.Evaluate on ARC Challenge (optional, overall sanity check)
see script in [arc_challenge_vllm_eval.py](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_eval/arc_challenge_vllm_eval.py)

### Example usage patterns:

``` bash
python arc_challenge_vllm_eval.py \
  --endpoint http://localhost:8000/v1\
  --models gemma3:27b,phi4:latest
```

### Key flags:

- **--samples**: total questions to sample (default 20); ARC is not categorized in our script
- Other flags mirror the MMLU-Pro script

### What it outputs per model:

- **results/<model_name>_(direct|cot)/**
  - **detailed_results.csv**: one row per question with is_correct and category
  - **analysis.json**: overall_accuracy, avg_response_time
  - **summary.json**: condensed metrics
- **arc_challenge_vllm_eval.txt**: prompts and answers log (debug/inspection)

### Note:
ARC results do not feed categories[].model_scores directly, but they can help spot regressions.

## 4.Visualize per-category performance
see script in [plot_category_accuracies.py](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_eval/plot_category_accuracies.py)

### Example usage patterns:

```bash
# Use results/ to generate bar plot
python src/training/model_eval/plot_category_accuracies.py \
  --results-dir results \
  --plot-type bar \
  --output-file model_eval/category_accuracies.png

# Use results/ to generate heatmap plot
python src/training/model_eval/plot_category_accuracies.py \
  --results-dir results \
  --plot-type heatmap \
  --output-file model_eval/category_accuracies.png

# Use sample-data to generate example plot
python src/training/model_eval/plot_category_accuracies.py \
  --sample-data \
  --plot-type heatmap \
  --output-file model_eval/category_accuracies.png
```

### Key flags:

- **--results-dir**: where analysis.json files are
- **--plot-type**: bar or heatmap
- **--output-file**: output image path (default model_eval/category_accuracies.png)
- **--sample-data**: if no results exist, generates fake data to preview the plot

### What it does:

- Finds all results/**/analysis.json, aggregates analysis["category_accuracy"] per model
- Adds an Overall column representing the average across categories
- Produces a figure to quickly compare model/category performance

### Note:

- It merges “direct” and “cot” as distinct model variants by appending :direct or :cot to the label; the legend hides “:direct” for brevity.

## 5.Generate performance-based routing config
see script in [result_to_config.py](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_eval/result_to_config.py)

### Example usage patterns:

```bash
# Use results/ to generate a new config file (not overridded)
python src/training/model_eval/result_to_config.py \
  --results-dir results \
  --output-file config/config.eval.yaml

# Modify similarity-thredshold
python src/training/model_eval/result_to_config.py \
  --results-dir results \
  --output-file config/config.eval.yaml \
  --similarity-threshold 0.85

# Generate from specific folder
python src/training/model_eval/result_to_config.py \
  --results-dir results/mmlu_run_2025_09_10 \
  --output-file config/config.eval.yaml
```

### Key flags:

- **--results-dir**: points to the folder where analysis.json files live
- **--output-file**: target config path (default config/config.yaml)
- **--similarity-threshold**: semantic cache threshold to set in the generated config

### What it does:

- Reads all analysis.json files, extracting analysis["category_accuracy"]
- Constructs a new config:
  - default_model: the best average performer across categories
  - categories: For each category present in results, ranks models by accuracy:
    - category.model_scores = [{model: "<name>", score: <float>}, ...], highest first
  - category reasoning settings: auto-filled from a built-in mapping (math, physics, chemistry, CS, engineering -> high reasoning; others default to low/medium; you can adjust after generation)
  - Leaves out any special “auto” placeholder models if present

### Schema alignment:

- **categories[].name**: the MMLU-Pro category string
- **categories[].model_scores**: descending ranking by accuracy for that category
- **default_model**: a top performer across categories (approach suffix removed, e.g., gemma3:27b from gemma3:27b:direct)
- Keeps other config sections (semantic_cache, tools, classifier, prompt_guard) with reasonable defaults; you can edit them post-generation if your environment differs

### Note:

- Existing config.yaml can be overwritten. Consider writing to a temp file first and diffing:
  - --output-file config/config.eval.yaml
- If your production config.yaml carries environment-specific settings (endpoints, pricing, policies), port the evaluated categories[].model_scores and default_model back into your canonical config.
