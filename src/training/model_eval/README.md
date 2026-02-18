# MoM Collection Evaluation

**Multi-lingual Mixture of Models (MoM) Evaluation Script**

A clean, unified Python script to evaluate the **10 multi-lingual Mixture of Models (MoM)**  
from the `llm-semantic-router` collection on Hugging Face , both merged models and LoRA adapters.

Supports:

- **Text Classification** (feedback, jailbreak, fact-check, intent)
- **Token Classification** (PII detection)
- Merged models + LoRA variants
- Custom datasets, language filtering, batch size control, parallel evaluation and more !!!

## Features

- Evaluate **all 10 models** with one script
- Comprehensive metrics: **Accuracy, Precision, Recall, F1, Confusion Matrix, Latency (avg / p50 / p99)**
- Special handling for **MMLU-Pro** (intent) and **Presidio** (PII) datasets
- Works on **GPU** and **CPU**
- Saves results as **JSON** + **confusion matrix PNG** (for text classification)
- Robust error handling (missing columns, OOM, network issues, etc.)
- Supports **custom local datasets** (.json / .csv)
- Language filtering for multilingual evaluation

## Models Supported

| Model Name | Task Type            | Merged Model ID                                        | LoRA Model ID                                        |
| ---------- | -------------------- | ------------------------------------------------------ | ---------------------------------------------------- |
| feedback   | Text Classification  | `llm-semantic-router/mmbert-feedback-detector-merged`  | `llm-semantic-router/mmbert-feedback-detector-lora`  |
| jailbreak  | Text Classification  | `llm-semantic-router/mmbert-jailbreak-detector-merged` | `llm-semantic-router/mmbert-jailbreak-detector-lora` |
| fact-check | Text Classification  | `llm-semantic-router/mmbert-fact-check-merged`         | `llm-semantic-router/mmbert-fact-check-lora`         |
| intent     | Text Classification  | `llm-semantic-router/mmbert-intent-classifier-merged`  | `llm-semantic-router/mmbert-intent-classifier-lora`  |
| pii        | Token Classification | `llm-semantic-router/mmbert-pii-detector-merged`       | `llm-semantic-router/mmbert-pii-detector-lora`       |

## Installation

1. Clone the repository:

```bash
git clone https://github.com/vllm-project/semantic-router.git
cd semantic-routegit clone https://github.com/vllm-project/semantic-router.git
cd semantic-route
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage Examples

**1. Evaluate one model (GPU recommended)**

```bash
python src/training/model_eval/mom_collection_eval.py --model feedback --device cuda
```

**2. Evaluate LoRA version**

```bash
python src/training/model_eval/mom_collection_eval.py --model fact-check --use_lora --device cuda
```

**3. Evaluate multiple models at once**

```bash
python src/training/model_eval/mom_collection_eval.py --model feedback jailbreak fact-check intent pii --device cuda
```

**4. Quick test with few sample**

```bash
python src/training/model_eval/mom_collection_eval.py --model pii --limit 100 --device cpu
```

**5. Use a custom local dataset**

```bash
python src/training/model_eval/mom_collection_eval.py --model intent --custom_dataset ./my_test_data.json
```

**6.  Run models in parallel**

```bash
python src/training/model_eval/mom_collection_eval.py --model feedback jailbreak intent --parallel --device cuda
```

**7.  Smaller batch size**

```bash
python src/training/model_eval/mom_collection_eval.py --model pii --batch_size 8 --device cuda
```

**8. Filter by language (multilingual evaluation)**

```bash
python src/training/model_eval/mom_collection_eval.py --model feedback --language es --device cuda
```

### Output

Results are saved in:

```context
src/training/model_eval/results/
```

You will get files like:

- `feedback_results.json`
- `feedback_cm.png` (confusion matrix heatmap which is only for text classification)
- ...and similar files for each evaluated model

#### Common Commands

<style type="text/css"></style>

| Goal                            | Command Example                 |
| ------------------------------- | ------------------------------- |
| Quick test on CPU               | `--limit 50 --device cpu`       |
| Fast evaluation on GPU          | `--device cuda --batch_size 64` |
| Single model                    | `--model jailbreak`             |
| Use LoRA instead of merged      | `--use_lora`                    |
| Custom dataset                  | `--custom_dataset ./test.json`  |
| Run multiple models in parallel | `--parallel`                    |
| Evaluate only English samples   | `--language en`                 |
| Debug with very few samples     | `--limit 10`                    |
