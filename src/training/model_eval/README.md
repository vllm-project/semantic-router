# MoM Collection Evaluation

This directory contains evaluation scripts for the Multi-lingual Mixture of Models (MoM) collection.

## mom_collection_eval.py

A unified script to evaluate both Text Classification and Token Classification models, including support for LoRA adapters and local fine-tuned models.

### Requirements

Ensure you have the following installed:

```
pip install -r requirements.txt
```

### Usage Examples

#### 1. Evaluate a Text Classification Model (Merged)

```
python src/training/model_eval/mom_collection_eval.py --model feedback --device cuda
```

#### 2. Evaluate a LoRA variant (e.g., Fact Check)

```
python src/training/model_eval/mom_collection_eval.py --model fact-check --use_lora
```

#### 3. Evaluate a Local Fine-Tuned Model

Use `--model_id` to point to a local directory or a specific Hugging Face repository.

```
python src/training/model_eval/mom_collection_eval.py --model feedback --model_id ./my-local-checkpoint
```

#### 4. Performance Tuning (Batch Size)

Use `--batch_size` to speed up inference on GPU (default is 16).

```
python src/training/model_eval/mom_collection_eval.py --model pii --batch_size 64 --device cuda
```

#### 5. Evaluation with a Custom Local Dataset

```
python src/training/model_eval/mom_collection_eval.py --model intent --custom_dataset ./data/test_data.json
```

#### 6. Debugging (Limited to 10 samples)

```
python src/training/model_eval/mom_collection_eval.py --model jailbreak --limit 10
```

##### Output

Results are saved to `src/training/model_eval/results/` by default.

- **JSON:** Contains Accuracy, F1, Precision, Recall and Latency stats.

- **PNG:** Confusion Matrix heatmap (for text classification tasks).
