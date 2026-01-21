### MoM Collection Evaluation

This directory contains evaluation scripts for the Multi-lingual Mixture of Models (MoM) collection. 



`mom_collection_eval.py` 

A unified script to evaluate both Text Classification and Token Classification models, including support for LoRA adapters.



###### Requirements

Ensure you have the following installed:



```bash
pip install -r requirements.txt
```

**Usage Examples**

**1. Evaluate a Text Classification Model (Merged):**

```bash
python mom_collection_eval.py --model feedback --device cuda
```

**2. Evaluate a LoRA variant (e.g., Fact Check)**: 

```bash
python mom_collection_eval.py --model fact-check --use_lora
```

**3. Evaluation with a Custom Local Dataset:**

```bash
python mom_collection_eval.py --model intent --custom_dataset ./data/test_data.json
```

**4. Debugging (Limited to 10 samples):**

```bash
python mom_collection_eval.py --model jailbreak --limit 10
```



###### Output

Results are saved to `src/training/model_eval/results/` by default.

- **JSON:** Contains Accuracy, F1, Precision, Recall and Latency stats.

- **PNG:** Confusion Matrix heatmap (for text classfication tasks).
