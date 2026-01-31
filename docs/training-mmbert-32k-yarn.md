# Training guide: mmbert-32k-yarn base model (LoRA vs full fine-tune)

This document explains how to train the classifiers in this repo using the `llm-semantic-router/mmbert-32k-yarn` base model.

It covers both:
- **LoRA fine-tuning** (adapter-only, optionally merge)
- **Full fine-tuning** (non-LoRA)

> Note: The exact scripts and arguments differ slightly between the LoRA training scripts under `src/training/training_lora/` and the feedback detector pipeline under `src/training/modernbert_dissat_pipeline/`.

---

## 0) Prerequisites

- Python 3
- GPU environment (ROCm or CUDA). The example `train_mmbert_gpu.sh` is designed for GPU.
- Install dependencies (per script/pipeline):
  - LoRA scripts: `peft`, `accelerate`, `datasets`, `transformers`, `scikit-learn`, `tqdm`
  - Feedback pipeline: see `src/training/modernbert_dissat_pipeline/requirements.txt`

---

## 1) Base model selection

The base model can be selected via:
- `mmbert-32k-yarn` (short name)
- `llm-semantic-router/mmbert-32k-yarn` (full HF id)

The mapping is defined in:
- `src/training/training_lora/common_lora_utils.py`

---

## 2) LoRA training (recommended for iteration)

### 2.1 Train LoRA adapters for the 4 router classifiers

Scripts:
- Intent: `src/training/training_lora/classifier_model_fine_tuning_lora/ft_linear_lora.py`
- Fact-check: `src/training/training_lora/fact_check_fine_tuning_lora/fact_check_bert_finetuning_lora.py`
- PII: `src/training/training_lora/pii_model_fine_tuning_lora/pii_bert_finetuning_lora.py`
- Jailbreak: `src/training/training_lora/prompt_guard_fine_tuning_lora/jailbreak_bert_finetuning_lora.py`

Quick start (runs all 4 tasks):

```bash
cd src/training/training_lora
bash train_mmbert_gpu.sh
```

This script sets the base model via `MODEL` and passes LoRA params (rank/alpha) to each task script.

### 2.2 (Optional) Merge LoRA into base

Some pipelines (e.g. the feedback detector) support `--merge_lora` / `--merge_after_training` to produce a **merged** checkpoint.

Merged checkpoints are often easier to deploy and can simplify inference stacks.

---

## 3) Full fine-tuning (non-LoRA)

Full fine-tuning means updating all model weights instead of training adapters.

### 3.1 Feedback detector pipeline (supports both)

Pipeline:
- `src/training/modernbert_dissat_pipeline/train_feedback_detector.py`

Full fine-tuning example:

```bash
cd src/training/modernbert_dissat_pipeline
pip install -r requirements.txt

python train_feedback_detector.py \
  --model_name llm-semantic-router/mmbert-32k-yarn \
  --output_dir models/mmbert_feedback_detector \
  --epochs 5 \
  --batch_size 16
```

LoRA example (adapter + merge):

```bash
python train_feedback_detector.py \
  --model_name llm-semantic-router/mmbert-32k-yarn \
  --output_dir models/mmbert_feedback_detector \
  --use_lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --merge_lora \
  --epochs 5
```

### 3.2 Router classifiers (intent/fact-check/pii/jailbreak)

The scripts under `src/training/training_lora/` are primarily designed around the LoRA workflow.

If you need full fine-tuning for these classifiers, the intended approach is:
- disable PEFT/LoRA in the task scripts
- run a standard `transformers.Trainer` fine-tune for the same datasets

(We can add a follow-up PR to standardize a full fine-tune entrypoint for all 4 tasks if needed.)

---

## 4) Sequence length (important)

Even though `mmbert-32k-yarn` is a long-context model, most classifier datasets are short.

Recommendations:
- Keep `max_seq_length` modest (e.g. 512/1024) unless you have a concrete long-context training requirement.
- If you truly want >8k context training, set task-specific `max_seq_length` explicitly and ensure the base model config/tokenizer supports the requested window.

---

## 5) Troubleshooting

- If you hit OOM:
  - reduce batch size
  - reduce max_seq_length
  - enable gradient accumulation
  - prefer LoRA over full fine-tuning

