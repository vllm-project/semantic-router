# Fact-Check Classifier Training

This script fine-tunes a sequence classifier to predict whether a user prompt
needs factual verification:

- `FACT_CHECK_NEEDED`
- `NO_FACT_CHECK_NEEDED`

The label describes the prompt's verification need. It does not verify an
answer or replace a fact-checking system.

## Setup

Create an isolated environment and install PyTorch, Transformers, Datasets,
PEFT, Accelerate, scikit-learn, and the other imports required by the script.
`setup_datasets.sh` can pre-populate a local dataset cache:

```bash
./setup_datasets.sh ./datasets_cache
```

Review the source datasets and their licenses before training. The loader
combines information-seeking, QA, creative-writing, instruction, and coding
datasets into the two labels; inspect the generated balance and examples rather
than treating source dataset names as ground truth.

## Train

Start with a limited sample:

```bash
python fact_check_bert_finetuning_lora.py \
  --mode train \
  --model mmbert-32k \
  --max-samples 2000 \
  --epochs 1 \
  --data-dir ./datasets_cache \
  --output-dir ./models/fact-check-smoke
```

Increase the sample size and epochs only after checking label distribution,
validation behavior, and available memory. Run `python
fact_check_bert_finetuning_lora.py --help` for current model and LoRA options.

## Test an Export

```bash
python fact_check_bert_finetuning_lora.py \
  --mode test \
  --model-path ./models/fact-check-smoke
```

Before publishing, evaluate a held-out set that reflects the prompts expected
in deployment. Report per-class precision and recall, the decision threshold,
dataset revisions, split policy, base model, and known false-positive and
false-negative cases.
