# Standalone Hallucination Detection Training

## Install

```bash
pip install -r requirements.txt
```

## Quick Start (Docker)

```bash
# Run full training pipeline with 32K ModernBERT
./run_training.sh
```

## Prepare Data

```bash
# Option 1: Use existing local files
python prepare_data.py \
    --ragtruth-path /path/to/ragtruth_data.json \
    --dart-path /path/to/dart_spans.json \
    --e2e-path /path/to/e2e_spans.json \
    --output-dir ./data

# Option 2: Download DART/E2E from HuggingFace
python prepare_data.py \
    --ragtruth-path /path/to/ragtruth_data.json \
    --download-augmentation \
    --output-dir ./data
```

## Train

```bash
python finetune.py \
    --train-path ./data/train.json \
    --dev-path ./data/dev.json \
    --test-path ./data/test.json \
    --output-dir ./output/haldetect-32k \
    --batch-size 8 \
    --learning-rate 1e-5 \
    --epochs 6
```

## Inference

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained("./output/haldetect-32k")
tokenizer = AutoTokenizer.from_pretrained("./output/haldetect-32k")

inputs = tokenizer(context, answer, return_tensors="pt", truncation="only_first", max_length=8192)
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)  # 0=supported, 1=hallucinated
```
