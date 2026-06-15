# Fact-Check Classifier Training

Fine-tune a ModernBERT/BERT model with LoRA to classify prompts as `FACT_CHECK_NEEDED` or `NO_FACT_CHECK_NEEDED`.

## Quick Start on Training

### Prerequisites

```bash
# Install dependencies
pip install transformers datasets peft torch accelerate scikit-learn
```

### Optional: Pre-download datasets

```bash
# Pre-download script-based datasets for faster training
./setup_datasets.sh ./datasets_cache
```

### Training Command

```bash
# Train with full dataset (50k samples, ~10 min on GPU)
python fact_check_bert_finetuning_lora.py \
    --mode train \
    --model modernbert-base \
    --max-samples 50000 \
    --epochs 3 \
    --batch-size 32 \
    --data-dir ./datasets_cache
```

## Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | bert-base-uncased | Model: `modernbert-base`, `bert-base-uncased`, `roberta-base` |
| `--max-samples` | 2000 | Total samples (50000 for full training) |
| `--epochs` | 5 | Training epochs (3 is usually sufficient) |
| `--batch-size` | 16 | Batch size (32 for faster training with enough VRAM) |
| `--lora-rank` | 16 | LoRA rank |
| `--data-dir` | None | Path to cached datasets from `setup_datasets.sh` |

## Testing

### Test Command

```bash
# Test the trained model
python fact_check_bert_finetuning_lora.py \
    --mode test \
    --model-path lora_fact_check_classifier_modernbert-base_r16_model_rust
```

## Datasets Used

### FACT_CHECK_NEEDED

- **NISQ-ISQ** - Information-Seeking Questions (Gold standard dataset, ACL LREC 2024)
- **HaluEval** - QA questions from hallucination benchmark (ACL EMNLP 2023)
- **FaithDial** - Information-seeking dialogue questions (TACL 2022)
- **FactCHD** - Fact-conflicting hallucination queries (Chen et al., 2024)
- **RAG** - Questions for retrieval-augmented generation (neural-bridge/rag-dataset-12000)
- **SQuAD** - Stanford Question Answering Dataset (100k+ Wikipedia fact questions)
- **TriviaQA** - Factual trivia questions (650k question-answer-evidence triples)
- **TruthfulQA** - High-risk factual queries about common misconceptions
- **HotpotQA** - Multi-hop factual reasoning questions
- **CoQA** - Conversational factual questions (127k questions across domains)
- **QASPER** - Information-seeking questions over research papers (NAACL 2021)
- **ELI5** - Explain Like I'm 5 - factual explanation questions
- **Natural Questions** - Google Natural Questions (real user queries)

### NO_FACT_CHECK_NEEDED

- **NISQ-NonISQ** - Non-Information-Seeking Questions (Gold standard dataset)
- **Dolly** - Creative writing, brainstorming, opinion (helps with edge cases)
- **WritingPrompts** - Creative writing prompts from Reddit (300k prompts)
- **Alpaca** - Non-factual instructions (coding, creative, math, opinion)
- **CodeSearchNet** - Programming/technical requests (code documentation)
