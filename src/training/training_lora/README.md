# LoRA Training Scripts

## 📖 Overview

This directory contains **LoRA (Low-Rank Adaptation)** training scripts for fine-tuning transformer models on multiple tasks:

### Classification Tasks

- **Intent Classification** (`classifier_model_fine_tuning_lora/`)
- **PII Detection** (`pii_model_fine_tuning_lora/`)  
- **Security Detection** (`prompt_guard_fine_tuning_lora/`)

### Problem Solving Tasks

- **MMLU-Pro Specialized Solvers** (`mmlu_pro_solver_lora/`) ⭐ NEW!
  - Fine-tune Qwen3-0.6B models to solve graduate-level academic problems
  - 6 specialized experts (math, science, humanities, law, etc.)
  - Chain-of-Thought reasoning with baseline comparison
  - Expected: 40-60% accuracy (vs 10% random baseline)

## 🧠 What is LoRA?

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique that:

- **Reduces trainable parameters** by 99%+ (from 110M to ~1M parameters)
- **Maintains model performance** while using significantly less memory
- **Enables fast training** on consumer hardware (CPU/single GPU)
- **Preserves original model weights** by learning additive low-rank matrices

### Technical Details

LoRA decomposes weight updates into two smaller matrices:

```
W = W₀ + ΔW = W₀ + BA
```

Where:

- `W₀`: Original frozen weights
- `B`: Low-rank matrix (d × r)  
- `A`: Low-rank matrix (r × k)
- `r`: Rank (typically 8-64, we use 16)

## 🏗️ Architecture Support

Our LoRA implementation supports three transformer architectures:

### BERT-base-uncased

- **Target Modules**: `attention.self.query`, `attention.self.value`, `attention.output.dense`, `intermediate.dense`, `output.dense`
- **Performance**: Excellent (0.99+ confidence)
- **Training Time**: ~45-60 minutes per task

### RoBERTa-base  

- **Target Modules**: Same as BERT
- **Performance**: Excellent (0.99+ confidence)
- **Training Time**: ~45-60 minutes per task

### ModernBERT-base

- **Target Modules**: `attn.Wqkv`, `attn.Wo`, `mlp.Wi`, `mlp.Wo`
- **Performance**: Good (0.5-0.7 confidence)
- **Training Time**: ~30-45 minutes per task

## 📁 Directory Structure

```
src/training/training_lora/
├── README.md                           # This file
├── common_lora_utils.py               # Shared utilities
│
├── classifier_model_fine_tuning_lora/ # Intent Classification
│   ├── ft_linear_lora.py             # Training script
│   ├── ft_qwen3_generative_lora.py   # Category classifier
│   ├── ft_linear_lora_verifier.go    # Go verification
│   ├── train_cpu_optimized.sh        # Training automation
│   └── go.mod
│
├── pii_model_fine_tuning_lora/        # PII Detection
│   ├── pii_bert_finetuning_lora.py   # Training script
│   ├── pii_bert_finetuning_lora_verifier.go # Go verification
│   ├── train_cpu_optimized.sh        # Training automation
│   ├── presidio_synth_dataset_v2.json # Training data
│   └── go.mod
│
├── prompt_guard_fine_tuning_lora/     # Security Detection
│   ├── jailbreak_bert_finetuning_lora.py # Training script
│   ├── jailbreak_bert_finetuning_lora_verifier.go # Go verification
│   ├── train_cpu_optimized.sh        # Training automation
│   └── go.mod
│
└── mmlu_pro_solver_lora/              # ⭐ MMLU-Pro Problem Solvers
    ├── ft_qwen3_mmlu_solver_lora[_no_leakage].py  # Main training script, _no_leakage version has no MMLU-Pro data leakage
    └── train_all_specialists[_no_leakage].sh       # Batch training, _no_leakage version has no MMLU-Pro data leakage
```

## 🚀 Quick Start

### Prerequisites

1. **Python Environment**:

2. **Required Libraries**:
   - `torch`, `transformers`, `peft`, `datasets`
   - `accelerate`, `tqdm`, `scikit-learn`

### Training a Model

**Option 1: Automated Training (Recommended)**

```bash
cd classifier_model_fine_tuning_lora/
./train_cpu_optimized.sh
```

**Option 2: Manual Training**

```bash
cd classifier_model_fine_tuning_lora/
python ft_linear_lora.py \
  --model_name bert-base-uncased \
  --rank 16 \
  --alpha 32 \
  --epochs 3 \
  --batch_size 8 \
  --learning_rate 2e-4
```

### Verification

**Python Verification**:

```bash
python ft_linear_lora.py --mode test --model_path ./models/lora_intent_classifier_bert-base-uncased_r16_model_rust
```

**Go Verification**:

```bash
LD_LIBRARY_PATH=~/candle-binding/target/release \
go run ft_linear_lora_verifier.go --model models/lora_intent_classifier_bert-base-uncased_r16_model_rust
```

## 📊 Performance Results

### Key Findings

- **BERT/RoBERTa**: Consistently excellent performance across all tasks
- **ModernBERT**: Good for PII detection, but lower confidence for classification tasks
- **Python-Go Consistency**: Exact numerical consistency achieved for BERT/RoBERTa
- **Training Efficiency**: 99%+ parameter reduction with maintained performance

## 🔧 Configuration

### LoRA Hyperparameters

```python
# Recommended settings (used in our training)
lora_config = LoraConfig(
    r=16,                    # Rank - balance between performance and efficiency
    lora_alpha=32,          # Scaling factor (typically 2×rank)
    target_modules=get_target_modules_for_model(model_name),
    lora_dropout=0.1,       # Regularization
    bias="none",            # Don't adapt bias terms
    task_type=TaskType.SEQ_CLS  # or TOKEN_CLS for PII
)
```

### Training Parameters

```python
# Optimized for CPU training
training_args = TrainingArguments(
    output_dir="./models",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_num_workers=0,  # CPU optimization
    fp16=False,               # CPU compatibility
    push_to_hub=False
)
```

## 🎯 Task-Specific Details

### Intent Classification

- **Task Type**: Sequence Classification
- **Classes**: `business`, `law`, `psychology`
- **Dataset**: Synthetic business/legal/psychology queries
- **Metric**: Accuracy, Confidence

### PII Detection  

- **Task Type**: Token Classification
- **Labels**: `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `STREET_ADDRESS`, `US_SSN`, etc.
- **Dataset**: Presidio synthetic dataset (29K examples)
- **Metric**: Token-level F1, Entity-level accuracy

### Security Detection

- **Task Type**: Sequence Classification  
- **Classes**: `safe`, `unsafe`
- **Dataset**: Toxic-chat, Salad-data
- **Metric**: Binary classification accuracy

## 🔍 Verification & Testing

Each training directory includes:

1. **Python Demo**: `--mode test` flag for inference testing
2. **Go Verifier**: CGO bindings for production inference
3. **Consistency Check**: Ensures Python-Go numerical consistency

### Example Verification Commands

```bash
# Intent Classification
python ft_linear_lora.py --mode test
go run ft_linear_lora_verifier.go --model path/to/model

# PII Detection  
python pii_bert_finetuning_lora.py --mode test
go run pii_bert_finetuning_lora_verifier.go --pii-token-model path/to/model

# Security Detection
python jailbreak_bert_finetuning_lora.py --mode test  
go run jailbreak_bert_finetuning_lora_verifier.go --jailbreak-model path/to/model
```

## 🛠️ Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `per_device_train_batch_size` to 4 or 2
2. **Slow Training**: Ensure `dataloader_num_workers=0` for CPU
3. **Go Compilation**: Set `LD_LIBRARY_PATH` to Rust library path
4. **Model Loading**: Use absolute paths for model directories

### Environment Setup

```bash
# Set library path for Go
export LD_LIBRARY_PATH=~/candle-binding/target/release

# Verify Rust library
ls -la ~/candle-binding/target/release/libcandle_semantic_router.so
```

## 📈 Production Integration

Trained LoRA models are automatically discovered and used by the semantic-router system:

1. **Model Discovery**: `model_discovery.go` automatically finds LoRA models
2. **Architecture Selection**: Prioritizes BERT > RoBERTa > ModernBERT  
3. **Batch Inference**: `UnifiedClassifier` uses high-confidence LoRA models
4. **API Integration**: `/api/v1/classify/batch` endpoint leverages LoRA performance

### Model Naming Convention

```
lora_{task}_{architecture}_r{rank}_model_rust/
├── config.json
├── adapter_config.json  
├── adapter_model.safetensors
├── label_mapping.json (for token classification)
└── tokenizer files...
```

## 📚 References

- **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **PEFT Library**: [Hugging Face PEFT](https://github.com/huggingface/peft)
