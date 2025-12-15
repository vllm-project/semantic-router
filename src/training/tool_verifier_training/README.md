# Jailbreak & Prompt Injection Detection

Train a two-stage pipeline for detecting jailbreak/prompt injection attacks and unauthorized tool calls.

## Architecture

Following the hallucination detection pipeline pattern:

| Stage | Model | Task | Key Metric |
|-------|-------|------|------------|
| **Stage 1** | FunctionCallSentinel | Prompt classification | Injection F1 |
| **Stage 2** | ToolCallVerifier | Token-level tool-call verification | UNAUTHORIZED F1 |

## Labels

### Stage 1: Sentinel (Sequence Classification)

| Label | Description |
|-------|-------------|
| SAFE | Normal, benign prompts |
| INJECTION_RISK | Jailbreak, prompt injection, or tool-abuse attempts |

### Stage 2: ToolCallVerifier (Token Classification)

| Label | Severity | Description |
|-------|----------|-------------|
| AUTHORIZED | 0 | Tool call aligns with user request |
| SUSPICIOUS | 2 | Potentially problematic argument |
| UNAUTHORIZED | 4 | Clearly violates policy/intent |

## Quick Start

```bash
# 1. Install dependencies
pip install transformers datasets scikit-learn accelerate tqdm

# 2. Login to HuggingFace (required for gated datasets like WildJailbreak)
huggingface-cli login

# 3. Generate training data
python -m data_generation.cli --output-dir data/generated --preset full

# 4. Train Stage 1 (Sentinel - prompt classifier)
python train_sentinel.py \
    --train-data data/generated/stage1_train.json \
    --dev-data data/generated/stage1_dev.json \
    --output-dir output/sentinel \
    --model-name answerdotai/ModernBERT-base \
    --batch-size 32 \
    --epochs 5 \
    --learning-rate 3e-5

# 5. Train Stage 2 (ToolCallVerifier - token classifier)
python train_tool_verifier.py \
    --train-data data/generated/stage2_train.json \
    --dev-data data/generated/stage2_dev.json \
    --output-dir output/verifier \
    --model-name answerdotai/ModernBERT-base \
    --batch-size 4 \
    --epochs 6 \
    --learning-rate 1e-5
```

## Files

| File/Directory | Purpose |
|----------------|---------|
| `data_generation/` | **Modular data generation package** |
| `data_generation/cli.py` | Command-line interface for data generation |
| `data_generation/config.py` | Configuration settings and presets |
| `data_generation/orchestrator.py` | Coordinates all generators and loaders |
| `data_generation/generators/` | Attack pattern generators (filesystem, network, email, etc.) |
| `data_generation/loaders/` | HuggingFace dataset loaders |
| `dataset_jailbreak.py` | Dataset classes for both stages |
| `train_sentinel.py` | Train Stage 1 sequence classifier |
| `train_tool_verifier.py` | Train Stage 2 token classifier |

## Data Sources

### Stage 1 (Prompt Classification)

**Jailbreak/Injection datasets:**

- [jailbreak_llms](https://github.com/verazuo/jailbreak_llms) - 15K prompts with 1,405 jailbreaks
- [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) - 262K synthetic safety pairs
- [HackAPrompt](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset) - ~600K competition prompts
- [Qualifire benchmark](https://huggingface.co/datasets/qualifire/prompt-injections-benchmark) - 5K labeled prompts

**Benign datasets:**

- [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) - 52K instructions
- [Dolly](https://huggingface.co/datasets/databricks/dolly-15k) - 15K diverse instructions

### Stage 2 (Tool-Call Verification)

**HuggingFace datasets (converted to tool-call format):**

- [LLMail-Inject](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) - ~10K tool-call attack attempts
- [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) - Converted to tool-call scenarios
- [HackAPrompt](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset) - Converted to tool-call scenarios
- [JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) - Harmful behavior patterns

**Synthetic generators:**

- `adversarial.py` - Intent-mismatch attacks (correct tool, wrong args)
- `filesystem.py` - File/directory operation attacks
- `network.py` - Network/API attacks
- `email.py` - Email tool attacks
- `financial.py` - Financial transaction attacks
- `code_execution.py` - Code execution attacks
- `authentication.py` - Authentication/access attacks
- `sysadmin.py` - System administration attacks

## Training Configuration

### Stage 1: Sentinel

```python
{
    "model_name": "answerdotai/ModernBERT-base",
    "batch_size": 32,
    "epochs": 5,
    "learning_rate": 3e-5,
    "max_length": 512,
}
```

### Stage 2: ToolCallVerifier

```python
{
    "model_name": "answerdotai/ModernBERT-base",
    "batch_size": 4,
    "epochs": 6,
    "learning_rate": 1e-5,
    "max_length": 2048,
}
```

## Training on AMD ROCm GPUs

ModernBERT requires special handling on AMD ROCm due to Flash Attention compatibility issues. Use the official ROCm PyTorch Docker container which has proper Flash Attention support.

### Prerequisites

- AMD GPU (e.g., MI300X, MI250, MI210)
- Docker with ROCm support
- ROCm drivers installed on host

### Using Docker (Recommended)

```bash
# 1. Pull the ROCm PyTorch image
docker pull rocm/pytorch:latest

# 2. Start container with GPU access and bind mount
docker run -d --name train_modernbert \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size=16g \
    -v /path/to/semantic-router-dev:/workspace \
    -w /workspace/src/training/tool_verifier_training \
    rocm/pytorch:latest \
    sleep infinity

# 3. Install dependencies inside container
docker exec train_modernbert pip install transformers datasets scikit-learn accelerate tqdm

# 4. Run training inside container
docker exec train_modernbert python train_sentinel.py \
    --train-data data/full_dataset_v2/stage1_train.json \
    --dev-data data/full_dataset_v2/stage1_dev.json \
    --output-dir output/sentinel_modernbert \
    --model-name answerdotai/ModernBERT-base \
    --batch-size 32 \
    --epochs 5 \
    --learning-rate 3e-5

docker exec train_modernbert python train_tool_verifier.py \
    --train-data data/full_dataset_v2/stage2_train.json \
    --dev-data data/full_dataset_v2/stage2_dev.json \
    --output-dir output/verifier_modernbert \
    --model-name answerdotai/ModernBERT-base \
    --batch-size 4 \
    --epochs 6 \
    --learning-rate 1e-5
```

### Key Configuration for ROCm

The training scripts use `attn_implementation="sdpa"` which enables PyTorch's Scaled Dot-Product Attention. On ROCm with PyTorch 2.3+, this automatically uses Flash Attention when available.

```python
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    attn_implementation="sdpa",  # Required for ROCm compatibility
    ...
)
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| NaN/Inf loss | Use Docker container with PyTorch 2.3+ (rocm/pytorch:latest) |
| Flash Attention errors | Set `attn_implementation="sdpa"` instead of `flash_attention_2` |
| Out of memory | Reduce `batch_size` or `max_length` |
| Slow training | Ensure GPU is detected: `python -c "import torch; print(torch.cuda.is_available())"` |

### Verified Performance on MI300X

| Model | Epochs | Time | Best F1 |
|-------|--------|------|---------|
| Stage 1 (Sentinel) | 5 | ~18 min | 97.7% |
| Stage 2 (Verifier) | 6 | ~40 min | 93.8% |

## Expected Results

### Stage 1: Sentinel

| Metric | Target | Achieved |
|--------|--------|----------|
| Accuracy | > 90% | **97.6%** |
| Injection Precision | > 85% | **98.0%** |
| Injection Recall | > 90% | **97.2%** |
| Injection F1 | > 87% | **97.7%** |
| ROC-AUC | > 0.95 | **99.8%** |

### Stage 2: ToolCallVerifier

| Metric | Target | Achieved |
|--------|--------|----------|
| Accuracy | > 85% | **93.9%** |
| UNAUTHORIZED Precision | > 70% | **97.6%** |
| UNAUTHORIZED Recall | > 60% | **90.3%** |
| UNAUTHORIZED F1 | > 65% | **93.8%** |

## Integration with Semantic Router

After training, export models for production:

```bash
# Models will be saved to:
# - output/function_call_sentinel/
# - output/tool_call_verifier/

# Copy to semantic router models directory:
cp -r output/function_call_sentinel models/jailbreak_sentinel
cp -r output/tool_call_verifier models/tool_call_verifier
```

Configure in `config.yaml`:

```yaml
jailbreak_detection:
  enabled: true
  sentinel_model:
    model_id: "models/jailbreak_sentinel"
    threshold: 0.7
  tool_verifier_model:
    model_id: "models/tool_call_verifier"
    threshold: 0.5
  on_injection_detected: "block"
  on_unauthorized_tool_call: "block"
```

## References

- [LLMail-Inject Paper](https://arxiv.org/abs/2402.05827) - Microsoft tool-call jailbreak research
- [jailbreak_llms (CCS'24)](https://github.com/verazuo/jailbreak_llms) - In-the-wild jailbreak analysis
- [Hallucination Detection Pipeline](../nli_hallucination_detection/) - Reference architecture
- [Implementation Plan](../../../docs/jailbreak-detection-implementation-plan.md) - Full design document

## Directory Structure

```
tool_verifier_training/
├── README.md                      # This file
├── dataset_jailbreak.py           # Dataset classes
├── train_sentinel.py              # Stage 1 training
├── train_tool_verifier.py         # Stage 2 training
│
├── data_generation/               # Data generation package
│   ├── __init__.py
│   ├── cli.py                     # Command-line interface
│   ├── config.py                  # Configuration presets
│   ├── orchestrator.py            # Main coordinator
│   ├── base.py                    # Base generator class
│   ├── entities.py                # Realistic data pools
│   ├── generators/                # Attack pattern generators
│   │   ├── adversarial.py         # Intent-mismatch attacks
│   │   ├── filesystem.py          # File operation attacks
│   │   ├── network.py             # Network/API attacks
│   │   ├── email.py               # Email tool attacks
│   │   ├── financial.py           # Financial transaction attacks
│   │   └── ...
│   └── loaders/                   # HuggingFace dataset loaders
│       ├── huggingface.py         # All HF dataset loaders
│       └── __init__.py
│
├── data/generated/                # Generated training data
│   ├── stage1_train.json          # ~27K samples
│   ├── stage1_dev.json            # ~7K samples
│   ├── stage2_train.json          # ~30K samples
│   └── stage2_dev.json            # ~8K samples
│
└── output/
    ├── sentinel/                  # Trained Stage 1 model
    └── verifier/                  # Trained Stage 2 model
```
