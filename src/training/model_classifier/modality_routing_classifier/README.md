# Modality Routing Classifier

Trains an mmBERT-32K YaRN model with LoRA to classify user prompt intent into one of three response modalities:

| Label | Description | Example |
|-------|-------------|---------|
| **AR** | Text-only response (autoregressive LLM) | "What is the capital of France?" |
| **DIFFUSION** | Image generation (diffusion model) | "A cyberpunk city at night, neon lights" |
| **BOTH** | Text explanation + image/diagram | "Explain photosynthesis and show a diagram" |

## Quick Start

```bash
# Basic training (uses template-generated BOTH class)
bash run_training.sh

# With vLLM synthesis for higher-quality BOTH class data
VLLM_ENDPOINT=http://localhost:8000/v1 bash run_training.sh

# Custom configuration
MODEL=mmbert-32k EPOCHS=10 MAX_SAMPLES=10000 bash run_training.sh
```

## Datasets

### DIFFUSION class (image generation intent)

- **DiffusionDB** (`poloclub/diffusiondb`): 1.8M unique real Stable Diffusion prompts
  - ACL 2023 Best Paper Honorable Mention
  - Real user prompts, not synthetic

### AR class (text-only intent)

- **OASST2** (`OpenAssistant/oasst2`): 135K instruction-following conversations (35+ languages)
- **Alpaca** (`tatsu-lab/alpaca`): 52K instruction-following examples (Stanford)
- **Dolly** (`databricks/databricks-dolly-15k`): 15K categorized instructions

### BOTH class (mixed modality intent)

- **vLLM Synthesis**: Use `--vllm-endpoint` to generate diverse BOTH prompts via LLM
- **Seed examples**: 40+ curated examples across diverse domains
- **Template generation**: Fallback template-based generation

## Architecture

- **Base model**: mmBERT-32K YaRN (ModernBERT architecture, 32K context, 1800+ languages)
- **Adaptation**: LoRA (Low-Rank Adaptation) on attention + MLP layers
- **Target modules**: `attn.Wqkv`, `attn.Wo`, `mlp.Wi`, `mlp.Wo`
- **Trainable parameters**: ~0.02% of total (99%+ reduction)

## Training Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `mmbert-32k` | Base model |
| `--lora-rank` | `32` | LoRA rank (capacity) |
| `--lora-alpha` | `64` | LoRA alpha (2x rank) |
| `--epochs` | `8` | Training epochs |
| `--batch-size` | `32` | Batch size |
| `--learning-rate` | `2e-5` | Learning rate |
| `--max-samples` | `6000` | Total samples (2000 per class) |
| `--vllm-endpoint` | None | vLLM endpoint for BOTH synthesis |
| `--synthesize-both` | `0` | BOTH prompts to synthesize (auto: 2000 if endpoint set) |

## Output

The trained model is saved with:

- `adapter_config.json` - LoRA adapter configuration
- `adapter_model.safetensors` - LoRA weights
- `label_mapping.json` - Label ID mapping
- `modality_mapping.json` - Go/Rust compatibility mapping
- `lora_config.json` - LoRA hyperparameters
- `eval_results.json` - Final evaluation metrics
