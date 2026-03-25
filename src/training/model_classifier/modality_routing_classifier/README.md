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

## Export a Fixed Dataset

The training script builds modality data dynamically. If you want a reproducible
dataset artifact for review or Hugging Face upload, export the complete
`train / validation / test` splits first:

```bash
python src/training/model_classifier/modality_routing_classifier/export_modality_dataset.py \
  --output-dir ./modality-routing-dataset \
  --max-samples 6000 \
  --overwrite
```

With vLLM-generated BOTH examples:

```bash
python src/training/model_classifier/modality_routing_classifier/export_modality_dataset.py \
  --output-dir ./modality-routing-dataset \
  --max-samples 6000 \
  --vllm-endpoint http://localhost:8000/v1 \
  --vllm-model your-model \
  --synthesize-both 2000 \
  --overwrite
```

The exporter writes:

- `train.jsonl`, `validation.jsonl`, `test.jsonl`
- `README.md` dataset card
- `label_mapping.json`
- `dataset_stats.json`
- `export_config.json`
- `hf_dataset/` via `DatasetDict.save_to_disk()`

These root files are upload-friendly if you want to create a Hugging Face dataset
repo locally.

To export and upload in one command:

```bash
HF_TOKEN=hf_xxx python src/training/model_classifier/modality_routing_classifier/export_modality_dataset.py \
  --output-dir ./modality-routing-dataset \
  --max-samples 6000 \
  --repo-id your-name/modality-routing-dataset \
  --push-to-hub \
  --overwrite
```

Note: the current training script still uses a legacy flow where it materializes
the full dataset, merges `train + validation`, and then does an internal `80/20`
re-split for training. The exporter preserves the original deterministic
`train / validation / test` split from `prepare_datasets()`.

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
