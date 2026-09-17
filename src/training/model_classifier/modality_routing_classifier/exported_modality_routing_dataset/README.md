---
pretty_name: Modality Routing Dataset
task_categories:
- text-classification
language:
- en
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
  - split: test
    path: test.jsonl
---

# Modality Routing Dataset

This dataset materializes the dynamic modality routing data builder used by the local
mmBERT-32K modality router training pipeline. The export is intended for review,
versioning, and uploading to a Hugging Face dataset repository.

## Labels

| Label | ID | Description |
|-------|----|-------------|
| AR | 0 | Text-only requests that should route to an autoregressive LLM. |
| DIFFUSION | 1 | Image-generation requests that should route to a diffusion model. |
| BOTH | 2 | Requests that benefit from both text and image responses. |

## Schema

| Column | Type | Description |
|--------|------|-------------|
| text | string | Input user prompt |
| label | int64 | Integer class id |
| label_name | string | Human-readable class label |

## Splits

| Split | Rows | AR | DIFFUSION | BOTH |
|-------|------|------------|--------------------|--------------|
| train | 3538 | 1400 | 1400 | 738 |
| validation | 758 | 300 | 300 | 158 |
| test | 759 | 300 | 300 | 159 |

## Export Configuration

- `max_samples`: 6000
- `synthesize_both`: 0
- `vllm_synthesis_enabled`: disabled
- `vllm_endpoint`: None
- `vllm_model`: None
- `split_strategy`: 70% train / 15% validation / 15% test with random_state=42

## Sources

- `FredZhang7/stable-diffusion-prompts-2.47M`
- `succinctly/midjourney-prompts`
- `Falah/image_generation_prompts_SDXL`
- `nateraw/parti-prompts`
- `fal/image-generation-prompts`
- `OpenAssistant/oasst2`
- `tatsu-lab/alpaca`
- `databricks/databricks-dolly-15k`
- `stingning/ultrachat`
- `lmsys/lmsys-chat-1m`
- `allenai/WildChat`
- `mqliu/InterleavedBench`
- Optional vLLM-generated BOTH prompts when enabled

## Files

- `train.jsonl`, `validation.jsonl`, `test.jsonl`: upload-friendly JSONL splits
- `label_mapping.json`: label to integer mapping
- `dataset_stats.json`: row counts per split and label
- `export_config.json`: reproducibility metadata for this export
- `hf_dataset/`: local `DatasetDict.save_to_disk()` artifact
