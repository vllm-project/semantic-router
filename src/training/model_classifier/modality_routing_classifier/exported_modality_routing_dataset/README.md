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

## Provenance and licenses

Every row comes from one of the public sources below, or from the router's own
template generator, and keeps the terms of its source. The licenses are the ones
each Hugging Face dataset card declares (checked 2026-09-19). Check them before
redistributing or using this export commercially.

| Source | Declared license | Rows in this export |
|--------|------------------|---------------------|
| `Gustavosta/Stable-Diffusion-Prompts` | `unknown` | DIFFUSION rows (1,864 from the SD prompt sources, which the exporter tries in order) |
| `FredZhang7/stable-diffusion-prompts-2.47M` | CreativeML OpenRAIL-M | fallback for the SD prompt rows above |
| `tatsu-lab/alpaca` | CC BY-NC 4.0 (non-commercial) | 500 (AR) |
| `databricks/databricks-dolly-15k` | CC BY-SA 3.0 | 500 (AR) |
| `OpenAssistant/oasst2` | Apache-2.0 | 500 (AR) |
| `allenai/WildChat` | ODC-By | 654 (AR, DIFFUSION and BOTH, labelled by regex) |
| `mqliu/InterleavedBench` | none declared | 447 (BOTH) |
| Router template generator | this repository | 590 (BOTH) |

- The Alpaca rows are non-commercial, so this export is not cleanly reusable
  under the repository's license.
- The sources listed under "Sources" above are everything the exporter can pull
  from. The table lists only what contributed rows to this export.
