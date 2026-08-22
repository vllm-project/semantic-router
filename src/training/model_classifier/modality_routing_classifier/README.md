# Modality Routing Classifier

This pipeline trains a three-class prompt classifier:

| Label | Intended response |
|---|---|
| `AR` | text |
| `DIFFUSION` | generated image |
| `BOTH` | text plus an image or diagram |

The classifier predicts requested output modality, not whether an image model
is available or whether image generation is safe for the prompt.

## Train

`run_training.sh` installs its Python packages, builds the dataset, trains an
mmBERT LoRA adapter, and runs the script's inference demo:

```bash
bash run_training.sh
```

Environment variables such as `MODEL`, `EPOCHS`, `BATCH_SIZE`, `MAX_SAMPLES`,
and `LEARNING_RATE` override its defaults. If `VLLM_ENDPOINT` is set, the script
can synthesize examples for the `BOTH` class; review those examples before
using them as labels.

For direct control:

```bash
python modality_routing_bert_finetuning_lora.py \
  --mode train \
  --model mmbert-32k \
  --max-samples 6000 \
  --output-dir models/modality-router
```

Use `--help` for current LoRA, GPU, and synthesis options.

## Export a Reviewable Dataset

The exporter writes deterministic train, validation, and test JSONL files,
label mappings, dataset statistics, export configuration, a dataset card, and a
Hugging Face `DatasetDict`:

```bash
python export_modality_dataset.py \
  --output-dir modality-routing-dataset \
  --max-samples 6000 \
  --overwrite
```

To add model-generated `BOTH` examples, pass `--vllm-endpoint`,
`--vllm-model`, and `--synthesize-both`. Publishing with `--push-to-hub`
requires `--repo-id` and an `HF_TOKEN`.

The training script currently rebuilds and internally splits its dataset,
whereas the exporter preserves the split returned by `prepare_datasets()`.
Use the exported split for dataset review and reproducible evaluation.

## Data and Evaluation

The data loader draws text-only prompts from instruction datasets,
image-generation prompts from DiffusionDB, and mixed-modality prompts from
curated templates or optional synthesis. Check dataset revisions and class
balance before every training run.

Report per-class precision and recall, the confusion matrix, multilingual
coverage, and failure cases such as requests that mention an image without
asking to create one. Validate the exported adapter through the router's actual
modality signal path before treating it as supported.
