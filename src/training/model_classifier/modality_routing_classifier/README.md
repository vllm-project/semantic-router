# Modality Routing Classifier

For the Vela workflow with source-group isolation and expanded output contracts,
see [Vela Modality repair](VELA_REPAIR.md). This page describes the original
training pipeline.

This pipeline trains a three-class prompt classifier:

| Label | Intended response |
|---|---|
| `AR` | text |
| `DIFFUSION` | generated image |
| `BOTH` | text plus an image or diagram |

The classifier predicts requested output modality, not whether an image model
is available or whether image generation is safe for the prompt.

## Environment

This directory has its own uv project (`pyproject.toml` and `uv.lock`). Create
the environment from the lock and run Python through it:

```bash
uv sync --locked                          # training, evaluation and label_audit
uv sync --locked --group dev              # also pytest, to run the tests
uv run --group dev pytest                 # from this directory
uv run python modality_routing_fixed_split_trainer.py --help
```

Optional groups: `audit` for the Claude API judge in `label_audit/`, and
`exploration` for the SCX Router scripts in `exploration_lfm25_scx/`.

On Linux the default `torch` wheel is a CUDA build, so no extra index is needed
for a GPU host. `requirements.txt` and `requirements-lock.txt` are kept for the
existing docs. Note that `run_training.sh` below still runs its own unpinned
`pip install` and does not use this lock yet.

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
