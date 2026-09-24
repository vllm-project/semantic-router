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

## Same-run Evaluation Harness (#3856)

`same_run_harness.py` and `same_run_pair.py` run a candidate and its baseline
on the same frozen QSL one prompt at a time, then report latency, peak RSS,
CPU seconds, and per-row routing output. Pairs are joined on `row_id` and must
run on the same host.

**Run BERT baseline:**

```bash
python same_run_harness.py \
  --qsl exported_modality_routing_dataset/test.jsonl \
  --model llm-semantic-router/mmbert32k-modality-router-merged \
  --binding hf \
  --role baseline \
  --warmup 20 --max-length 256 --min-duration-s 60 \
  --output same_run_bert_singlestream.json
```

**Run candidate (e.g. DistilBERT):**

```bash
python same_run_harness.py \
  --qsl exported_modality_routing_dataset/test.jsonl \
  --model <your-checkpoint> \
  --binding hf \
  --role candidate \
  --warmup 20 --max-length 256 --min-duration-s 60 \
  --output same_run_distilbert_singlestream.json
```

**Pair the two runs (same host required):**

```bash
python same_run_pair.py \
  --baseline same_run_bert_singlestream.json \
  --candidate same_run_distilbert_singlestream.json \
  --output same_run_paired.json
```

The pair script exits non-zero and writes no output file if the two runs came
from different machines (`cpu_model`, `core_count`, or `ram_gb` differ).

**Run production baseline (`--binding candle`, `ClassifyMmBert32KModality`):**

Unlike `--binding hf`, candle does not auto-download — build the native
library and helper once, then point `--model` at a local model directory:

```bash
# 1. Build the native library (from repository root)
cd candle-binding
cargo build --release --no-default-features   # CPU; see candle-binding/README.md for CUDA/Metal
go build -o candle-classify ./cmd/classify-helper/
export LD_LIBRARY_PATH="$PWD/target/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"  # DYLD_LIBRARY_PATH on macOS
export CANDLE_CLASSIFY_HELPER="$PWD/candle-classify"
cd ..

# 2. Download a local copy of the model (candle needs a directory, not a Hub ID)
huggingface-cli download llm-semantic-router/mmbert32k-modality-router-merged \
  --local-dir /tmp/mmbert32k-modality

# 3. Run
cd src/training/model_classifier/modality_routing_classifier
python same_run_harness.py \
  --qsl exported_modality_routing_dataset/test.jsonl \
  --model /tmp/mmbert32k-modality \
  --binding candle \
  --role baseline \
  --warmup 20 --max-length 256 --min-duration-s 60 \
  --output same_run_candle_singlestream.json
```

Passing a Hub ID instead of a local directory for `--binding candle` fails
fast with a message telling you which `huggingface-cli download` command to
run. `cpu_s` and `peak_rss_mb` for this binding include the helper
subprocess's own resource usage (`run.includes_helper_resources: true`),
since inference happens in that child process, not in the Python harness.

Quality metrics (per-class precision, recall, threshold selection) are defined
by [#3194](https://github.com/vllm-project/semantic-router/issues/3194). The
harness emits `records[].label` and `records[].output` per row.

**Unit tests (no model download required):**

```bash
python test_same_run.py
```
