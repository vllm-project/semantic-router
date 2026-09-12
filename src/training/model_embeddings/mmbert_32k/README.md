# mmBERT-32K training

This package trains the three related mmBERT-32K text models used for
long-context embedding and reranking.

| Model | Entry point | Configuration | Objective |
| --- | --- | --- | --- |
| `mmbert-32k-yarn` | `foundation.py` | `configs/foundation.json` | 32K masked-language continuation with YaRN |
| `mmbert-embed-32k-2d-matryoshka` | `embedder.py` | `configs/embedder.json` | Multiple-negatives ranking with 2D Matryoshka supervision |
| `mmbert-rerank-32k-2d-matryoshka` | `reranker.py` | `configs/reranker.json` | Binary relevance loss across 20 layer/dimension heads |

The public [mmBERT-32K model guide](../../../../website/docs/training/mmbert-32k-models.md)
explains when to use each architecture. This README focuses on running the
checked training configurations.

## Reproduce Vela text checkpoints

The Vela task-weight continuation uses the run-specific `reproduction/`
package shipped with `llm-semantic-router/Vela-1.0-Encoder-307M-Embedding`
and `llm-semantic-router/Vela-1.0-Encoder-307M-Reranker`. Find each published
checkpoint and its immutable revision in the
[Vela model collection](https://huggingface.co/collections/llm-semantic-router/vela-10-router-models-6aa555ba70cc6997d6d67798).
The BGE/AllNLI configurations below are the original training entry points;
they do not reproduce the Vela MIRACL, semantic-pair, and long-context repair
mixtures.

Download the chosen checkpoint's reproduction package, using the exact
revision reported by its model card:

```bash
export VELA_MODEL=llm-semantic-router/Vela-1.0-Encoder-307M-Embedding
export VELA_REVISION="REPLACE_WITH_MODEL_CARD_COMMIT_SHA"
export VELA_SNAPSHOT=/path/to/vela-snapshot

hf download "$VELA_MODEL" --revision "$VELA_REVISION" \
  --include 'reproduction/*' --local-dir "$VELA_SNAPSHOT"
cd "$VELA_SNAPSHOT/reproduction"
python -m pip install --requirement requirements.txt
export VELA_REPRODUCTION_ROOT="$PWD/data-and-runs"
python download_inputs.py --root "$VELA_REPRODUCTION_ROOT"
python train_vela_long_repair_clean.py --help
```

Install the package's recorded PyTorch accelerator build before its Python
requirements. Follow `reproduction/README.md` for the complete initial
training, development selection, clean correction, frozen final evaluation,
and ONNX export commands. It includes pinned inputs, source hashes, actual
training arguments, and the explicit intermediate/full-layer representation
contract. The task models continue their original task weights; shared encoder
architecture does not imply descent from the newly continued Vela Base weights.

The Vela PAWS-X loader defaults to excluding either-sentence empty or `NS`
placeholders with `strip().casefold()`, using [pawsx_data.py](pawsx_data.py).
The clean correction records exclusion counts and asserts that sampled pairs
contain no placeholders. Archived recipes explicitly opt into historical
sampling only to reproduce their recorded runs; their existing manifests and
scores remain unchanged. Raw and cleaned PAWS-X evaluations are reported
separately, with thresholds selected on development data. The BGE/AllNLI and
BGE/Quora/FEVER loaders in this repository do not load PAWS-X, so this
dataset-specific placeholder rule is not applied to those sources.

## Install

Use a Python environment with a compatible PyTorch build, then install the
family dependencies:

```bash
python -m pip install --requirement \
  src/training/model_embeddings/mmbert_32k/requirements.txt
```

The accelerator image and Python package versions for a release run are pinned
in `runtime.json`. Keep that environment, the resolved configuration, and
output checksums with the run.

## Configure paths

Run from the repository root and keep datasets and checkpoints outside Git:

```bash
export PYTHONPATH="$PWD/src"
export MMBERT32K_FOUNDATION_DATA=/path/to/tokenized-cc100-32k
export MMBERT32K_FOUNDATION_OUTPUT=/path/to/mmbert-32k-yarn
export MMBERT32K_BGE_DATA=/path/to/bge-m3-data
export MMBERT32K_EMBEDDER_OUTPUT=/path/to/mmbert-embed-32k-2d
export MMBERT32K_RERANKER_OUTPUT=/path/to/mmbert-rerank-32k-2d
```

All model and dataset revisions, random seeds, objectives, and production
hyperparameters live in the JSON configurations. Command-line arguments are
applied after the config and are explicit overrides.

## Inspect commands first

The family runner can resolve a command without importing the ML stack:

```bash
python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/foundation.json \
  --stage prepare --print-command

python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/foundation.json \
  --stage train --print-command

python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/embedder.json \
  --print-command

python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/reranker.json \
  --print-command
```

Remove `--print-command` to execute the selected workflow.

## Train the 32K foundation

Foundation data preparation streams the nine configured CC-100 languages,
inserts tokenizer document boundaries, and emits only complete 32,768-token
sequences. The target is 30,774 sequences: 3,420 each for `en`, `zh-Hans`, and
`de`, followed by 3,419 each for `fr`, `es`, `ru`, `ar`, `ja`, and `ko`.

Preparation uses two passes so the second pass can verify the exact compressed
source prefixes observed by the first:

```bash
export MMBERT32K_FOUNDATION_AUDIT=/path/to/mmbert32k-cc100-audit

python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/foundation.json \
  --stage prepare --output_dir "$MMBERT32K_FOUNDATION_AUDIT"

python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/foundation.json \
  --stage prepare \
  --source_prefix_contract_dir "$MMBERT32K_FOUNDATION_AUDIT"
```

The replay writes a packing manifest that binds source counters and hashes,
packed-token content, Arrow schema, and every shard checksum. Foundation
training validates that manifest before loading model weights:

```bash
python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/foundation.json \
  --stage train
```

The training config applies official YaRN position scaling before model
construction, validates each rotary layer, uses SDPA attention, and continues
masked-language modeling for one epoch. It also flushes and correctly scales
the final partial gradient-accumulation window.

CC-100's dataset card does not declare a license and its content is derived
from Common Crawl. Preparation requires the explicit acknowledgement recorded
in the config. Review the dataset card, Common Crawl terms, privacy, and the
intended release jurisdiction before redistributing packed data or publishing
new weights.

## Train the embedder

Download the BGE-M3 training data at the revision pinned in the config, then
run:

```bash
hf download Shitao/bge-m3-data \
  --repo-type dataset \
  --revision a69db8b86e9c1767d193ee0de95e5c4001a71eae \
  --local-dir "$MMBERT32K_BGE_DATA"

python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/embedder.json
```

The configuration trains normalized embeddings at dimensions 768, 512, 256,
128, and 64 and enables adaptive-layer supervision. It also includes the
pinned AllNLI training source and STS-B evaluation source.

## Train the reranker

The reranker uses the same BGE-M3 data directory:

```bash
python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/reranker.json
```

The trainer creates heads at layers 3, 6, 11, and 22 for dimensions 768, 512,
256, 128, and 64. Keep `classification_heads.pt` and
`matryoshka_config.json` with the encoder and tokenizer when packaging the
artifact.

## Portable launchers

These wrappers resolve through the same configs:

```bash
bash src/training/model_embeddings/mmbert_32k/run_rope_training.sh --print-command
bash src/training/model_embeddings/mmbert_32k/run_bge_style_training.sh --print-command
bash src/training/model_embeddings/mmbert_32k/run_rerank_2d_matryoshka_training.sh --print-command
```

Set `MMBERT32K_STAGE=prepare` when using `run_rope_training.sh` for data
preparation. `MMBERT32K_PYTHON_BIN` selects a non-default Python executable.

## Validate

The lightweight contract suite does not download models:

```bash
python -m unittest discover \
  -s src/training/model_embeddings/mmbert_32k/tests \
  -p 'test_*.py'
```

Before release, evaluate the embedder at every supported dimension and selected
layer, evaluate all reranker heads, and run long-context loading and loss checks
for the foundation. Record the exact revisions, resolved config, dataset
manifest, metrics, and artifact checksums.
