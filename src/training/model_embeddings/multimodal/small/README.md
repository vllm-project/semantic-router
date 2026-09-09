# Small multimodal embedding training

This package trains and packages
[`llm-semantic-router/multi-modal-embed-small`](https://huggingface.co/llm-semantic-router/multi-modal-embed-small).
`artifacts.json` maps the public artifact to its checked configuration and
local train, evaluation, and release entrypoints.

The compact model uses:

- `sentence-transformers/all-MiniLM-L6-v2` for text;
- `google/siglip-base-patch16-512` for images;
- `openai/whisper-tiny` for audio;
- a two-layer transformer fusion module;
- 384-dimensional normalized embeddings with Matryoshka truncation and
  adaptive layer exits.

The checked Stage 1 configuration runs six frozen-encoder alignment epochs on
LLaVA-CC3M with batch 64 per process, BF16, learning rate `1e-4`, cosine decay,
and Matryoshka-wrapped InfoNCE training.

## Layout

| Path | Responsibility |
| --- | --- |
| `models/` | text, image, audio, fusion, and adaptive-exit architecture |
| `losses/` | InfoNCE, alignment, and 2D Matryoshka objectives |
| `data/` | random-access and sequential cached tensor shards |
| `raw_data.py`, `download_data.py` | explicit raw dataset inputs and materialization |
| `stages.py`, `wrappers.py` | stage-specific freezing and modality-pair forwards |
| `training.py`, `runner.py` | epoch mechanics and orchestration |
| `evaluate.py`, `release.py` | retrieval metrics and guarded publication packaging |

Cached shards are validated before distributed training starts, each rank gets
an equal shard count, and checkpoints are written only at optimizer
boundaries. The final partial gradient-accumulation window is included.

## Train

Install the compact workflow dependencies, then set all storage locations:

```bash
python -m pip install --requirement \
  src/training/model_embeddings/multimodal/small/requirements.txt

export PYTHONPATH="$PWD/src"
export MM_EMBED_SMALL_TRAIN_CACHE=/path/to/cache/train
export MM_EMBED_SMALL_VAL_CACHE=/path/to/cache/validation
export MM_EMBED_SMALL_OUTPUT_DIR=/path/to/output
```

Inspect the resolved configuration without starting a run:

```bash
python -m training.model_embeddings.multimodal.small.train \
  --config src/training/model_embeddings/multimodal/small/configs/production.yaml \
  --print-config
```

Launch the recorded eight-device topology with `torchrun`:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m training.model_embeddings.multimodal.small.train \
  --config src/training/model_embeddings/multimodal/small/configs/production.yaml
```

Use `--max-steps 2` for an accelerator smoke. Stages `1`, `2`, and `4` cover
frozen, partial, and full image-text training. Stages `5` through `7` retain
the audio-text continuation workflow; use cached `input_features` shards and
set `data.feature_key` to `input_features` in a reviewed run config.

## Evaluate and package

```bash
python -m training.model_embeddings.multimodal.small.evaluate \
  --config src/training/model_embeddings/multimodal/small/configs/production.yaml \
  --checkpoint "$MM_EMBED_SMALL_OUTPUT_DIR/best" \
  --cache "$MM_EMBED_SMALL_VAL_CACHE" \
  --output "$MM_EMBED_SMALL_OUTPUT_DIR/evaluation.json"

python -m training.model_embeddings.multimodal.small.release \
  --checkpoint "$MM_EMBED_SMALL_OUTPUT_DIR/best" \
  --output-dir /path/to/release
```

`release.py` prepares and checksums locally by default. Upload is opt-in, and
an existing Hugging Face artifact is protected unless `--allow-existing` is
given explicitly.

## Reproducible runs

For every release-quality run, retain the resolved config, encoder revisions,
cached-shard manifest, dependency versions, metrics, and input/output
checksums. These records distinguish a new trained checkpoint from an existing
published artifact and make later evaluation comparable.
