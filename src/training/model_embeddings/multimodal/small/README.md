# Small multimodal embedding training

This package trains and packages
[`vllm-sr/multi-modal-embed-small`](https://huggingface.co/vllm-sr/multi-modal-embed-small).
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

## Install

This directory is a uv project (`pyproject.toml` and `uv.lock`).
`pyproject.toml` bounds each dependency and `uv.lock` pins the exact versions.
Create the environment from the lock:

```bash
uv sync --locked --project src/training/model_embeddings/multimodal/small
```

That install covers the contract tests and the libraries the modules import.
It does not install a torch build.

`requirements.txt` stays for anyone who installs with pip. That file sends pip
at the ROCm 7.1 wheel index. The uv project records the same index as the
`rocm` extra, and also records a CPU build and the default PyPI build. Pick
one accelerator. `accelerate` is installed with the extra, because it depends
on torch and the modules do not import it themselves:

```bash
uv sync --locked --project src/training/model_embeddings/multimodal/small --extra cpu
uv sync --locked --project src/training/model_embeddings/multimodal/small --extra cuda
uv sync --locked --project src/training/model_embeddings/multimodal/small --extra rocm
```

`cpu` installs wheels from `https://download.pytorch.org/whl/cpu`. `cuda`
installs the default PyPI build, which is the CUDA build on Linux. `rocm`
installs wheels from `https://download.pytorch.org/whl/rocm7.1` and applies on
Linux. The three extras are mutually exclusive. Each one resolves
`torch>=2.10.0,<2.11`, `torchvision>=0.25.0,<0.26`, and
`torchaudio>=2.10.0,<2.11`, the same lines as `requirements.txt`.

`download_data.py` and `release.py --upload` need the `hub` group:

```bash
uv sync --locked --project src/training/model_embeddings/multimodal/small \
  --extra rocm --group hub
```

The contract tests run from the repository root through `uv run --project`:

```bash
uv run --project src/training/model_embeddings/multimodal/small \
  python -m unittest discover \
  -s src/training/model_embeddings/multimodal/small/tests -p 'test_*.py'
```

## Train

After the uv install above, set all storage locations. `requirements.txt`
remains available for a pip install:

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
