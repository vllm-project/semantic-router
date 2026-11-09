# Large multimodal embedding training

This package trains and packages
[`vllm-sr/multi-modal-embed-large`](https://huggingface.co/vllm-sr/multi-modal-embed-large).
`artifacts.json` maps the public artifact to its local config and entrypoints.

The released artifact and `configs/production.yaml` use the same tri-encoder:

- text: `vllm-sr/mmbert-embed-32k-2d-matryoshka`;
- image: `google/siglip2-so400m-patch14-384`;
- audio: `openai/whisper-medium`;
- shared output: 768 dimensions, with text up to 32,768 tokens.

The production path uses cached tensor shards, Accelerate, BF16, sequential
shard loading, and symmetric multiple-negatives ranking loss. The alternative
native path is retained for Sentence Transformers-compatible models and small
smoke runs.

## Layout

| File | Responsibility |
| --- | --- |
| `train.py` | small CLI and native-path orchestration |
| `native_training.py` | Sentence Transformers model, loss, collator, and trainer |
| `tri_encoder.py` | cached dataset/model/checkpoint/evaluation components |
| `tri_encoder_training.py` | tri-encoder optimizer and epoch orchestration |
| `records.py`, `cached_data.py` | manifest records and bounded cached-shard loading |
| `preprocess.py`, `download_data.py` | data acquisition and cache materialization |
| `evaluate.py`, `upload.py`, `watch.py` | evaluation and gated publication |

## Install

This directory is a uv project (`pyproject.toml` and `uv.lock`). `pyproject.toml`
bounds each dependency and `uv.lock` pins the exact versions. Create the
environment from the lock:

```bash
uv sync --locked --project src/training/model_embeddings/multimodal/large
```

`requirements.txt` stays for anyone who installs with pip. That file sends pip
at the ROCm 7.1 wheel index. The uv project records the same index as the
`rocm` extra, and also records a CPU build and the default PyPI build. Pick
one accelerator:

```bash
uv sync --locked --project src/training/model_embeddings/multimodal/large --extra cpu
uv sync --locked --project src/training/model_embeddings/multimodal/large --extra cuda
uv sync --locked --project src/training/model_embeddings/multimodal/large --extra rocm
```

`cpu` installs wheels from `https://download.pytorch.org/whl/cpu`. `cuda`
installs the default PyPI build, which is the CUDA build on Linux. `rocm`
installs wheels from `https://download.pytorch.org/whl/rocm7.1` and applies on
Linux. The three extras are mutually exclusive.

`download_data.py` needs the `download` group. Upload and safetensors
evaluation need the `publish` group:

```bash
uv sync --locked --project src/training/model_embeddings/multimodal/large \
  --extra rocm --group download --group publish
```

The contract tests run from the repository root through `uv run --project`:

```bash
uv run --project src/training/model_embeddings/multimodal/large \
  python -m unittest discover \
  -s src/training/model_embeddings/multimodal/large/tests -p 'test_*.py'
```

## Production command

Set all storage locations explicitly and run from the repository root:

```bash
export PYTHONPATH="$PWD/src"
export MM_EMBED_LARGE_OUTPUT_DIR=/path/to/output
export MM_EMBED_LARGE_TRAIN_CACHE=/path/to/cache/train
export MM_EMBED_LARGE_VAL_CACHE=/path/to/cache/val

uv run --project src/training/model_embeddings/multimodal/large \
  python -m training.model_embeddings.multimodal.large.train \
  --config src/training/model_embeddings/multimodal/large/configs/production.yaml
```

The checked production config uses
10 epochs, per-device batch 12, accumulation 8, learning rate `1e-5`, BF16,
and loss scale 20.

For raw-manifest smoke tests, set `MM_EMBED_LARGE_DATA_ROOT` and use one of the
`native-*-smoke.yaml` configs. Data download, preprocessing, evaluation, and
upload are independently callable with `python -m` and `--help`.

## Reproducible runs

For every release-quality run, record immutable revisions for all three
encoders, the resolved production config, cached dataset manifest and checksums,
dependency versions, metrics, and output checksums. Encoder names alone are not
enough to reproduce a checkpoint after any upstream model changes.
