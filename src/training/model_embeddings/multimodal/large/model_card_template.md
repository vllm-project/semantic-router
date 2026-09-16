---
license: apache-2.0
library_name: pytorch
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- multimodal
- embeddings
- retrieval
- image-text
- audio-text
- text-image-audio
- tri-encoder
- semantic-router
- pytorch
model-index:
- name: __MODEL_NAME__
  results:
  - task:
      type: sentence-similarity
    dataset:
      name: Internal cached validation set
      type: cached_retrieval_validation
    metrics:
    - name: Eval loss
      type: eval_loss
      value: __EVAL_LOSS__
    - name: Eval top1
      type: eval_top1
      value: __EVAL_TOP1__
---

# __MODEL_NAME__

`__MODEL_NAME__` is the large production multimodal embedding model from the [llm-semantic-router](https://huggingface.co/llm-semantic-router) project.

It is designed for routing, retrieval, and cross-modal matching across text, image, and audio rather than for generative chat. The model uses a tri-encoder architecture with separate text, image, and audio towers projected into one shared embedding space.

## Purpose

This release exists to provide a large multimodal embedding model for production systems where inputs may arrive as text, screenshots or images, and audio. It is built for semantic routing, multimodal retrieval, and cross-modal similarity.

## What Is In This Repository

This repository contains the minimum artifacts needed to load and run the exported model:

- `model.pt`: trained weights for the final exported model
- `config.json`: model configuration and encoder names
- `src/hf_st_mm/...`: the Python source package used to construct and run the tri-encoder
- `README.md`: this model card, including usage examples and validation summary

This is not a generic Hugging Face Transformers checkpoint with a built-in auto-class loader. It is a packaged custom PyTorch model export.

## Advantages And Innovation

Most multimodal models are optimized for generation, captioning, or chat. This model is optimized for embeddings and operational use.

What is different here:

- map text, image, and audio into one shared semantic space
- support routing and retrieval instead of text generation
- preserve a strong multilingual text backbone
- use stronger modality-specific encoders instead of forcing every modality into one monolithic checkpoint
- support production training and evaluation on cached shard datasets

## Model Overview

This release packages the large routing-grade tri-encoder trained in PyTorch with the server training stack from this project.

Architecture:

- text encoder: `__TEXT_ENCODER_NAME__`
- image encoder: `__IMAGE_ENCODER_NAME__`
- audio encoder: `__AUDIO_ENCODER_NAME__`
- shared embedding dimension: `__EMBEDDING_DIM__`
- max text length: `__MAX_TEXT_LENGTH__`

Training characteristics:

- objective: cached multiple negatives ranking loss
- training stack: PyTorch + Accelerate
- target hardware: AMD MI300X
- data pipeline: cached tensor shards with sequential shard loading and worker-local prefetch

## How To Use It

## Installation

```bash
pip install torch sentence-transformers transformers accelerate safetensors pillow librosa soundfile huggingface_hub
```

## Python Usage

The simplest way to use the model is to download the repository snapshot, load the packaged source code, and then encode one or more modality-tagged items.

```python
import json
import os
import sys

import torch
from huggingface_hub import snapshot_download

repo_id = "__REPO_ID__"
local_dir = snapshot_download(repo_id=repo_id)

sys.path.insert(0, os.path.join(local_dir, "src"))

from hf_st_mm.data import PairItem
from hf_st_mm.model import MultiModalSentenceEmbedder

with open(os.path.join(local_dir, "config.json"), "r", encoding="utf-8") as handle:
    cfg = json.load(handle)

model = MultiModalSentenceEmbedder(
    text_encoder_name=cfg["model"]["text_encoder_name"],
    image_encoder_name=cfg["model"]["image_encoder_name"],
    audio_encoder_name=cfg["model"]["audio_encoder_name"],
    embedding_dim=int(cfg["model"]["embedding_dim"]),
    max_text_length=int(cfg["model"]["max_text_length"]),
)
state_dict = torch.load(os.path.join(local_dir, "model.pt"), map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

items = [
  PairItem(modality="text", value="route this request to the billing team"),
    PairItem(modality="image", value="/path/to/screenshot.png"),
    PairItem(modality="audio", value="/path/to/call.wav"),
]

with torch.no_grad():
    embeddings = model.encode_items(items)

print(embeddings.shape)  # [3, __EMBEDDING_DIM__]

import torch.nn.functional as F

query = PairItem(modality="text", value="refund request for wrong charge")
candidate = PairItem(modality="audio", value="/path/to/refund_call.wav")

with torch.no_grad():
    embs = model.encode_items([query, candidate])

similarity = F.cosine_similarity(embs[0:1], embs[1:2]).item()
print(f"similarity={similarity:.4f}")
```

## Validation Snapshot

At upload time, the final export was evaluated with the repository's tri-encoder evaluator.

- `eval_loss`: `__EVAL_LOSS__`
- `eval_top1`: `__EVAL_TOP1__`

## Practical Notes

- Text inputs can be provided as raw strings or tokenized features.
- Image and audio inputs can be provided as file paths.
- Cached tensor payloads are supported by the training stack, but the simplest inference path is to use file paths or raw text.
- This release is intended for production retrieval and routing use cases rather than for instruction-following or caption generation.

## Limitations

- This is a custom tri-encoder export, not a standard Transformers auto-class package.
- Inference currently relies on the packaged `hf_st_mm` source code.
- The validation metrics reported here come from the repository's cached retrieval validation path, not from a public benchmark leaderboard.

## Training Code

Training and evaluation code live in the Semantic Router repository.

- trainer: `src/training/model_embeddings/multimodal/large/train.py`
- evaluator: `src/training/model_embeddings/multimodal/large/evaluate.py`
- model: `src/training/model_embeddings/multimodal/large/model.py`
