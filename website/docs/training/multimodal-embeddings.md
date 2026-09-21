---
title: Multimodal Embedding Models
sidebar_label: Multimodal Embeddings
---

# Multimodal embedding models

The multimodal embedders map text, images, and audio into a shared vector space.
Once aligned, the router can compare inputs across modalities with the same
similarity operation used for text retrieval.

Choose the small model when memory and latency are the primary constraints.
Choose the large model when representation capacity, long text, and stronger
vision/audio towers justify a larger serving footprint.

## Architecture comparison

| Component | Small | Large |
| --- | --- | --- |
| Text tower | `all-MiniLM-L6-v2` | mmBERT-32K 2D Matryoshka embedder |
| Image tower | SigLIP base, patch 16, 512-pixel input | SigLIP2 SO400M, patch 14, 384-pixel input |
| Audio tower | Whisper tiny encoder | Whisper medium encoder |
| Cross-modal layer | Two-layer Transformer fusion | Independent towers projected into one space |
| Output | Normalized 384-dimensional vector | Normalized 768-dimensional vector |
| Text limit | 128 tokens in the checked production config | 32,768 tokens |
| Primary loss | Matryoshka-wrapped contrastive alignment | Cached multiple-negatives ranking loss |

Both models use modality-specific encoders because pixels, waveforms, and text
tokens need different front ends. Projection and alignment training make the
resulting representations comparable.

## Small model

[`multi-modal-embed-small`](https://huggingface.co/llm-semantic-router/multi-modal-embed-small)
combines compact pretrained towers with a two-layer fusion Transformer. Its
normalized 384-dimensional output is supervised at dimensions 32, 64, 128,
256, and 384 so deployments can trade vector size for quality.

Training is staged to avoid destabilizing every encoder at once:

1. Train projection and fusion layers while the pretrained towers are frozen.
2. Unfreeze selected upper layers for partial adaptation.
3. Fine-tune the full image-text path when the aligned representation is
   stable.
4. Continue with audio-text alignment using cached Whisper input features.

The checked Stage 1 configuration uses cached LLaVA-CC3M image-text examples,
six epochs, batch 64 per process, learning rate `1e-4`, mixed precision,
temperature `0.07`, and Matryoshka contrastive loss. Stages 2, 4, and 5–7
change which towers are trainable and which modality pair is sampled; review
the stage configuration before launching them.

Run from the repository root:

```bash
python -m pip install --requirement \
  src/training/model_embeddings/multimodal/small/requirements.txt

export PYTHONPATH="$PWD/src"
export MM_EMBED_SMALL_TRAIN_CACHE=/path/to/cache/train
export MM_EMBED_SMALL_VAL_CACHE=/path/to/cache/validation
export MM_EMBED_SMALL_OUTPUT_DIR=/path/to/output

python -m training.model_embeddings.multimodal.small.train \
  --config src/training/model_embeddings/multimodal/small/configs/production.yaml \
  --print-config
```

After inspecting the resolved paths and stage, remove `--print-config` and use
the distributed launcher appropriate for your environment. `--max-steps 2`
provides a short accelerator smoke run.

## Large model

[`multi-modal-embed-large`](https://huggingface.co/llm-semantic-router/multi-modal-embed-large)
uses the long-context mmBERT embedder, a larger SigLIP2 vision tower, and a
larger Whisper audio tower. Each tower is projected into a shared
768-dimensional space. Unlike the small fusion architecture, the production
tri-encoder keeps modality encoding independent so cached embeddings and
pairwise contrastive training remain straightforward.

Raw examples are preprocessed into validated tensor shards. Training loads
those shards sequentially with bounded prefetch and uses cached
multiple-negatives ranking loss: matching pairs are positives, other examples
in the effective batch are negatives, and configured hard negatives make the
boundary more informative.

The production configuration uses 10 epochs, per-device batch 12, gradient
accumulation 8, learning rate `1e-5`, BF16, loss scale 20, and a 50% hard-
negative ratio.

```bash
python -m pip install --requirement \
  src/training/model_embeddings/multimodal/large/requirements.txt

export PYTHONPATH="$PWD/src"
export MM_EMBED_LARGE_OUTPUT_DIR=/path/to/output
export MM_EMBED_LARGE_TRAIN_CACHE=/path/to/cache/train
export MM_EMBED_LARGE_VAL_CACHE=/path/to/cache/validation

python -m training.model_embeddings.multimodal.large.train \
  --config src/training/model_embeddings/multimodal/large/configs/production.yaml
```

The [workflow README](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_embeddings/multimodal/large)
documents preprocessing, smoke configurations, evaluation, and packaging.

## Evaluate cross-modal quality

Evaluate every modality pair you intend to serve: text-image, image-text,
text-audio, and audio-text. Report Recall@k in both directions, plus per-domain
and per-language slices. For the small model, repeat the evaluation at every
supported Matryoshka dimension. Also test same-modality retrieval if your
application relies on it; cross-modal alignment does not automatically prove
strong image-image or audio-audio retrieval.

Keep the preprocessing contract with the artifact. Image resizing, audio
feature extraction, text truncation, normalization, and output dimension are
part of the model, not interchangeable serving details.
