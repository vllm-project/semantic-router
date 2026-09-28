---
license: apache-2.0
language:
- en
library_name: transformers
tags:
- sentence-transformers
- multimodal
- embeddings
- image-text
- audio-text
- retrieval
- 2DMSE
- matryoshka
pipeline_tag: sentence-similarity
model-index:
- name: multi-modal-embed-small
  results:
  - task:
      type: image-text-retrieval
    dataset:
      name: COCO
      type: coco
    metrics:
    - name: Image-to-Text R@1
      type: recall_at_1
      value: 41.88
    - name: Image-to-Text R@5
      type: recall_at_5
      value: 71.64
    - name: Image-to-Text R@10
      type: recall_at_10
      value: 82.16
  - task:
      type: audio-text-retrieval
    dataset:
      name: LibriSpeech
      type: librispeech
    metrics:
    - name: Audio-to-Text R@1
      type: recall_at_1
      value: 36.38
    - name: Audio-to-Text R@5
      type: recall_at_5
      value: 68.22
    - name: Audio-to-Text R@10
      type: recall_at_10
      value: 79.52
---

# multi-modal-embed-small

A compact multimodal embedding model that unifies text, image, and audio representations in a shared semantic space. Part of the [MoM (Mixture of Models)](https://huggingface.co/llm-semantic-router) family.

## Model Description

**multi-modal-embed-small** is a lightweight multimodal encoder (~120M parameters) supporting:

- **Text encoding** via MiniLM-L6-v2 (22M params)
- **Image encoding** via SigLIP-base-patch16-512 (86M params)
- **Audio encoding** via Whisper-tiny encoder (8M params)
- **Cross-modal fusion** via 2-layer transformer attention
- **2DMSE**: Two-Dimensional Matryoshka Sentence Embeddings for adaptive compute
- **MRL**: Matryoshka Representation Learning for flexible embedding dimensions

### Key Features

| Feature | Description |
|---------|-------------|
| **Embedding Dimension** | 384 (supports MRL truncation to 32, 64, 128, 256) |
| **Image Resolution** | 512×512 |
| **Audio Input** | Up to 30s, 16kHz (Whisper Mel spectrogram) |
| **Modalities** | Text, Image, Audio, Multimodal fusion |
| **2DMSE Support** | Early exit at any encoder layer |
| **Languages** | English |

## Installation

```bash
pip install torch transformers pillow safetensors
```

## Usage

### Load Model

Two checkpoint formats are available:

- `model.pt` (932 MB) - PyTorch format
- `model.safetensors` (1.35 GB) - SafeTensors format

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, SiglipModel, SiglipProcessor, WhisperModel, WhisperFeatureExtractor
from huggingface_hub import hf_hub_download

class MultiModalEmbedder(nn.Module):
    """Standalone multimodal embedder - no external dependencies."""

    def __init__(self):
        super().__init__()
        # Text encoder (384d, no projection needed)
        self.text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.text_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        # Image encoder (768d -> 384d projection)
        self.image_processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-512")
        self.image_encoder = SiglipModel.from_pretrained("google/siglip-base-patch16-512").vision_model
        self.image_proj = nn.Linear(768, 384)

        # Audio encoder (384d, no projection needed)
        self.audio_processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
        self.audio_encoder = WhisperModel.from_pretrained("openai/whisper-tiny").encoder

    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.text_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return F.normalize(embeddings, p=2, dim=-1)

    def encode_image(self, images):
        inputs = self.image_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        outputs = self.image_encoder(**inputs)
        embeddings = outputs.pooler_output
        embeddings = self.image_proj(embeddings)  # 768 -> 384
        return F.normalize(embeddings, p=2, dim=-1)

    def encode_audio(self, waveform):
        # waveform: numpy array or tensor at 16kHz
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.squeeze().numpy()
        inputs = self.audio_processor(waveform, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        outputs = self.audio_encoder(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return F.normalize(embeddings, p=2, dim=-1)

# Load model
model = MultiModalEmbedder()

# Download and load trained weights
checkpoint_path = hf_hub_download(
    repo_id="llm-semantic-router/multi-modal-embed-small",
    filename="model.pt"
)
state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# Load text encoder weights
model.text_encoder.load_state_dict({
    k.replace("text_encoder.encoder.", ""): v
    for k, v in state_dict.items()
    if k.startswith("text_encoder.encoder.")
})

# Load image encoder and projection weights
model.image_encoder.load_state_dict({
    k.replace("image_encoder.vision_encoder.", ""): v
    for k, v in state_dict.items()
    if k.startswith("image_encoder.vision_encoder.")
})
model.image_proj.load_state_dict({
    k.replace("image_encoder.projection.", ""): v
    for k, v in state_dict.items()
    if k.startswith("image_encoder.projection.")
})

# Load audio encoder weights
model.audio_encoder.load_state_dict({
    k.replace("audio_encoder.encoder.", ""): v
    for k, v in state_dict.items()
    if k.startswith("audio_encoder.encoder.")
})

model.eval()
print("Model loaded successfully!")
```

### Text Embedding

```python
import torch.nn.functional as F

# Single text
text_embedding = model.encode_text("A photo of a cat")  # Shape: [1, 384]

# Batch of texts
texts = ["A fluffy orange cat", "A golden retriever dog", "A red sports car"]
text_embeddings = model.encode_text(texts)  # Shape: [3, 384]

# Compute similarity
similarities = F.cosine_similarity(text_embeddings[0:1], text_embeddings[1:], dim=-1)
print(f"Cat vs Dog: {similarities[0]:.3f}")
print(f"Cat vs Car: {similarities[1]:.3f}")
```

### Image Embedding

```python
from PIL import Image
import requests
from io import BytesIO

# Load image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
image = Image.open(BytesIO(requests.get(url).content)).convert('RGB')

# Get embedding
image_embedding = model.encode_image(image)  # Shape: [1, 384]
```

### Audio Embedding

```python
import torchaudio

# Load audio (16kHz)
waveform, sample_rate = torchaudio.load("speech.wav")
if sample_rate != 16000:
    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

# Get embedding
audio_embedding = model.encode_audio(waveform)  # Shape: [1, 384]
```

### Cross-Modal Retrieval

```python
# Image-to-text retrieval
image = Image.open("cat.jpg").convert('RGB')
image_emb = model.encode_image(image)

captions = [
    "A cat sleeping on a bed",
    "A dog playing in the park",
    "A car driving on the highway",
]
text_embs = model.encode_text(captions)

similarities = F.cosine_similarity(image_emb, text_embs)
best_idx = similarities.argmax().item()
print(f"Best match: {captions[best_idx]} ({similarities[best_idx]:.3f})")
```

### Matryoshka Dimension Reduction

```python
# Full 384-dim embedding
full_emb = model.encode_text("Hello world")  # [1, 384]

# Truncate to smaller dimensions
emb_256 = F.normalize(full_emb[:, :256], p=2, dim=-1)  # 1.5x faster retrieval
emb_128 = F.normalize(full_emb[:, :128], p=2, dim=-1)  # 3x faster retrieval
emb_64 = F.normalize(full_emb[:, :64], p=2, dim=-1)    # 6x faster retrieval
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  multi-modal-embed-small                     │
├──────────────────────────────────────────────────────────────┤
│  Text Encoder:  MiniLM-L6-v2           (22M params, 6 layers)│
│  Image Encoder: SigLIP-base-patch16-512 (86M params)         │
│  Audio Encoder: Whisper-tiny encoder    (8M params, 4 layers) │
│  Fusion:        2-layer Transformer                          │
├──────────────────────────────────────────────────────────────┤
│  Output: 384-dim normalized embeddings                       │
│  2DMSE:  Layer 0-5 early exit support                        │
│  MRL:    32, 64, 128, 256, 384 dim truncation                │
└──────────────────────────────────────────────────────────────┘
```

## Training

### Training Data

| Modality | Dataset | Samples | Purpose |
|----------|---------|---------|---------|
| Image-Text | LLaVA-CC3M | 595K | Image-text alignment |
| Image-Text | COCO Captions | 25K | Validation |
| Audio-Text | LibriSpeech | 105K | Audio-text alignment |

### Training Stages

| Stage | Description | Trainable | Epochs |
|-------|-------------|-----------|--------|
| 1 | Initial alignment | Projection layers only | 6 |
| 2 | Partial unfreeze | Top encoder layers + projections | 3 |
| 4 | Full image-text | All image/text parameters | 3 |
| 5 | Audio alignment | Audio encoder (text/image frozen) | 5 |

### Training Configuration

- **Hardware**: 8× AMD MI300X GPUs
- **Precision**: BF16 mixed precision
- **Batch Size**: 64 per GPU (512 effective)
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4 → 5e-5 (stage dependent)
- **Loss**: InfoNCE contrastive + Matryoshka loss

## Evaluation

### Image-Text Retrieval (COCO Validation)

| Metric | Image→Text | Text→Image |
|--------|------------|------------|
| R@1    | 41.88%     | 39.21%     |
| R@5    | 71.64%     | 69.15%     |
| R@10   | 82.16%     | 80.02%     |

### Audio-Text Retrieval (LibriSpeech)

| Metric | Audio→Text |
|--------|------------|
| R@1    | 36.38%     |
| R@5    | 68.22%     |
| R@10   | 79.52%     |

### MRL Quality Retention

| Dimension | Compression | Quality |
|-----------|-------------|---------|
| 384 (full)| 1×          | 100%    |
| 256       | 1.5×        | ~98%    |
| 128       | 3×          | ~95%    |
| 64        | 6×          | ~90%    |

## Limitations

- English language only
- Image resolution fixed at 512×512
- Audio limited to 30 seconds
- Best for retrieval/similarity, not generation

## Citation

```bibtex
@misc{multi-modal-embed-small-2026,
  title={multi-modal-embed-small: Compact Multimodal Embeddings with 2DMSE},
  author={Semantic Router Team},
  year={2026},
  url={https://huggingface.co/llm-semantic-router/multi-modal-embed-small}
}
```

## License

Apache 2.0
