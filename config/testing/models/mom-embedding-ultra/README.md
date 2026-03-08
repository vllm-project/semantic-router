---
library_name: sentence-transformers
tags:
  - sentence-transformers
  - sentence-similarity
  - feature-extraction
  - embeddings
  - multilingual
  - matryoshka
  - 2d-matryoshka
  - long-context
  - modernbert
base_model: llm-semantic-router/mmbert-32k-yarn
datasets:
  - BAAI/bge-m3-data
language:
  - multilingual
license: apache-2.0
pipeline_tag: sentence-similarity
model-index:
  - name: mmbert-embed-32k-2d-matryoshka
    results:
      - task:
          type: STS
        dataset:
          name: STS Benchmark
          type: mteb/stsbenchmark-sts
        metrics:
          - type: spearman
            value: 80.5
---

# mmBERT-Embed-32K-2D-Matryoshka

A **multilingual embedding model** with 32K context window and **2D Matryoshka** support for flexible efficiency-quality tradeoffs.

## Model Highlights

| Feature | Value |
|---------|-------|
| **Parameters** | 307M |
| **Context Length** | 32,768 tokens |
| **Languages** | 1800+ (via Glot500) |
| **Embedding Dim** | 768 (supports 64-768 via Matryoshka) |
| **Architecture** | ModernBERT encoder with YaRN scaling |

### Key Results

| Metric | Score |
|--------|-------|
| **MTEB Mean (24 tasks)** | **61.4** |
| **STS Benchmark** | **80.5** (exceeds Qwen3-0.6B's 76.17) |
| **Dimension Retention** | 99% @ 256d, 98% @ 64d |
| **Layer Speedup** | 3.3× @ 6L, 5.8× @ 3L |
| **Latency vs BGE-M3** | **1.6-3.1× faster** (FA2 advantage) |

## What is 2D Matryoshka?

This model supports **two dimensions of flexibility**:

1. **Dimension Reduction** (Matryoshka): Truncate embeddings to smaller dimensions with minimal quality loss
2. **Layer Reduction** (Adaptive): Use intermediate layer outputs for faster inference

| Config | Quality | Speedup | Storage |
|--------|---------|---------|---------|
| 22L, 768d | 100% | 1.0× | 100% |
| 22L, 256d | 99% | 1.0× | 33% |
| 22L, 64d | 98% | 1.0× | 8% |
| 6L, 768d | 56% | 3.3× | 100% |
| 6L, 256d | 56% | 3.3× | 33% |

## Usage

### Basic Usage (Sentence Transformers)

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("llm-semantic-router/mmbert-embed-32k-2d-matryoshka")

# Encode sentences
sentences = [
    "This is a test sentence.",
    "这是一个测试句子。",
    "Dies ist ein Testsatz.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 768)
```

### Matryoshka Dimension Reduction

```python
import torch.nn.functional as F

# Encode with full dimensions
embeddings = model.encode(sentences, convert_to_tensor=True)

# Truncate to smaller dimension (e.g., 256)
embeddings_256d = embeddings[:, :256]
embeddings_256d = F.normalize(embeddings_256d, p=2, dim=1)

# Or truncate to 64 dimensions for maximum compression
embeddings_64d = embeddings[:, :64]
embeddings_64d = F.normalize(embeddings_64d, p=2, dim=1)
```

### Long Context (up to 32K tokens)

```python
# For long documents, set max_seq_length
model.max_seq_length = 8192  # or up to 32768

long_document = "..." * 10000  # Very long text
embedding = model.encode(long_document)
```

### Layer Reduction (Advanced)

For latency-critical applications, you can extract embeddings from intermediate layers:

```python
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

model = AutoModel.from_pretrained(
    "llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
    trust_remote_code=True,
    output_hidden_states=True
)
tokenizer = AutoTokenizer.from_pretrained("llm-semantic-router/mmbert-embed-32k-2d-matryoshka")

# Encode
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    
    # Use layer 6 for 3.3× speedup (56% quality)
    hidden = outputs.hidden_states[6]
    hidden = model.final_norm(hidden)
    
    # Mean pooling
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1)
    embeddings = F.normalize(pooled, p=2, dim=1)
```

## Evaluation Results

### MTEB Benchmark (24 tasks)

| Category | Score |
|----------|-------|
| STS (7 tasks) | **79.3** |
| Classification (6) | 62.4 |
| Pair Classification (2) | 76.2 |
| Reranking (2) | 64.4 |
| Clustering (4) | 36.9 |
| Retrieval (3) | 38.2 |
| **Overall Mean** | **61.4** |

### STS Benchmark

| Model | Parameters | STS Score |
|-------|------------|-----------|
| Qwen3-Embed-0.6B | 600M | 76.17 |
| **mmBERT-Embed** | **307M** | **80.5** |
| Qwen3-Embed-8B | 8B | 81.08 |

### 2D Matryoshka Quality Matrix (STS)

| Layers | 768d | 256d | 64d |
|--------|------|------|-----|
| 22L | **80.5** | 79.9 | 78.5 |
| 11L | 53.7 | 48.0 | 44.4 |
| 6L | 45.2 | 45.2 | 43.5 |
| 3L | 44.0 | 44.1 | 41.8 |

### Long-Context Retrieval (4K tokens)

| Metric | Score |
|--------|-------|
| R@1 | 68.8% |
| R@10 | 81.2% |
| MRR | 71.9% |

### Throughput (AMD MI300X)

| Layers | Throughput | Speedup |
|--------|------------|---------|
| 22L | 477/s | 1.0× |
| 11L | 916/s | 1.9× |
| 6L | 1573/s | 3.3× |
| 3L | 2761/s | 5.8× |

### Latency Comparison vs BGE-M3 and Qwen3-Embedding-0.6B

mmBERT-Embed is significantly faster due to:

1. **Flash Attention 2** - BGE-M3 lacks FA2 (O(n) vs O(n²) attention)
2. **Encoder architecture** - Qwen3 uses decoder with causal masking
3. **Smaller model** - 307M vs 569M/600M params

#### Batch Size = 1

| Seq Len | mmBERT-Embed | Qwen3-0.6B | BGE-M3 | mmBERT Speedup |
|---------|--------------|------------|--------|----------------|
| 512 | **17.6ms** (57/s) | 20.7ms (48/s) | 10.8ms (93/s) | 0.6× |
| 1024 | **18.6ms** (54/s) | 21.2ms (47/s) | 16.3ms (61/s) | 0.9× |
| 2048 | **19.5ms** (51/s) | 24.1ms (42/s) | 31.1ms (32/s) | **1.6×** |
| 4096 | **21.3ms** (47/s) | 43.5ms (23/s) | 60.5ms (17/s) | **2.8×** |

#### Batch Size = 8

| Seq Len | mmBERT-Embed | Qwen3-0.6B | BGE-M3 | mmBERT Speedup |
|---------|--------------|------------|--------|----------------|
| 512 | **21.1ms** (379/s) | 33.0ms (243/s) | 40.0ms (200/s) | **1.9×** |
| 1024 | **34.5ms** (232/s) | 58.5ms (137/s) | 77.4ms (103/s) | **2.2×** |
| 2048 | **65.2ms** (123/s) | 117.0ms (68/s) | 162.9ms (49/s) | **2.5×** |
| 4096 | **130.7ms** (61/s) | 254.9ms (31/s) | 411.3ms (19/s) | **3.1×** |

**Key insight**: The FA2 advantage grows with sequence length and batch size:

- At short sequences (512), BGE-M3 is faster (no FA2 overhead)
- At 2K+ tokens, mmBERT pulls ahead significantly
- At 4K batch=8: mmBERT is **3.1× faster** than BGE-M3

*Benchmarked on AMD MI300X, bf16 precision.*

## Training

### Data

Trained on [BAAI/bge-m3-data](https://huggingface.co/datasets/BAAI/bge-m3-data) (73GB, 279 JSONL files) with:

- Multilingual triplets (query, positive, negative)
- Diverse domains and languages

### Configuration

- **Base Model**: [llm-semantic-router/mmbert-32k-yarn](https://huggingface.co/llm-semantic-router/mmbert-32k-yarn)
- **Loss**: Matryoshka2dLoss (combines AdaptiveLayerLoss + MatryoshkaLoss)
- **Matryoshka Dimensions**: [768, 512, 256, 128, 64]
- **Epochs**: 1
- **Batch Size**: 16 (effective 32 with gradient accumulation)
- **Learning Rate**: 2e-5
- **Max Sequence Length**: 32,768
- **Hardware**: AMD Instinct MI300X

## Use Cases

### When to Use mmBERT-Embed

1. **Multilingual RAG** for 1800+ languages (especially low-resource languages not covered by Qwen3 or BGE-M3)
2. **Long-document retrieval** where chunking loses cross-section relationships
3. **Edge deployment** where 307M params matters vs 600M+
4. **Flexible inference** where you need to trade quality for speed/storage at runtime

### When to Use Alternatives

- **Maximum quality on major languages**: Qwen3-Embed-8B
- **Production stability**: BGE-M3 (more battle-tested)
- **Very short texts only**: Smaller models may suffice

## Limitations

- Layer reduction quality (56% at 6L) is lower than full model; use for latency-critical applications where moderate quality is acceptable
- MTEB mean (61.4) is slightly below BGE-M3 (64.5) but with 4× longer context and 2D flexibility
- Optimized for retrieval tasks; may need fine-tuning for other downstream tasks

## Citation

```bibtex
@misc{mmbert-embed-2d-matryoshka,
  title={mmBERT-Embed: Multilingual Embedding Model with 2D Matryoshka Training},
  author={vLLM Semantic Router Team},
  year={2025},
  url={https://huggingface.co/llm-semantic-router/mmbert-embed-32k-2d-matryoshka}
}
```

## ONNX Models for Production Deployment

Pre-exported ONNX models are available for production deployment with ONNX Runtime. Each layer model enables true early-exit speedup.

### Available Models

| Layer | Size | Latency | Throughput | Speedup | Quality |
|-------|------|---------|------------|---------|---------|
| `onnx/layer-6` | 454 MB | **2.56ms** | **390/sec** | **4.44×** | ~56% |
| `onnx/layer-11` | 505 MB | 4.87ms | 205/sec | 2.33× | ~75% |
| `onnx/layer-16` | 555 MB | 7.64ms | 131/sec | 1.49× | ~90% |
| `onnx/layer-22` | 616 MB | 11.37ms | 88/sec | 1.0× | 100% |

*Benchmarked on AMD MI300X with ROCm, fp16 precision, batch=1, dynamic sequence length.*

### Batch Performance (batch=8)

| Layer | Throughput | Speedup |
|-------|------------|---------|
| 6 | **634/sec** | **2.97×** |
| 11 | 428/sec | 2.00× |
| 16 | 286/sec | 1.34× |
| 22 | 214/sec | 1.0× |

### Download ONNX Models

```python
from huggingface_hub import hf_hub_download

# Download layer-6 for fast inference
model_path = hf_hub_download(
    "llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
    "onnx/layer-6/model.onnx"
)
data_path = hf_hub_download(
    "llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
    "onnx/layer-6/model.onnx.data"
)
```

### Usage with ONNX Runtime (Python)

```python
import onnxruntime as ort
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "llm-semantic-router/mmbert-embed-32k-2d-matryoshka"
)

# Load ONNX model (use ROCMExecutionProvider for AMD GPU)
session = ort.InferenceSession(
    "onnx/layer-6/model.onnx",
    providers=["ROCMExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Inference
inputs = tokenizer("Hello world", return_tensors="np", padding=True)
outputs = session.run(None, {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
})
hidden_state = outputs[0]  # Shape: (batch, seq_len, 768)

# Mean pooling
import numpy as np
mask = inputs["attention_mask"][..., np.newaxis]
embeddings = (hidden_state * mask).sum(1) / mask.sum(1)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
```

### Usage with Rust (ort crate)

```rust
use ort::{Session, execution_providers::ROCmExecutionProvider};

// Load models at startup
let fast_model = Session::builder()?
    .with_execution_providers([ROCmExecutionProvider::default().build()])?
    .commit_from_file("onnx/layer-6/model.onnx")?;

let full_model = Session::builder()?
    .with_execution_providers([ROCmExecutionProvider::default().build()])?
    .commit_from_file("onnx/layer-22/model.onnx")?;

// Runtime selection based on latency/quality needs
let embedding = if need_fast_response {
    fast_model.run(inputs)?   // ~2.6ms
} else {
    full_model.run(inputs)?   // ~11ms
};
```

### Recommended Layer Selection

| Use Case | Layer | Why |
|----------|-------|-----|
| Real-time routing/classification | 6 | Lowest latency (2.56ms) |
| Balanced speed/quality | 11 | Good tradeoff (4.87ms) |
| High accuracy tasks | 16 | Near-full quality (7.64ms) |
| Search/RAG | 22 | Maximum quality (11.37ms) |

### Why Separate ONNX Models?

Unlike PyTorch where you can use `output_hidden_states=True` for runtime layer selection, ONNX graphs are **static DAGs** - all nodes execute regardless of which output you read. Separate model files are required for true early-exit speedup.

## License

Apache 2.0
