# ONNX Binding for Semantic Router

This module provides ONNX Runtime-based embedding generation with 2D Matryoshka support. **CPU inference is highly optimized** and matches GPU performance for this model size, making it ideal for deployment without GPU dependencies.

## Features

- **Optimized CPU Inference**: ONNX Runtime's CPU backend (with AVX512/AVX2) matches GPU performance
- **No GPU Required**: Deploy anywhere without CUDA/ROCm dependencies
- **Optional GPU Support**: AMD ROCm and NVIDIA CUDA available when needed
- **Cross-platform**: Works on Linux, Windows, macOS
- **2D Matryoshka**: Layer early exit + dimension truncation for 3-4x speedup
- **32K Context**: Support for long sequences (32,768 tokens)
- **Multilingual**: 1800+ languages via mmBERT base model

## Why CPU?

Benchmarks show **CPU and GPU have equivalent performance** for this model:

| SeqLen | Batch | CPU (ms) | GPU (ms) | Speedup |
|--------|-------|----------|----------|---------|
| 64 | 1 | 7.78 | 7.79 | 1.00x |
| 128 | 1 | 14.28 | 14.19 | 1.01x |
| 256 | 1 | 26.43 | 26.34 | 1.00x |
| 512 | 1 | 57.01 | 56.26 | 1.01x |

**The real optimization is layer early-exit (3-4x speedup), not GPU acceleration.**

This makes CPU the preferred choice because:

- No GPU driver dependencies
- Simpler deployment (no CUDA/ROCm setup)
- Works in any container or VM
- Lower infrastructure cost

## 2D Matryoshka Support

The mmBERT model supports two dimensions of flexibility:

1. **Dimension Reduction** (Matryoshka): Truncate embeddings to smaller dimensions
   - Supported: 768, 512, 256, 128, 64

2. **Layer Early-Exit** (Adaptive): Use intermediate layer outputs for faster inference
   - Supported: 22 (full), 16, 11, 6 layers

### Layer Early-Exit Speedup (Actual Benchmarks)

**This is where the real performance gains come from:**

| Layer | SeqLen=64 | SeqLen=256 | SeqLen=512 | Speedup vs L22 |
|-------|-----------|------------|------------|----------------|
| 6 | 7.79ms | 26.34ms | 56.26ms | **3.6-3.9x** |
| 11 | 14.53ms | 47.78ms | 99.90ms | **2.0x** |
| 16 | 21.48ms | 68.84ms | 149.75ms | **1.3-1.5x** |
| 22 | 29.26ms | 104.22ms | 201.80ms | baseline |

### Throughput (samples/sec) - Batch=8

| SeqLen | Layer 6 | Layer 11 | Layer 16 | Layer 22 |
|--------|---------|----------|----------|----------|
| 32 | **330** | 183 | 124 | 92 |
| 64 | **192** | 108 | 73 | 54 |
| 128 | **113** | 63 | 43 | 31 |
| 256 | **59** | 33 | 22 | 17 |
| 512 | **21** | 13 | 9 | 7 |

### Quality vs Speed Tradeoffs

| Layer | Dimension | Quality | Speedup |
|-------|-----------|---------|---------|
| 22    | 768       | 100%    | 1x      |
| 11    | 512       | ~67%    | ~2x     |
| 6     | 256       | ~56%    | ~3.7x   |

## Tested Configurations

| Platform | Backend | glibc | Status |
|----------|---------|-------|--------|
| Linux (any) | CPU | 2.38+ | ✓ Recommended |
| `rocm/pytorch:latest` | AMD MI300X | 2.39 | ✓ Working |
| NVIDIA containers | CUDA | varies | ✓ Working |

## Quick Start: CPU (Recommended)

```bash
# Download ONNX models from HuggingFace
pip install huggingface_hub
huggingface-cli download llm-semantic-router/mmbert-embed-32k-2d-matryoshka --include "onnx/*" --local-dir ./mmbert-onnx

# Build for CPU (no GPU dependencies)
cargo build --release --example benchmark_cpu_vs_gpu

# Run benchmark
./target/release/examples/benchmark_cpu_vs_gpu ./mmbert-onnx/onnx
```

### Quick Start: GPU (Optional)

```bash
# Build with ROCm support
cargo build --release --features rocm --example test_gpu

# Run in ROCm-capable container (AMD GPU required)
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -v $(pwd):/workspace \
  rocm/pytorch:latest \
  /workspace/target/release/examples/test_gpu
```

## Installation

### Prerequisites

- Rust 1.88+ (for building)
- Go 1.23+ (for Go bindings)
- ONNX Runtime (automatically downloaded via `ort` crate)

### Building for CPU (Recommended)

```bash
cargo build --release
```

This is the recommended configuration. CPU performance equals GPU for this model size, with simpler deployment.

### Building with GPU Support (Optional)

```bash
# AMD GPU (ROCm)
cargo build --release --features rocm

# NVIDIA GPU (CUDA)
cargo build --release --features cuda
```

GPU support is optional - only use if you have specific infrastructure requirements.

## Usage

### Go Example

```go
package main

import (
    "fmt"
    "log"
    
    onnx "github.com/llm-d/semantic-router/onnx-binding"
)

func main() {
    // Initialize the model with CPU (recommended - matches GPU performance)
    useCPU := true
    err := onnx.InitMmBertEmbeddingModel("/path/to/mmbert-embed-32k-2d-matryoshka", useCPU)
    if err != nil {
        log.Fatal(err)
    }

    // Generate full embedding (768 dimensions) - ~30ms for 64 tokens
    output, err := onnx.GetEmbedding("Hello world")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Full embedding: %d dimensions, %.2fms\n", len(output.Embedding), output.ProcessingTimeMs)

    // Generate with 2D Matryoshka (6 layers, 256 dimensions) - ~8ms for 64 tokens (3.7x faster!)
    output, err = onnx.GetEmbedding2DMatryoshka("Hello world", 6, 256)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("2D Matryoshka: %d dimensions, %.2fms\n", len(output.Embedding), output.ProcessingTimeMs)

    // Calculate similarity
    result, err := onnx.CalculateEmbeddingSimilarity("Hello world", "Hi there", 0, 0)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Similarity: %.4f\n", result.Similarity)
}
```

### Rust Example

```rust
use onnx_semantic_router::model_architectures::embedding::mmbert_embedding::MmBertEmbeddingModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model
    let model = MmBertEmbeddingModel::load("/path/to/mmbert-embed-32k-2d-matryoshka", true)?;
    
    // Generate full embedding
    let embedding = model.encode_single("Hello world", None, None)?;
    println!("Full embedding: {} dimensions", embedding.len());
    
    // Generate with 2D Matryoshka
    let embedding = model.encode_single("Hello world", Some(6), Some(256))?;
    println!("2D Matryoshka: {} dimensions", embedding.len());
    
    Ok(())
}
```

## Model Preparation

### Download ONNX Models

Download the pre-exported ONNX models from HuggingFace:

```bash
pip install huggingface_hub
huggingface-cli download llm-semantic-router/mmbert-embed-32k-2d-matryoshka --include "onnx/*" --local-dir ./mmbert-onnx
```

### Model Directory Structure

The ONNX models are organized as:

```
mmbert-onnx/onnx/
├── layer-6/
│   ├── model.onnx       # Fast model (3.7x speedup)
│   ├── tokenizer.json
│   └── config.json
├── layer-11/
│   └── ...
├── layer-16/
│   └── ...
├── layer-22/
│   ├── model.onnx       # Full model (baseline)
│   ├── tokenizer.json
│   └── config.json
└── model_config.json    # Layer metadata
```

## Testing

```bash
# Set the model path
export MMBERT_MODEL_PATH=./mmbert-onnx/onnx

# Run tests
go test -v ./...

# Run benchmarks
go test -bench=. -benchtime=10s
```

## API Reference

### Go Functions

| Function | Description |
|----------|-------------|
| `InitMmBertEmbeddingModel(path, useCPU)` | Initialize the model |
| `GetEmbedding(text)` | Generate full embedding (768 dim) |
| `GetEmbeddingWithDim(text, dim)` | Generate embedding with dimension truncation |
| `GetEmbedding2DMatryoshka(text, layer, dim)` | Full 2D Matryoshka control |
| `GetEmbeddingsBatch(texts, layer, dim)` | Batch embedding generation |
| `CalculateEmbeddingSimilarity(text1, text2, layer, dim)` | Calculate cosine similarity |
| `CalculateSimilarityBatch(query, candidates, topK, layer, dim)` | Find top-k similar candidates |

### Supported Dimensions

- **768** (full): Best quality
- **512**: ~99.5% quality
- **256**: ~99% quality  
- **128**: ~98.5% quality
- **64**: ~98% quality, fastest

### Supported Layers

- **22** (full): Best quality
- **16**: ~80% quality, 1.5x faster
- **11**: ~67% quality, 2x faster
- **6**: ~56% quality, 3.7x faster

## Performance Recommendations

### Use CPU (Default)

For this model size (307M params), **CPU inference matches GPU performance**. The ONNX Runtime CPU backend is highly optimized with:

- AVX512/AVX2 SIMD instructions
- Multi-threaded execution
- Optimized BLAS operations

### Optimize with Layer Early-Exit

The **biggest speedup comes from layer early-exit**, not GPU acceleration:

| Use Case | Recommended Config | Latency (64 tokens) |
|----------|-------------------|---------------------|
| Maximum quality | Layer 22, dim 768 | ~30ms |
| Balanced | Layer 11, dim 512 | ~15ms |
| Fast routing | Layer 6, dim 256 | **~8ms** |

### When to Use GPU

Consider GPU only if:

- You're running many models and need to share GPU memory
- Your deployment already has GPU infrastructure
- You're processing very large batches (100+ concurrent requests)

For most use cases, **CPU is simpler and equally fast**.

## Running the Benchmark

```bash
# Build benchmark
cargo build --release --example benchmark_cpu_vs_gpu

# Run comprehensive CPU vs GPU comparison
./target/release/examples/benchmark_cpu_vs_gpu ./mmbert-onnx/onnx

# Or in Docker with GPU
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  -v $(pwd):/workspace rocm/pytorch:latest \
  /workspace/target/release/examples/benchmark_cpu_vs_gpu /workspace/mmbert-onnx/onnx
```

## License

MIT OR Apache-2.0
