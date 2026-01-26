# ML Binding for Semantic Router

This directory contains Rust-based ML algorithm implementations using [Linfa](https://github.com/rust-ml/linfa), the Rust ML framework.

> **Note:** This package provides **inference only**. Training is done in Python. See `src/training/ml_model_selection/`.

## Algorithms

| Algorithm | Linfa Crate | Status |
|-----------|-------------|--------|
| **KNN** (K-Nearest Neighbors) | `linfa-nn` | ✅ Inference |
| **KMeans** (Clustering) | `linfa-clustering` | ✅ Inference |
| **SVM** (Support Vector Machine) | `linfa-svm` | ✅ Inference |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING (Python)                             │
├─────────────────────────────────────────────────────────────────┤
│  src/training/ml_model_selection/                               │
│  ├── train.py          # Train models using scikit-learn        │
│  ├── upload_model.py   # Upload to HuggingFace                  │
│  └── download_model.py # Download from HuggingFace              │
│                                                                  │
│  Output: knn_model.json, kmeans_model.json, svm_model.json      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE (Rust/Go)                           │
├─────────────────────────────────────────────────────────────────┤
│  ml-binding/                                                    │
│  ├── src/knn.rs    # Load JSON, select using Linfa Ball Tree    │
│  ├── src/kmeans.rs # Load JSON, select using cluster centroids  │
│  ├── src/svm.rs    # Load JSON, select using decision function  │
│  └── ml_binding.go # Go bindings via CGO                        │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
ml-binding/
├── Cargo.toml           # Rust dependencies (Linfa)
├── go.mod               # Go module
├── ml_binding.go        # Go wrapper with CGO bindings
├── README.md            # This file
└── src/
    ├── lib.rs           # Library entry point
    ├── knn.rs           # KNN inference implementation
    ├── kmeans.rs        # KMeans inference implementation
    ├── svm.rs           # SVM inference implementation
    └── ffi.rs           # C FFI exports for Go (inference only)
```

> **Note:** Requires Linux/macOS/WSL with Rust and CGO. Windows native is not supported.

## Building

### Prerequisites

- Rust 1.70+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Go 1.22+

### Build the Rust Library

```bash
cd ml-binding

# Build release version
cargo build --release

# The library will be at:
# - Linux: target/release/libml_semantic_router.so
# - macOS: target/release/libml_semantic_router.dylib
```

### Set Library Path

```bash
# Linux
export LD_LIBRARY_PATH=$(pwd)/target/release:$LD_LIBRARY_PATH

# macOS
export DYLD_LIBRARY_PATH=$(pwd)/target/release:$DYLD_LIBRARY_PATH
```

### Run Tests

```bash
# Rust tests
cargo test

# Go tests (after building Rust library)
go test -v ./...
```

## Usage in Go

### Loading Pretrained Models

```go
package main

import (
    ml "github.com/vllm-project/semantic-router/ml-binding"
    "os"
)

func main() {
    // Load pretrained KNN model from JSON
    jsonData, _ := os.ReadFile("models/knn_model.json")
    knn, _ := ml.KNNFromJSON(string(jsonData))
    defer knn.Close()

    // Run inference
    query := []float64{0.9, 0.1, 0.0, /* ... 1038 dims total (1024 embedding + 14 category) */}
    selected, _ := knn.Select(query)
    // selected == "llama-3.2-3b" (or whichever model the KNN selects)

    // Same pattern for KMeans and SVM
    kmeansData, _ := os.ReadFile("models/kmeans_model.json")
    kmeans, _ := ml.KMeansFromJSON(string(kmeansData))
    
    svmData, _ := os.ReadFile("models/svm_model.json")
    svm, _ := ml.SVMFromJSON(string(svmData))
}
```

### Available Functions

| Function | Description |
|----------|-------------|
| `KNNFromJSON(json)` | Load KNN model from JSON |
| `KMeansFromJSON(json)` | Load KMeans model from JSON |
| `SVMFromJSON(json)` | Load SVM model from JSON |
| `knn.Select(embedding)` | Select best model for query |
| `knn.IsTrained()` | Check if model is loaded |
| `knn.ToJSON()` | Serialize model to JSON |
| `knn.Close()` | Release resources |

## Training Models

Training is done in Python using scikit-learn. See `src/training/ml_model_selection/`:

```bash
# Install dependencies
cd src/training/ml_model_selection
pip install -r requirements.txt

# Train models
python train.py \
  --data-file benchmark.jsonl \
  --output-dir models/

# Or download pretrained from HuggingFace
python download_model.py --output-dir models/
```

## Why Linfa for Inference?

1. **Performance**: Native Rust speed for inference
2. **Consistency**: Same pattern as `candle-binding` for embeddings
3. **Memory safety**: Rust guarantees
4. **No Python dependency**: Production inference without Python runtime

## Algorithm Details

### KNN (K-Nearest Neighbors)

- Uses Linfa Ball Tree for O(log n) neighbor search
- Quality-weighted voting: `score = 0.9 * quality + 0.1 * speed`
- Loads embeddings and metadata from JSON

### KMeans

- Loads cluster centroids from JSON
- Assigns queries to nearest centroid
- Each cluster maps to best model (by quality+speed)

### SVM (Support Vector Machine)

- Supports Linear and RBF kernels
- Loads support vectors from JSON
- One-vs-All classification for multi-model selection

## License

Apache-2.0 (same as semantic-router)
