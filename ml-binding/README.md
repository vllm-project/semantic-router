# ML Binding for Semantic Router

This directory contains Rust-based ML algorithm implementations using [Linfa](https://github.com/rust-ml/linfa), the Rust ML framework.

## Algorithms

| Algorithm | Linfa Crate | Status |
|-----------|-------------|--------|
| **KNN** (K-Nearest Neighbors) | `linfa-nn` | ✅ Implemented |
| **KMeans** (Clustering) | `linfa-clustering` | ✅ Implemented |
| **SVM** (Support Vector Machine) | `linfa-svm` | ✅ Implemented |

> **Note:** All ML algorithms use the [Linfa](https://github.com/rust-ml/linfa) Rust ML framework.

## Directory Structure

```
ml-binding/
├── Cargo.toml           # Rust dependencies (Linfa)
├── go.mod               # Go module
├── ml_binding.go        # Go wrapper with CGO bindings
├── README.md            # This file
└── src/
    ├── lib.rs           # Library entry point
    ├── knn.rs           # KNN implementation
    ├── kmeans.rs        # KMeans implementation
    ├── svm.rs           # SVM implementation
    └── ffi.rs           # C FFI exports for Go
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
# - Windows: target/release/ml_semantic_router.dll
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

```go
package main

import (
    ml "github.com/vllm-project/semantic-router/ml-binding"
)

func main() {
    // KNN Example
    knn := ml.NewKNNSelector(5)
    defer knn.Close()

    embeddings := [][]float64{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
    }
    labels := []string{"model-a", "model-b"}

    knn.Train(embeddings, labels)

    query := []float64{0.9, 0.1, 0.0}
    selected, _ := knn.Select(query)
    // selected == "model-a"

    // Save/Load
    json, _ := knn.ToJSON()
    restored, _ := ml.KNNFromJSON(json)
}
```

## Integration with Model Selection

The `ml-binding` package provides all ML algorithms for model selection:

```go
// Use Linfa-based KNN
knn := ml_binding.NewKNNSelector(5)

// Use Linfa-based KMeans
kmeans := ml_binding.NewKMeansSelector(8)

// Use Linfa-based SVM
svm := ml_binding.NewSVMSelector()
```

## Why Linfa?

1. **Battle-tested**: Community-maintained implementations
2. **Performance**: Native Rust speed
3. **Consistency**: Same pattern as `candle-binding` for embeddings
4. **Reduced maintenance**: Less custom code to maintain

## Algorithm Coverage

| Required (Issue #986) | Linfa | Status |
|-----------------------|-------|--------|
| KNN | ✅ `linfa-nn` | Implemented |
| KMeans | ✅ `linfa-clustering` | Implemented |
| SVM | ✅ `linfa-svm` | Implemented |

## License

Apache-2.0 (same as semantic-router)
