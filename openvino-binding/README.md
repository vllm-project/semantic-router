# OpenVINO Binding for Semantic Router

High-performance Go bindings for semantic routing using Intel® OpenVINO™ Toolkit. This binding provides BERT-based text embeddings, similarity search, and classification capabilities optimized for Intel CPUs and accelerators.

## Features

- 🚀 **High Performance**: Optimized inference with OpenVINO on Intel hardware
- 🔍 **Semantic Search**: BERT embeddings and cosine similarity
- 📊 **Classification**: Text classification with confidence scores
- 🏷️ **Token Classification**: Named entity recognition and PII detection
- 🔄 **Batch Processing**: Efficient batch similarity computation
- 💻 **Multi-Device**: Support for CPU, GPU, VPU, and other Intel accelerators
- 🔌 **CGO Bindings**: Native C++ integration with Go

## Environment Variables

The following environment variables are required or recommended:

- **`OPENVINO_TOKENIZERS_LIB`** (Required): Path to `libopenvino_tokenizers.so`

  ```bash
  export OPENVINO_TOKENIZERS_LIB="/path/to/libopenvino_tokenizers.so"
  ```

- **`OPENVINO_MODEL_PATH`** (Optional): Path to OpenVINO model XML file
  - Default: `../../test_models/category_classifier_modernbert/openvino_model.xml`

- **`CANDLE_MODEL_PATH`** (Optional): Path to Candle model directory (for benchmarks)
  - Default: `../../../models/category_classifier_modernbert-base_model`

- **`LD_LIBRARY_PATH`** (Required): Include the path to the built library

  ```bash
  export LD_LIBRARY_PATH="/path/to/openvino-binding/build:$LD_LIBRARY_PATH"
  ```

## Building

### 1. Build C++ Library

```bash
cd openvino-binding

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j$(nproc)

# Install (optional)
sudo cmake --install .
```

### 2. Build Go Bindings

```bash
# Go back to openvino-binding directory
cd ..

# Test Go bindings
go build -v ./...

# Run tests (if available)
go test -v ./...
```

## Running Benchmarks

The benchmark compares OpenVINO and Candle implementations:

```bash
# Set up environment variables
export OPENVINO_TOKENIZERS_LIB="/path/to/libopenvino_tokenizers.so"
export OPENVINO_MODEL_PATH="/path/to/openvino_model.xml"
export CANDLE_MODEL_PATH="/path/to/candle/model"
export LD_LIBRARY_PATH="/path/to/openvino-binding/build:/path/to/candle-binding/target/release:$LD_LIBRARY_PATH"

# Run benchmark
cd cmd/benchmark
go run main.go
```

## Converting Models to OpenVINO IR Format

OpenVINO requires models in Intermediate Representation (IR) format (`.xml` and `.bin` files).
