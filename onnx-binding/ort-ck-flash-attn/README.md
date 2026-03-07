# ORT CK Flash Attention

Standalone ONNX Runtime custom-op library that replaces dense attention with AMD Composable Kernel (CK) Flash Attention on ROCm GPUs.

## What it does

- Compiles CK-tile FMHA forward kernels (FP16, hdim 32/64/128) for MI300X (gfx942)
- Registers `com.ck::CKFlashAttention` as an ORT custom op via `RegisterCustomOps`
- Provides a Python graph rewriter to replace the dense attention subgraph in mmBERT ONNX models

## Build

```bash
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGPU_TARGETS=gfx942
cmake --build build --parallel $(nproc)
```

Requires: ROCm 7.0+, composablekernel-dev, CMake 3.21+, hipcc.

ORT C API headers are downloaded automatically during configuration.

## Usage

### 1. Rewrite the ONNX model

```bash
python3 scripts/rewrite_graph.py model_sdpa_fp16.onnx model_fa.onnx
```

### 2. Load in ORT (Rust)

```rust
let session = Session::builder()?
    .with_operator_library("/usr/lib/libort_ck_flash_attn.so")?
    .with_execution_providers([ROCmExecutionProvider::default().build()])
    .commit_from_file("model_fa.onnx")?;
```

Or set `ORT_CK_FLASH_ATTN_LIB` environment variable and the binding picks it up automatically.

### 3. Load in ORT (Python)

```python
import onnxruntime as ort
so = ort.SessionOptions()
so.register_custom_ops_library("build/libort_ck_flash_attn.so")
session = ort.InferenceSession("model_fa.onnx", so, providers=["ROCmExecutionProvider"])
```

## Docker

```bash
docker build -t ort-ck-flash-attn .
```
