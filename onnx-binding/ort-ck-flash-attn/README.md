# ONNX Runtime CK Flash Attention custom op

This library registers `com.ck::CKFlashAttention` with ONNX Runtime and
provides a graph rewriter for mmBERT attention blocks. It targets AMD ROCm
GPUs and CK-tile FP16 forward kernels with head dimensions 32, 64, or 128.

## Build

Requirements:

- ROCm 7.0 or later and `hipcc`
- `composablekernel-dev`
- CMake 3.21 or later and Ninja
- network access during the first configure step to fetch the selected ONNX
  Runtime C API headers

Run from `onnx-binding/ort-ck-flash-attn`:

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS=gfx942
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Change `GPU_TARGETS` when compiling for a supported architecture other than
the default `gfx942`. The shared library is written to
`build/libort_ck_flash_attn.so`.

## Rewrite a model

The rewriter requires the `numpy` and `onnx` Python packages:

```bash
python3 -m pip install numpy onnx
python3 scripts/rewrite_graph.py \
  model_sdpa_fp16.onnx \
  model_fa_fp16.onnx
```

Use `--hdim` and `--local-attention` only when they match the source model's
architecture. Keep the original SDPA model for correctness comparison.

## Load the custom op

The Semantic Router ONNX binding reads `ORT_CK_FLASH_ATTN_LIB`:

```bash
export ORT_CK_FLASH_ATTN_LIB="$PWD/build/libort_ck_flash_attn.so"
```

Python ONNX Runtime can register the same library explicitly:

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.register_custom_ops_library("build/libort_ck_flash_attn.so")
session = ort.InferenceSession(
    "model_fa_fp16.onnx",
    options,
    providers=["ROCmExecutionProvider"],
)
```

The rewritten model is not portable to an ONNX Runtime process that has not
registered this custom-op library.

## Container image

From this directory:

```bash
docker build -t ort-ck-flash-attn:local .
```

The image installs the library under `/usr/lib` and sets
`ORT_CK_FLASH_ATTN_LIB` accordingly. GPU devices still need to be exposed by
the container runtime.
