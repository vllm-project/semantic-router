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

The input decides the output precision, read from its weight tensors. The
rewriter keeps the weights as it finds them and, for an FP32 graph, adds fp32↔fp16 casts around each
`CKFlashAttention` node. A rewrite of the FP32 `model.onnx` must therefore be
named `model_fa.onnx`; `model_fa_fp16.onnx` needs an FP16 input graph. The
script refuses an fp16 name for an FP32 graph, because `find_onnx_models`
ranks candidates by name alone. A recognized RoPE position-to-sin/cos branch
may keep its FP32 frequency constant and calculations before casting to FP16;
this numerical constant is not counted as an encoder or classifier weight.
For an FP16 encoder exported with FP32 pooling and a task head, add
`--fp32-task-head`. The rewriter preserves FP32 constants only when they
contribute to the output and cannot feed any attention block. It still rejects
mixed encoder weight precision. In this mode the `fp16` filename describes the
encoder; the artifact's export receipt must record the FP32 head separately.
`make ck-rewrite-test` runs the rewriter's
unit tests (`scripts/test_rewrite_graph.py`); the changed-file gate runs them
for any change under this directory.

The matcher accepts `Softmax` fed by
`Add(MatMul(Mul(q), Mul(k)), mask)`, including the published intent FP16
export's `Where(IsNaN(probabilities), 0, probabilities)` guard before the AV
`MatMul` and its reshape/transpose/reshape K path. A post-QK scaling export
is not supported; unmatched or partially matched graphs are refused.

Local/global attention is determined from the mask's positional comparisons,
not exporter tensor names or layer frequency. The inclusive ModernBERT
condition `abs(query_position - key_position) <= 64` becomes CK windows
`(64, 64)`. For the 22-layer intent model with global attention every third
layer, expect **8 global and 14 local nodes**. `--local-attention` checks the
source window; it does not override it. Unknown mask comparisons are refused.
The matcher also accepts the Transformers 4.57.6 / Torch 2.10 token export's
`Where(bool(1-padding), negative, 1-padding)` and separate local-window fill,
with the key-padding broadcast axes checked explicitly.
The old dense mask computation is removed after all layers are rewritten;
each custom op receives one `[B, 1, 1, S]` padding bias.

Recognized FP16 masked mean-pooling heads accumulate and divide in FP32 before
casting back to the head dtype, preventing long-sequence sum overflow. This
does not add pooling to token-classification graphs or change their output rank.

Check the original SDPA graph's task and output rank against the native Hugging
Face model before using it as a correctness reference: a sequence-classification
export cannot validate a token-classification task. Re-export the native task
when they disagree; changing output shape metadata does not fix its computation.
Existing published FA artifacts may contain different window assignments and
must not be used to validate a new rewrite. Run the Python graph tests and the
GPU SDPA tests (including local windows, one-dimensional padding bias, mixed
batch lengths, and non-tile-aligned lengths) before model-level comparison.
Graph tests alone do not establish GPU correctness or 32K task accuracy.

## Bound FP32 attention memory

For precision-sensitive classifiers, `scripts/rewrite_blocked_attention.py`
provides an alternative using standard ONNX operators and FP32 arithmetic:

```bash
python3 scripts/rewrite_blocked_attention.py \
  original/model.onnx blocked/model.onnx \
  --block-size 256 --max-score-bytes 536870912
```

Global attention keeps the complete key/value sequence. Local attention crops
keys and values to the query block's window only when the entire batch has no
padding and conservative numeric bounds prove that omitted mask probabilities
are zero. Otherwise it keeps the complete sequence. An ONNX `Loop` bounds the
score tensor using the batch, head count, and full key length. Local masks keep
absolute positions, including the last partial block. The rewrite preserves
separate Q/K scaling, mask fill values, and the NaN guard, and removes the
original quadratic mask. Global attention still performs quadratic computation.
The score budget covers one FP32 score tensor; it is not a total device-memory
limit. Weights, other intermediates, and runtime workspaces require extra memory.

The input must be a recognized FP32 self-attention export with a rank-two binary
padding mask. Unknown masks, mixed precision, partial attention matches, and
existing control flow are rejected. The output uses an external weight file
beside the graph; keep both files together. This variant does not require the
CK custom-op library. It does require runtime support for `Loop` and its body
operators.

Add `--batch-size 1` when qualifying only the Router's single-request token-ID
path. This fixes both input batch dimensions so ONNX Runtime rejects larger
batches. It does not split inputs or change attention math. Native HF batching
and a separate dynamic ONNX variant require their own evidence.

`make ck-rewrite-test` executes dynamic-shape, padding, local-window, and tail
comparisons on the CPU runtime. For every checkpoint, separately compare full
probabilities and task decisions against the native reference at each supported
length and batch size. Different FP32 kernels can still produce different
rounding. Check the runtime profile to confirm that attention inside the loop
executes on the requested GPU. A successful rewrite alone qualifies neither an
execution provider nor a model artifact.

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
