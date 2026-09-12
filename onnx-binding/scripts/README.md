# Export Vela text models

Export an immutable, local checkpoint after task evaluation has selected and
frozen it. Export parity checks numerical agreement; it does not establish
accuracy, robustness, or useful context length.

## Sequence and token classifiers

`export_classifier.py` supports merged `ModernBertForSequenceClassification`
and `ModernBertForTokenClassification` artifacts. It checks the complete task
head and label mapping before exporting. Install PyTorch for your platform,
Transformers 4.57.6, NumPy, ONNX, onnxscript, and ONNX Runtime. The export receipt
records the versions actually used.

```bash
python onnx-binding/scripts/export_classifier.py \
  --model snapshots/classifier --output exports/classifier --dtype float32
python onnx-binding/scripts/export_classifier.py \
  --model snapshots/classifier --output exports/classifier --dtype float16
```

The files are `model.onnx` and `model_sdpa_fp16.onnx`, including their external
tensor files. The FP16 variant quantizes the encoder while keeping the original
FP32 head parameters, head computation, and mean-pool accumulation. Avoid
rounding a low-variance pooled vector to FP16 before its normalization layer.
Both graphs accept dynamic batch and sequence lengths. Sequence outputs are
`[batch, labels]`; token outputs are `[batch, tokens, labels]`.

Verification compares both variants with native FP32 inference. FP32 exports
must match logits; both variants must match the complete probability vector.
Multi-label heads use independent sigmoid probabilities. Dynamic checks include
local-attention boundaries and batches with different valid lengths. Padded
token predictions are excluded, while valid tokens must remain unaffected by
masked padding. These synthetic cases measure numerical agreement only.

Use `--validation-lengths` to add budgets supported by the checkpoint and the
test machine. A 32K architecture setting does not replace long-input task
evaluation. `--verify-only` repeats checks for an existing graph; it must still
match the supplied immutable checkpoint. Receipts contain source/export hashes
and measured errors. Debug metadata is stripped before publication.

For a validated FP16 encoder with an FP32 head, produce the AMD custom-op graph:

```bash
python onnx-binding/ort-ck-flash-attn/scripts/rewrite_graph.py \
  exports/classifier/model_sdpa_fp16.onnx \
  exports/classifier/model_fa_fp16.onnx --fp32-task-head
```

Retain the portable FP32 graph, tokenizer and native configuration in the model
package. Revalidate the rewritten graph against the native reference on AMD,
including padding and non-aligned lengths, before publishing it. Confirm actual
GPU execution with runtime profiling; initialization alone is insufficient.

## Embeddings and rerankers

`export_2d_matryoshka.py` exports physically shortened encoders for the selected
trained exits. It requires the checkpoint's explicit representation contract.
Embedding graphs return hidden states for mask-aware FP32 mean pooling,
dimension truncation, and L2 normalization. Reranker graphs return a single
logit using the selected layer/dimension's independent trained FP32 head.

Keep the available exits, dimensions, normalization, source hashes, and measured
quality matrix with the artifact. A lower layer count or dimension is a
different operating point; it is not automatically an accuracy-preserving
optimization. See each model's card for measured choices and defaults.

Run the dependency-light test entry with `make test-training-contracts`. With
the export dependencies installed, it also executes actual small dynamic graphs
against native CPU inference. The CK rewrite tests run with
`make ck-rewrite-test`.
