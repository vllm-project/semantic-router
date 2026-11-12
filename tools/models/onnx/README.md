# Export Vela text models

Export an immutable, local checkpoint after task evaluation has selected and
frozen it. Export parity checks numerical agreement; it does not establish
accuracy, robustness, or useful context length.

Exports retain all-masked attention protection using the standard ONNX
predicate `Not(Equal(x, x))`. This is equivalent to `IsNaN(x)`, including NaNs,
infinities, and signed zeros, and avoids requiring a dedicated provider kernel.
The transformation preserves opsets and tensor storage, and applies only when
the operator schemas prove compatible input types. Qualify the resulting graph
on each target provider; a portable operator set alone does not prove GPU execution.

## Sequence and token classifiers

`export_classifier.py` supports merged `ModernBertForSequenceClassification`
and `ModernBertForTokenClassification` artifacts. It checks the complete task
head and label mapping before exporting. Install PyTorch for your platform,
Transformers 4.57.6, NumPy, ONNX, onnxscript, and ONNX Runtime. The export receipt
records the versions actually used.

```bash
python tools/models/onnx/export_classifier.py \
  --model snapshots/classifier --output exports/classifier --dtype float32
python tools/models/onnx/export_classifier.py \
  --model snapshots/classifier --output exports/classifier --dtype float16
```

For a CUDA or ROCm host, export on the GPU with `--device cuda --export-only`.
This mode checks the graph structure and records numerical qualification as
pending. Validate the graph on the target runtime before publication. The default
CPU workflow includes numerical checks; `--verify-only` runs those checks again.

The files are `model.onnx` and `model_sdpa_fp16.onnx`, including their external
tensor files. The FP16 variant quantizes the encoder while keeping the original
FP32 head parameters, head computation, and mean-pool accumulation. Avoid
rounding a low-variance pooled vector to FP16 before its normalization layer.
Both graphs accept dynamic batch and sequence lengths. Sequence outputs are
`[batch, labels]`; token outputs are `[batch, tokens, labels]`.

Verification compares both variants with native FP32 inference. FP32 sequence
exports must match logits; both variants must match the complete probability vector.
Token exports must also preserve every valid token's exact BIO argmax. Their raw
logit differences are reported separately: rounding in low-probability classes
can change raw logits while preserving probabilities and entity decisions.
Multi-label heads use independent sigmoid probabilities. Dynamic checks include
local-attention boundaries and batches with different valid lengths. Padded
token predictions are excluded, while valid tokens must remain unaffected by
masked padding. These synthetic cases measure numerical agreement only.

Use `--validation-lengths` to add budgets supported by the checkpoint and the
test machine. A 32K architecture setting does not replace long-input task
evaluation. `--verify-only` repeats checks for an existing graph; it must still
match the supplied immutable checkpoint. Receipts contain source/export hashes
and measured errors. Debug metadata is stripped before publication.

Retain the portable FP32 graph, tokenizer and native configuration in the model
package: the model runtime's `onnxruntime` engine serves the package's `onnx/`
graphs, and its native engine serves the same package from the checkpoint.

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
against native CPU inference. `make onnx-artifact-test` packs shared external
weights and checks them with real CPU inference.
