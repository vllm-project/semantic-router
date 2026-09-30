# Synthetic ModernBERT RoPE references

These fixtures contain a randomly initialized two-layer ModernBERT backbone
(hidden size 32, vocabulary size 64), deterministic integer inputs and numerical
arrays. They contain no trained checkpoint, tokenizer, dataset or task examples.
The 79,672-byte `weights.safetensors.fixture` file is ordinary safetensors with a
fixture suffix and a binary Git attribute. Both reference versions use identical
weight bytes. The two official versions also produced bitwise-identical numerical
outputs, stored once in `reference.safetensors.fixture`. The small `rope.json` and
`tiny-output.json` files retain each case's configuration, inputs, tensor names
and shapes. The 27 named F32 tensors preserve all 16,928 original numerical
values without decimal rounding, compression or quantization. Their manifests
bind the generator, official implementation and shared output files by SHA-256.
The reference configurations use only the official `norm_eps` spelling.
Separate loader tests cover the legacy alias and reject conflicting values.

Generate with the pinned official Python packages, on CPU, in this order:

```bash
python generate_modernbert_rope_fixtures.py --mode tf4 --output /tmp/rope-reference
python generate_modernbert_rope_fixtures.py --mode tf5 --output /tmp/rope-reference
```

Use [the repository generator](../../scripts/generate_modernbert_rope_fixtures.py)
under Transformers 4.57.6 and 5.3.0 respectively. Copy the output directory;
the second run rejects any difference in tensor names, shapes, dtypes, bits or
shared case descriptions. No model download
or accelerator is needed. The stored fixtures used PyTorch 2.10.0 on CPU;
regeneration with another math library can vary in floating-point rounding.

Reference mathematics come from the official
[4.57.6 RoPE implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/modeling_rope_utils.py)
and [5.3.0 implementation](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/modeling_rope_utils.py).
The cache cases cover 12 positions through 32767, three theta/scaling recipes,
FP32/F16/BF16 cache values, and FP32 rotated Q/K. The tiny encoder cases use
batches of two at lengths 1, 17 and 65, with padding. Assertions compare valid
token hidden states and masked means; padded query hidden states are excluded.
The shared tests state fixed finite tolerances. Default RoPE additionally has
bitwise cache and complete encoder regression tests against the previous Candle
FP32 computation.

## Transformers 5 loading limitation

In Transformers 5.3.0, `_validate_yarn_rope_parameters` receives a per-layer
parameter dictionary but reads `original_max_position_embeddings` from the
outer `self.rope_parameters` dictionary. A normal nested YaRN configuration
therefore raises `KeyError: 'original_max_position_embeddings'` during config
construction/loading. The generator first constructs a normal default config,
then assigns the explicit nested parameters before invoking the unmodified
model. This is a **mathematical reference only**, not a successful standard
`AutoConfig` / checkpoint load. It does not patch installed code or add extra
checkpoint keys. The ordinary Transformers 4 config path succeeds. Candle
supports the explicit independent per-layer JSON semantics itself.
