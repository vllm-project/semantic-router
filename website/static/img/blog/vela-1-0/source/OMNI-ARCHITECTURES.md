# Omni architecture sources

`figures_omni.py` and `paper_svg.py` generate the two public embedding graphs.
The PNGs are the website assets; SVGs retain editable vector shapes and text.
`render_omni.cjs` exports PNGs and vector PDFs with Chromium and checks text
bounds. It only renders `07-omni-nano.svg` and `08-omni-mini.svg`.

## Release evidence

- Nano: `0496b39a51c8199592e58cbff81c250f056bd94b` (September 19, 2026).
  [Configuration](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Nano/blob/0496b39a51c8199592e58cbff81c250f056bd94b/config.json),
  [single-modality runtime](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Nano/blob/0496b39a51c8199592e58cbff81c250f056bd94b/omni_components/single_modality.py),
  [text encoder](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Nano/blob/0496b39a51c8199592e58cbff81c250f056bd94b/omni_components/text_encoder.py).
  Total parameters: 163,771,288. Twelve-layer GIST-small BERT, CLS readout and
  identity text projection. The text/image paths remain unchanged. A new CLAP residual augments the retained
  Whisper speech branch, and audio scores are newly measured.
- Mini: `f7fafd36abf49adf88b1b2ec0186c68b008eeb07` (September 19, 2026).
  [Configuration](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/f7fafd36abf49adf88b1b2ec0186c68b008eeb07/config.json),
  [Qwen readout](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/f7fafd36abf49adf88b1b2ec0186c68b008eeb07/omni_components/qwen_text_backbone.py),
  [modality paths](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/f7fafd36abf49adf88b1b2ec0186c68b008eeb07/omni_components/mini.py),
  [public wrapper](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/f7fafd36abf49adf88b1b2ec0186c68b008eeb07/vela_omni.py).
  Total parameters: 1,361,475,288. Twenty-eight-layer Qwen3, causal GQA with
  16 Q / 8 KV heads, 128 dimensions per head, pre-RMSNorm and SwiGLU. Last
  nonpadding token → full 1024-dimensional L2 normalization → 768-dimensional
  prefix → L2 normalization. The public path uses no learned text projection
  and no language-model head.
- Transformer internals follow [Transformers v4.57.6](https://github.com/huggingface/transformers/tree/v4.57.6/src/transformers/models),
  including Qwen3 per-head Q/K RMSNorm before rotary position encoding.

## Dual audio and optional text instructions

Both diagrams now follow the original-rate public audio path:

1. Derive 16 kHz mono and 48 kHz mono independently from the original PCM; never
   cascade the 16 kHz output into CLAP. Maximum duration is 30 seconds.
2. Preserve Whisper's full 1,500-frame mean and original unnormalized affine.
3. Encode 1–3 endpoint-spaced CLAP windows, each at most ten seconds. Short
   inputs use the pinned processor's repeat padding. The CLAP audio tower has
   spectrogram resize/reshape and four Swin stages with depths 2/2/6/2, widths 96/192/384/768 and 4/8/16/32 heads.
   The first three stages end in 2×2 patch merging; its expanded pre-LN block
   shows the two residual paths. No cyclic shift is used when the grid is no
   larger than the attention window.
4. CLAP applies final LayerNorm, global pooling and its 768→512→512 ReLU
   projection. Normalize each window vector, average multiple windows, and
   normalize the mean. One window uses its normalized vector directly.
5. Standardize by frozen TRAIN mean/scale, apply a learned bias-free 512→384
   (Nano) or 512→768 (Mini) map, add to the unnormalized Whisper affine, then
   normalize the combined vector.

The exact residual implementations are pinned in
[Nano's audio residual](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Nano/blob/0496b39a51c8199592e58cbff81c250f056bd94b/omni_components/tiny_clap_residual.py)
and [Mini's audio residual](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/f7fafd36abf49adf88b1b2ec0186c68b008eeb07/omni_components/medium_clap_residual.py).
The CLAP internal graph follows
[Transformers v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/clap/modeling_clap.py).

Mini's [optional instruction formatter](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/f7fafd36abf49adf88b1b2ec0186c68b008eeb07/omni_components/text_instructions.py)
changes the text input, not the Qwen computation graph. Default shared text
remains unprefixed; optional task/custom instructions share the 32,768-token
budget, and documents remain unprefixed. Explicit truncation is an opt-in API
choice. The text readout and image path are retained.

The total sizes include all modality branches and exclude non-parameter buffers.
The three output vectors share a space but are encoded independently, not summed
across modalities. Only the two audio paths are added before final normalization.
Dropout is omitted because it is inactive in inference.

## Rebuild

Requires Python 3.10+, Node.js, `playwright`, `sharp`, and Chromium or Chrome.
No model weights are downloaded or executed. From this `source` directory:

```sh
python3 figures_omni.py --output-dir ..
node render_omni.cjs ..
```

For an existing browser or packages installed outside Node's search path:

```sh
node render_omni.cjs .. --browser /path/to/chrome --node-modules /path/to/node_modules
```

The renderer produces 2× PNGs, transparent previews, PDFs, an HTML overview and
`layout-check.json`. Only the two PNGs and editable SVGs are published alongside
these sources. View both PNGs and PDFs after regeneration; an automatic bounds
check cannot prove correct tensor flow. The September 19 refresh was also
checked for single-page vector PDFs, extractable text, exact PNG dimensions,
and PDF-to-PNG visual agreement. Static structure checks do not establish
runtime latency or task quality.
