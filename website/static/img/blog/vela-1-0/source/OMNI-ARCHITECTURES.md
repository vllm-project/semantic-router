# Omni architecture sources

`figures_omni.py` and `paper_svg.py` generate the two public embedding graphs.
The PNGs are the website assets; SVGs retain editable vector shapes and text.
`render_omni.cjs` exports PNGs and vector PDFs with Chromium and checks text
bounds. It only renders `07-omni-nano.svg` and `08-omni-mini.svg`.

## Release evidence

- Nano: `d8ac5b5ac2274a501fc61aeb5be70cec1855a806` (September 18, 2026).
  [Configuration](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Nano/blob/d8ac5b5ac2274a501fc61aeb5be70cec1855a806/config.json),
  [single-modality runtime](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Nano/blob/d8ac5b5ac2274a501fc61aeb5be70cec1855a806/omni_components/single_modality.py),
  [text encoder](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Nano/blob/d8ac5b5ac2274a501fc61aeb5be70cec1855a806/omni_components/text_encoder.py).
  Total parameters: 135,383,808. Twelve-layer GIST-small BERT, CLS readout and
  identity text projection. The package removes unused fusion, text pooler and exit
  modules; it preserves the public embedding paths and the reported scores.
- Mini: `2aecd547915f5cffe68ba3c2e2c3a678de8b193e` (September 18, 2026).
  [Configuration](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/2aecd547915f5cffe68ba3c2e2c3a678de8b193e/config.json),
  [Qwen readout](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/2aecd547915f5cffe68ba3c2e2c3a678de8b193e/omni_components/qwen_text_backbone.py),
  [modality paths](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/2aecd547915f5cffe68ba3c2e2c3a678de8b193e/omni_components/mini.py),
  [public wrapper](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/2aecd547915f5cffe68ba3c2e2c3a678de8b193e/vela_omni.py).
  Total parameters: 1,332,891,200. Twenty-eight-layer Qwen3, causal GQA with
  16 Q / 8 KV heads, 128 dimensions per head, pre-RMSNorm and SwiGLU. Last
  nonpadding token → full 1024-dimensional L2 normalization → 768-dimensional
  prefix → L2 normalization. The public path uses no learned text projection
  and no language-model head.
- Transformer internals follow [Transformers v4.57.6](https://github.com/huggingface/transformers/tree/v4.57.6/src/transformers/models),
  including Qwen3 per-head Q/K RMSNorm before rotary position encoding.

The total sizes include all three modality branches. The three output vectors
share a space but are encoded independently, not summed. Dropout is omitted
because it is inactive in inference. Identity text projection is explicitly
labeled to distinguish it from the earlier learned residual projection.

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
check cannot prove correct tensor flow. The September 18 refresh was also
checked for single-page vector PDFs, extractable text, exact PNG dimensions,
and PDF-to-PNG visual agreement. Static structure checks do not establish
runtime latency or task quality.
