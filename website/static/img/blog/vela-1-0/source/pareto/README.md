# Omni size–quality figures

These are the original Matplotlib SVGs and complete observation data from the
[Nano model repository at `d8ac5b5a`](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Nano/tree/d8ac5b5ac2274a501fc61aeb5be70cec1855a806).
They retain the 17 September 2026 registry snapshot and separate 18 September
measured peer observations. Every input has a pinned URL and SHA-256 in
`provenance.json`; no peer observation is selected or dropped locally.

All four figures use the latest Nano package's **135,383,808 total parameters**
and Mini's **1,332,891,200 total parameters**, including all three modalities.
Both current model repositories agree on these sizes and plotted scores.
The pinned Nano revision supplies all four figures and their complete data
consistently; the companion [Mini revision](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/tree/2aecd547915f5cffe68ba3c2e2c3a678de8b193e)
contains the same updated Nano coordinate.

The four existing website PNG paths are raster exports at 2448 × 1496 pixels.
`export.mjs` exports the SVGs with Sharp 0.35.4; make the upstream DejaVu Sans
regular and bold fonts available to fontconfig, then run:

```sh
node export.mjs
python3 verify.py
```

`SHARP_MODULE` can specify an already installed Sharp package. The exporter does
not move points, rewrite labels or recompute chart layout. The SVGs are editable
vector source. `verify.py` separately recomputes strict Pareto dominance both
pairwise and with a sorted sweep, validates every supplied frontier flag and
records its result in `audit.json`.

Nano is on the NMSQA frontier and Mini on the SIBFLEURS frontier among displayed
observations. Neither Vela model is on the updated ArXiv or Vehicle frontier.
The F2 and AST measured points remain distinct from registry-reported points.
Protocols differ; these task-level comparisons do not establish benchmark-wide
leadership, latency, memory use or overall model quality.

See the pinned [methodology](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Nano/blob/d8ac5b5ac2274a501fc61aeb5be70cec1855a806/benchmarks/pareto-methodology.md)
for evaluated-artifact applicability and per-observation measurement protocols.
