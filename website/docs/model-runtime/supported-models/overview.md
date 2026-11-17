---
title: Supported models
sidebar_label: Supported models
description: Every System One model the built-in runtime serves, by family, with its size, question types, inputs and memory.
---

# Supported models

These are the decision models the built-in model runtime serves: every model
the Router can ask a System One question. Each one has its own page, generated
from the catalog the Router and the runtime use, so its question types, sizes
and pinned revision always match the code.

:::tip Which one?
Start with the default, **vela3-fast**, on the CPU. Move to a larger Vela 3.0
size for more accuracy with the built-in questions still trained, or to
Decision 3.0 for your own questions about images and video.
[Choose a model](../get-started/choose-a-model.md) compares them.
:::

## Families

| Family | What it is for | Spans | Runs on |
| --- | --- | --- | --- |
| [Decision 3.0](./decision-3/overview.md) | Your own questions about text, images and video, zero-shot, from 0.6B to 27B | No | GPU |
| [Decision 2.0](./decision-2/overview.md) | Your own questions about text, zero-shot, from 0.6B to 27B | No | CPU (Kai-0.6B) or GPU |
| [Decision 1.0](./decision-1/overview.md) | The first decision models, encoders and decoders from 0.6B to 9B | No | CPU (0.6B) or GPU |
| [Vela 3.0](./vela-3/overview.md) | The Router's own models: trained for the built-in questions, with spans; the default | Yes | CPU (vela3-fast) or GPU |
| [Vela 2.0](./vela-2/overview.md) | The previous Router models, with set and span questions | Yes | CPU (0.3B) or GPU |

Every model answers `choice` and `noul` questions; the tables below show the
rest. A model without spans can still be the decision model: bind `pii` and
`hallucination` to a span-capable deployment
([Span capability](../../system-one/get-started/concepts.md#span-capability)).

## All models

<!-- BEGIN GENERATED MODEL TABLE -->

### Decision 3.0

| Model | Parameters | Answers | Spans | Reads | Device memory |
| --- | --- | --- | --- | --- | --- |
| [d3-edge](./decision-3/d3-edge) | 0.595B | choice, noul, score | — | text, image, video | 4 GiB |
| [d3-lite](./decision-3/d3-lite) | 0.853B | choice, noul, score | — | text, image, video | 4 GiB |
| [d3-nano](./decision-3/d3-nano) | 2.21B | choice, noul, score | — | text, image, video | 8 GiB |
| [d3-mini](./decision-3/d3-mini) | 4.54B | choice, noul, score | — | text, image, video | 16 GiB |
| [d3-flash](./decision-3/d3-flash) | 8.39B | choice, noul, score | — | text, image, video | 32 GiB |
| [d3](./decision-3/d3) | 26.1B | choice, noul, score | — | text, image, video | 72 GiB |
| [d3-turbo](./decision-3/d3-turbo) | 26.1B | choice, noul, score | — | text, image, video | 72 GiB |
| [d3-ultra](./decision-3/d3-ultra) | 26.1B | choice, noul, score | — | text, image, video | 72 GiB |

### Decision 2.0

| Model | Parameters | Answers | Spans | Reads | Device memory |
| --- | --- | --- | --- | --- | --- |
| [Decision-2.0-Kai-0.6B](./decision-2/kai-0-6b) | 0.597B | choice, noul, score | — | text | 4 GiB |
| [Decision-2.0-Eos-0.8B](./decision-2/eos-0-8b) | 0.753B | choice, noul, score | — | text | 4 GiB |
| [Decision-2.0-Sol-2B](./decision-2/sol-2b) | 1.88B | choice, noul, score | — | text | 8 GiB |
| [Decision-2.0-Nox-4B](./decision-2/nox-4b) | 4.21B | choice, noul, score | — | text | 16 GiB |
| [Decision-2.0-Lux-9B](./decision-2/lux-9b) | 7.94B | choice, noul, score | — | text | 24 GiB |
| [Decision-2.0-Vega-27B](./decision-2/vega-27b) | 29.4B | choice, noul, score | — | text | 72 GiB |

### Decision 1.0

| Model | Parameters | Answers | Spans | Reads | Device memory |
| --- | --- | --- | --- | --- | --- |
| [Decision-1.0-Kai-0.6B](./decision-1/kai-0-6b) | 0.572B | choice, noul, score | — | text | 4 GiB |
| [Decision-1.0-Lex-0.6B](./decision-1/lex-0-6b) | 0.572B | choice, noul, score | — | text | 4 GiB |
| [Decision-1.0-Route-0.6B](./decision-1/route-0-6b) | 0.572B | choice, noul, score | — | text | 4 GiB |
| [Decision-1.0-Eos-0.8B](./decision-1/eos-0-8b) | 0.753B | choice, noul, score | — | text | 4 GiB |
| [Decision-1.0-Sol-2B](./decision-1/sol-2b) | 1.88B | choice, noul, score | — | text | 8 GiB |
| [Decision-1.0-Nox-4B](./decision-1/nox-4b) | 4.21B | choice, noul, score | — | text | 16 GiB |
| [Decision-1.0-Lux-9B](./decision-1/lux-9b) | 7.94B | choice, noul, score | — | text | 24 GiB |

### Vela 3.0

| Model | Parameters | Answers | Spans | Reads | Device memory |
| --- | --- | --- | --- | --- | --- |
| [vela3-fast](./vela-3/vela3-fast) | 1.52B | choice, noul, span | router | text | 12 GiB |
| [vela3-lite](./vela-3/vela3-lite) | 0.903B | choice, noul, score, span | router | text, image, video | 4 GiB |
| [vela3-mini](./vela-3/vela3-mini) | 4.79B | choice, noul, score, span | router | text, image, video | 16 GiB |
| [vela3-flash](./vela-3/vela3-flash) | 8.83B | choice, noul, score, span | router | text, image, video | 32 GiB |
| [vela3-turbo](./vela-3/vela3-turbo) | 27.6B | choice, noul, score, span | router | text, image, video | 72 GiB |
| [vela3.1-pico-150m](./vela-3/vela3-1-pico-150m) | 0.153B | choice, noul, span | router | text | 2 GiB |
| [vela3.1-pico-380m](./vela-3/vela3-1-pico-380m) | 0.381B | choice, noul, span | router | text | 3 GiB |

### Vela 2.0

| Model | Parameters | Answers | Spans | Reads | Device memory |
| --- | --- | --- | --- | --- | --- |
| [Vela-2.0-0.3B](./vela-2/vela-2-0-3b) | 0.309B | choice, noul, score, set, span | router | text | 4 GiB |
| [Vela-2.0-0.8B](./vela-2/vela-2-0-8b) | 0.756B | choice, noul, score, set, span | router, broad | text | 4 GiB |
| [Vela-2.0-4B](./vela-2/vela-2-4b) | 4.21B | choice, noul, score, set, span | router, broad | text | 16 GiB |
| [Vela-2.0-9B](./vela-2/vela-2-9b) | 7.95B | choice, noul, score, set, span | router, broad | text | 24 GiB |

<!-- END GENERATED MODEL TABLE -->

"Device memory" is the catalog's minimum for the model's device; measure your
own peak with your input lengths and concurrency.

:::note Embedding models
The embedding models that recall features use, Vela 1.0 Embedding,
Qwen3-Embedding-0.6B and Vela 1.0 Omni, are not System One models and are not
listed here. See [Embeddings](../guides/embeddings.md) and
[Images and audio](../guides/multimodal.md).
:::

## How these pages are made

The model pages are generated by `tools/docs/generate_model_pages.py` from the
Router's decision catalog, the runtime's registry tables, the recorded
reference answers and the Router's calibrated thresholds. Run
`make docs-model-pages` after changing a built-in model, and
`make docs-generated-check` fails until you do. The family pages and this page
are written by hand around their generated tables.
