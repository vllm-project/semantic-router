---
title: Choose a model
description: See the six Decision 1.0 model IDs and where the current runtime can run them.
---

Each `vllm-sr decision serve` instance serves one model. Copy its full ID into the
[start command](./overview.md) and the `model` field of every
[API request](./api.md). After launch, `GET /v1/models` shows the ID that the
instance accepts.

The Hugging Face repositories contain model data and metadata, not a bundled
server. Serving requires a Decision-enabled vLLM Semantic Router CLI and its
matching Decision Runtime image. The runtime reads each selected snapshot's
root `config.json` to find its weights and tokenizer; the packaged execution
profile supplies family, prompt, and hardware policy without pinning individual
model files.

| Model ID | Input limit | Current execution paths |
| --- | ---: | --- |
| `llm-semantic-router/Decision-1.0-Kai-0.6B` | 1,024 tokens | ROCm `gfx942`; Linux CPU |
| `llm-semantic-router/Decision-1.0-Lex-0.6B` | 1,024 tokens | ROCm `gfx942`; Linux CPU |
| `llm-semantic-router/Decision-1.0-Eos-0.8B` | 16,384 tokens | ROCm `gfx942`; Linux CPU |
| `llm-semantic-router/Decision-1.0-Sol-2B` | 16,384 tokens | ROCm `gfx942` |
| `llm-semantic-router/Decision-1.0-Nox-4B` | 16,384 tokens | ROCm `gfx942` |
| `llm-semantic-router/Decision-1.0-Lux-9B` | 16,384 tokens | ROCm `gfx942` |

The input limit applies to each prepared state-and-question input. A request
with several questions produces a separate input for each answer; it does not
make the model's token limit larger. An input that exceeds its model's limit
returns `413`.

The table lists execution paths present in the current code. Linux CPU is
limited to Kai, Lex, and Eos. CUDA has no installed Decision executor in this
build, and native Apple MLX launch is not yet available. See
[starting a model](./overview.md) for release availability.

## Model revisions

Without `--revision`, `decision serve` uses the model catalog's pinned model-only
release revision. To try a different revision of the **same model**, provide its
full 40-character commit SHA. The runtime verifies the selected files before
loading and checks that their layout fits the model family. Profiles are not
bound to a specific model-file hash or revision, but a different revision still
needs validation on the intended device before production use. The
[launch parameters](./parameters.md) explain the relevant options.
