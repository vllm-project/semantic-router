# Vela Router Models

## Overview

[Vela 1.0](https://huggingface.co/collections/vllm-sr/vela-10)
is a family of fourteen published model checkpoints for intelligent routing.
The Router registry currently includes the Vela 307M encoder base and ten task
models covering routing, prompt protection, content safety, retrieval and
reranking. Each registered release is pinned to an immutable revision.

| Model | Role |
| --- | --- |
| Encoder | Shared base for adapting new routing tasks |
| Domain | Request subject across 14 domains |
| Guard | Prompt injection and jailbreak detection |
| Safety | Unsafe content detection |
| Hazard | Twelve independent content risk categories |
| PII | Personal information spans across 17 entity types |
| FactCheck | Whether a request needs factual verification |
| Feedback | Four feedback types and a neutral `NO_FEEDBACK` class |
| Modality | Text, image, or combined output intent from a written request |
| Embedding | Multilingual retrieval and semantic matching |
| Reranker | Relevance scoring for query-document pairs |

The full model name is `Vela-1.0-Encoder-307M`, followed by the task suffix.
Modality classifies the requested output modality from text.
FactCheck requests verification and does not verify the truth of an answer.

The public collection also includes **Halu** for answer evidence-support
detection and **Omni Nano / Omni Mini** for text, image and audio embeddings.
Halu is the Router's specialist hallucination detector, and Omni serves its
multimodal embeddings; see [Check answers against context](../../model-runtime/guides/hallucination.md)
and [Images and audio](../../model-runtime/guides/multimodal.md).
See the [Vela 1.0 release announcement](/blog/vela-models) for the full family.

## What Problem Does It Solve?

A shared model family supplies task-specific signals and retrieval components
with explicit model identities, input budgets and serving policies.

## When to Use

Use Vela for built-in routing tasks or adapt the shared Encoder for a new task.
Choose input length and representation size against your workload's quality
and latency requirements.

## Omni checkpoints

The September 19 releases provide separate text, image and audio encoders in a
shared embedding space. The model runtime serves them as prepared bundles for
multimodal embeddings.

| Checkpoint | Total parameters | Output dimensions | Text limit | Text backbone and readout |
| --- | ---: | ---: | ---: | --- |
| [Omni Nano](https://huggingface.co/vllm-sr/Vela-1.0-Omni-Nano/blob/0496b39a51c8199592e58cbff81c250f056bd94b/README.md) | 163.8M (163,771,288) | 384 | 512 tokens | Frozen GIST-small; CLS readout |
| [Omni Mini](https://huggingface.co/vllm-sr/Vela-1.0-Omni-Mini/blob/f7fafd36abf49adf88b1b2ec0186c68b008eeb07/README.md) | 1.36B (1,361,475,288) | 768 | 32,768 tokens | Qwen3-Embedding-0.6B; last-token Matryoshka readout |

Parameter counts include every modality branch, including the CLAP audio branch;
integer and running-statistics buffers are not parameters. Nano retains CLS
readout with an identity projection. Mini normalizes the 1024-dimensional last
nonpadding token, takes its first 768 dimensions, then normalizes again.

Mini defaults to shared text for cross-modal comparisons. For text-only tasks,
`encode_text` accepts either `task=` or a custom `instruction=`. The retrieval
preset requires `role="query"` or `role="document"`; documents stay unprefixed.
The 32,768-token budget includes any instruction prefix and special tokens.
Inputs over the budget are rejected unless Mini's caller explicitly sets
`truncate=True`. Nano retains its 512-token limit and rejects longer inputs.
The instructed English benchmark uses fixed official task instructions, not a
sweep of the generic presets.

Both models accept original-rate PCM of at most 30 seconds, as mono or
channels-first arrays. Pass its actual sampling rate; each branch independently
resamples the original waveform to 16 kHz for Whisper and 48 kHz for CLAP.
Preserve higher-rate PCM instead of downsampling to 16 kHz before the call.
CLAP encodes endpoint-spaced windows of at most ten seconds, aggregates their
normalized vectors, and applies a learned residual map after frozen TRAIN
standardization. That residual is added to the retained, unnormalized Whisper
affine before final L2 normalization. Text/image paths are retained; audio is
newly trained and evaluated.

The latest cards compare routing and cross-modal retrieval with the
[original small](https://huggingface.co/vllm-sr/multi-modal-embed-small/tree/fdf8e01b7b0f3a69ac1ac8e2a64dcb1ede177ba4)
and [original large](https://huggingface.co/vllm-sr/multi-modal-embed-large/tree/e21cde3ccc414c56f504b322662f42c603a939ee)
models. Scores are 0–100; each cell shows original → current:

| Metric | Original small → Nano | Original large → Mini |
| --- | ---: | ---: |
| Banking77, accuracy | 70.42 → 87.99 | 75.78 → 86.56 |
| MASSIVE English, accuracy | 65.95 → 81.97 | 72.31 → 80.96 |
| COCO, image → text, R@1 | 40.83 → 60.87 | 42.53 → 67.44 |
| COCO, text → image, R@1 | 30.18 → 55.82 | 35.04 → 61.60 |
| LibriSpeech, audio → text, R@1 | 4.21 → 16.12 | 56.99 → 86.14 |
| LibriSpeech, text → audio, R@1 | 9.58 → 20.34 | 78.58 → 94.90 |

Both sides use the same held-out pools, reused across releases, with a 128-token
text cap. Classification uses labeled TRAIN prototypes; retrieval uses complete
candidate pools and all matching positives. This differs from official MTEB
classification probes. Full metrics and uncertainty are in the linked evaluations.

For the complete MTEB 2.21.0 panels, **Mean(TaskType)** is the primary metric;
Mean(Task) is supplementary. The original small/large models have not been
measured under these complete-panel protocols.

| Panel and mode | Mean(TaskType) | Global rank | Rank at ≤ size | Gap to best at ≤ size | Mean(Task) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nano · English v2, 41 tasks · default text | 60.78 | 66/188 | 5/75 | 0.61 pp | 64.88 |
| Mini · English v2, 41 tasks · official instructed text | 64.68 | 38/188 | 10/134 | 3.78 pp | 70.38 |
| Nano · MAEB audio-only, 19 tasks · default audio | 52.34 | 19/64 | 6/27 | 3.51 pp | 43.59 |
| Mini · MAEB audio-only, 19 tasks · default audio | 54.87 | 12/64 | 5/50 | 2.85 pp | 47.77 |

Ranks combine the September 17 registry snapshots with the current Vela models,
including single-modality specialists and differing reported protocols. The
size limit counts total parameters. **Neither Vela model lies on the observed
frontier of either complete panel**, under either aggregate. Selected task
strengths include Nano's IMDb **91.95 accuracy** and NMSQA **62.90 max AP**,
and Mini's Mridingham **68.14 accuracy** and SIBFLEURS **39.07 accuracy**;
these task-level frontiers do not establish overall benchmark leadership.

Nano's English score is retained through its unchanged frozen text path.
Mini's instructed English panel uses fixed official instructions and a fresh
matched raw-text control: Mean(TaskType) rises from **58.79 to 64.68**, with
38 tasks improving and three declining. This instruction-mode result does not
replace the default shared mode for image/audio comparisons. Both new audio
paths are freshly evaluated; speech–text retrieval regresses despite aggregate
audio gains. See the pinned [Nano evaluation](https://huggingface.co/vllm-sr/Vela-1.0-Omni-Nano/blob/0496b39a51c8199592e58cbff81c250f056bd94b/benchmarks/EVALUATION.md),
[Mini evaluation](https://huggingface.co/vllm-sr/Vela-1.0-Omni-Mini/blob/f7fafd36abf49adf88b1b2ec0186c68b008eeb07/benchmarks/EVALUATION.md)
and [matched instruction comparison](https://huggingface.co/vllm-sr/Vela-1.0-Omni-Mini/blob/f7fafd36abf49adf88b1b2ec0186c68b008eeb07/benchmarks/instruction-mode.md#matched-raw-comparison)
for measurement identities, all tasks and regressions. These panels do not
establish full multilingual or image coverage, latency, or memory performance.

## Defaults and input budgets

With no model configured, the built-in Domain, Guard, Safety, PII, FactCheck,
Feedback, Modality and hallucination signals run on Vela 2.0 0.3B, in one call
per request ([Choose a model](../../model-runtime/choose-a-model.md#vela-20)).
Semantic Embedding, Hazard and the Reranker use Vela 1.0, and naming a Vela 1.0
task model in `global.model_catalog.system` restores it. Only models required by
a recipe are loaded. The base encoder is a training parent and is not loaded as
an additional routing signal.

A module running a Vela 1.0 task model without a threshold of its own uses
**0.5** for Guard, **0.95** for FactCheck and **0.7** for Feedback. `NO_FEEDBACK` emits no feedback match. Safety is independent
of Guard, so an unsafe content request need not be classified as a prompt attack.
Hazard uses per-label thresholds from its artifact-bound operating point; a
single threshold does not represent its published decision policy.

An input budget is a deployment choice. A module `max_sequence_length` of `0`
keeps the conservative 512-token policy. Embedding defaults to 22 layers,
768 dimensions and `full_context: false`. Explicit model bindings can accept
up to **32,768 tokens**, including special tokens, with `overflow: reject`.
A rejected input is not silently shortened.

## Configuration

Every Vela model runs in the [model runtime](model-runtime/overview.md),
so the defaults need no configuration. The following excerpt binds Domain to
an explicit CPU deployment with a 32K budget. Apply it to an existing
configuration containing providers, signals and decisions.

```yaml
routing:
  model_bindings:
    domain_classifier:
      deployment: vela-domain
      contract: label_distribution.v1

global:
  model_catalog:
    deployments:
      vela-domain:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Domain
        device: cpu
        input:
          max_tokens: 32768
          overflow: reject
```

Use the same deployment and consumer-binding structure for other classifiers.
PII returns `token_spans.v1`, Embedding `embedding.v1` and Reranker
`relevance_scores.v1`. The runtime reads each model's architecture from its
package, so these bindings need no adapter. See
[Run it with the router](../../model-runtime/deploy.md) for the full contract.

For Hazard, bind the independent classifier contract `label_scores.v1` and pin
`operating_point.json` with its SHA-256. The policy binds the weights, tokenizer,
overlapping windows and twelve thresholds. Decisions select labels without
replacing those thresholds. The reference configuration includes this complete
pattern.

PII's overlapping scan and Hazard's windowed policy differ from whole-document
classification. Select the policy appropriate to the task, and measure latency
with the input lengths your application will send.

## Inference engines and hardware

The model runtime runs Vela models with PyTorch. CPU execution is validated,
including 32K inputs, and so are AMD Instinct MI300X and MI325X GPUs through
ROCm. CUDA works but is not yet validated; measure NVIDIA performance on the
target hardware. [Profiles](../../model-runtime/profiles.md) trade exactness
for speed, and [Choose a model](model-runtime/choose-a-model.md) lists
what each model costs.

The [Vela AMD recipe](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/vela-amd/README.md)
places all ten task models on an AMD GPU and preserves the published operating
policies. `--platform rocm` selects the AMD image and device access; it does not
make every authored model use a GPU or override an explicit CPU deployment.
See [AMD ROCm](../../installation/amd-rocm.md#run-vela-routing-models-on-amd).

Earlier releases ran Vela through Candle, ONNX Runtime, MIGraphX or OpenVINO
and selected exported ONNX graphs with `head`. Those providers are removed; the
model repositories keep their ONNX exports for other tools.
`vllm-sr config migrate` rewrites old deployments; see
[Migrate from the native bindings](model-runtime/migrate.md).

All models expose their supported input length, usage and comparable evaluation
results in their model cards. Published comparisons use the previous mmBERT
family on matched evaluation data. Quality scores, maximum accepted input length
and inference performance measure different properties.

## Verify live routing and reranking

For the Vela AMD recipe, download and validate the complete configuration, then
start it with the AMD image. Connect an existing vLLM backend served as
`vela-default` at `vllm:8000`, as explained in the recipe's Model Card.

```bash
curl --fail --location --output vela-amd.yaml \
  https://raw.githubusercontent.com/vllm-project/semantic-router/main/config/recipes/vela-amd/config.yaml
vllm-sr config validate --config vela-amd.yaml
vllm-sr serve --platform rocm --config vela-amd.yaml
```

Route Preview returns actual signal values, decisions and per-signal latency.
Keep its input within the recipe's 8K classifier budget:

```bash
curl --fail 'http://localhost:8080/api/v1/routing/preview?trace=true' \
  -H 'Content-Type: application/json' \
  -d '{"model":"vela-auto","text":"Help me debug this Python program."}' \
  | jq '{decision_result, signal_confidences, signal_values, signal_errors, metrics, eval_trace}'
```

Use the public model name declared by your entrypoint. Check `/ready` before
sending requests and inspect the returned signal values and evaluation trace.

Reranking executes inside the vectorstore RAG plugin after routing. Bind
`rag.reranker` and enable `rerank` as described in
[neural reranking](/docs/tutorials/plugin/rag#neural-reranking). Verify it through
a real chat request with indexed documents. Its request trace records candidate
count, reranker identity, relevance scores and reranking latency; routing Preview
does not execute the RAG plugin.

## Preserve an earlier deployment

Explicit older model paths and aliases retain their original repositories.
Select the matching mapping file, threshold and input policy when reproducing
an earlier classifier. Upgrading to Vela does not rewrite explicit model choices.

Changing embedding weights or representation changes the vector space. Response
caches and memory isolate data by representation identity. Existing vector stores
require compatible embeddings or reindexing; old data is not silently adopted
or deleted. See [stores and tools](./stores-and-tools.md).
