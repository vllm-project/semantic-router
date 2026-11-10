---
title: Choose a model, size and hardware
sidebar_label: Choose a model
sidebar_position: 3
description: Which model to use for each task, how big a decision model you need, and what hardware runs it.
---

# Choose a model, size and hardware

Start from the task, then pick the smallest model that is accurate enough and
the hardware you have. Every built-in model is pinned to an exact Hugging Face
revision, so the same name always loads the same files.

:::tip The short answer
Keep the default, **vela3-fast** on the CPU, until you measure a reason to
change it. It answers every built-in question with trained adapters and
calibrated thresholds, and returns spans for PII and hallucination checks.
:::

## By task {#by-task}

| You want to | Default model | Notes |
| --- | --- | --- |
| Answer the routing signals, your own questions and every other System One judgment | Vela 3.0 vela3-fast | The [decision model](../../system-one/get-started/overview.md); the sizes below trade latency for accuracy |
| Embeddings for the response cache, memory and RAG | `vllm-sr/Vela-1.0-Encoder-307M-Embedding` | Recall only; smaller sizes and fewer layers trade quality for speed |
| Larger or instructed text embeddings | `Qwen/Qwen3-Embedding-0.6B` | 0.6B, 1,024 dimensions |

With no model configured, every System One question asks one deployment,
`primary`, which batches compatible questions within each routing stage.
Recall features embed with `vela-embedding`, which starts only when one of them
needs it.

## Decision models {#decision-models}

Every family answers questions you write yourself. They differ in size, in the
question types they answer, and in whether they were trained on the Router's
built-in questions:

| Family | Sizes | Answers | Trained for built-in questions | Runs on |
| --- | --- | --- | --- | --- |
| [Vela 3.0](../supported-models/vela-3/overview.md) | vela3-fast (0.4B per question), lite 0.8B, mini 4B, flash 9B, turbo 27B | choice, noul, span; the larger sizes also score and read images | Yes | CPU (vela3-fast) or GPU |
| [Decision 3.0](../supported-models/decision-3/overview.md) | edge 0.6B to d3 27B, d3-turbo and d3-ultra 27B | choice, noul, score, over text, images and video | No (zero-shot) | GPU |
| [Vela 2.0](../supported-models/vela-2/overview.md) | 0.3B, 0.8B, 4B, 9B | choice, noul, score, set, span | Yes | CPU (0.3B) or GPU |
| [Decision 2.0](../supported-models/decision-2/overview.md) | Kai 0.6B to Vega 27B | choice, noul, score | No (zero-shot) | CPU (Kai) or GPU |
| [Decision 1.0](../supported-models/decision-1/overview.md) | Kai, Lex, Route 0.6B to Lux 9B | choice, noul, score | No (zero-shot) | CPU (0.6B) or GPU |

For your own questions on a small budget, Decision 2.0 Kai-0.6B answers two
questions in about 0.2 s on 16 CPU cores. For questions about images and video,
use a Decision 3.0 size on a GPU.
[Supported models](../supported-models/overview.md) has a page per model with
its memory, question types and a deployment snippet.

:::warning
Decision 1.0, 2.0 and 3.0 answer no span questions. When one of them is the
default judgment deployment, bind `pii` and `hallucination` to a span-capable
deployment; see [Span capability](../../system-one/get-started/concepts.md#span-capability).
:::

## The built-in signals run on vela3-fast {#built-in-signals}

Every application signal defaults to `vllm-sr/vela3-fast`
([collection](https://huggingface.co/collections/vllm-sr/vela-30-6ad1c430eb1718eb4754edcd)).
All of them share one deployment, `primary`, which batches compatible questions
from the same routing stage. They follow the decision model;
[Choose a model for a signal](../../installation/choose-a-model-for-a-signal.md)
shows how to give one signal, one recipe or one rule another deployment.

- **Questions:** each signal asks its preset in the form the model was trained
  on, so the domain, fact check, feedback, jailbreak, safety, refusal and
  modality questions run on trained adapters; the others are answered
  zero-shot. PII and hallucination use the span-set adapter, so their spans
  keep exact character offsets. The
  [support matrix](../../system-one/reference/support-matrix.md) lists every
  preset.
- **Precision:** FP32 on every device, as the package was evaluated.
- **Memory:** the deployment holds about 7 GB once loaded. Give the Router
  12 GiB, the Helm chart's default limit; less can be killed for memory while
  the model loads.
- **Input:** ordinary routing judgments may truncate to one 8,192-token window,
  and evidence spans to one 32,768-token window. Jailbreak, safety, PII and
  hallucination require complete input; window scans extend coverage up to the
  scan budget. Incomplete coverage leaves the answer unknown, except that a
  `jailbreak` or `pii` rule counts it as a match. See
  [Long inputs](../reference/http-api.md#long-inputs).
- **Thresholds:** a rule without a `predicate` takes the threshold calibrated
  for vela3-fast's scores (below).

Measured on the
[router signal suite](https://huggingface.co/datasets/vllm-sr/router-signal-suite)'s
test files at each signal's calibrated threshold
([record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela3-router-signals.md)),
vela3-fast is ahead of Vela 2.0 0.3B on prompt guard (true-positive rate 0.714
against 0.638), PII (0.953 against 0.888), domain, user feedback (balanced
accuracy 0.807 against 0.592) and modality (0.858 against 0.539), and behind on
safety (0.830 against 0.865) and fact check (0.542 against 0.698). End-to-end
CPU latency was not measured for it.

### Thresholds {#thresholds}

Each calibrated threshold keeps the Vela 1.0 specialist's operating point on
the suite's dev split: its false-positive rate, or for a confidence floor its
share of requests below the floor. On vela3-fast they are jailbreak 0.88,
domain 0.41, PII 0.18, fact check 0.9999, user feedback 0.47, safety 0.52 and
modality 0.63.

- **PII:** the span-set adapter applies its own per-length thresholds before it
  returns a span, so 0.18 accepts every span it returns.
- **Fact check:** the fact-check adapter's probabilities sit near 0 and 1, so
  its operating points are close to 1.
- **Other models:** a model without a calibrated threshold for a preset uses
  0.5, and the configuration load reports an `uncalibrated_preset` warning.
- **Your own predicates** are yours. The record maps each Vela 1.0 value to
  vela3-fast: prompt guard 0.3–0.9 → 0.87–0.91, PII → 0.18, safety 0.5 → 0.52,
  fact check 0.65–0.95 → 0.9997–0.9999, modality 0.7 → 0.63.

## Choose a size {#choose-a-size}

The decision model is the deployment that answers the Router's judgment tasks
and [`systemone` questions](../../tutorials/signal/systemone.md) that no binding
or rule gives another deployment. Declare the deployment once, then name its
exact key. Without an override, the built-in default is vela3-fast.

```bash title="Pass a model to change the default deployment"
vllm-sr serve vllm-sr/vela3-lite --platform rocm
```

```yaml title="The same change in config.yaml"
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/vela3-lite
        device: rocm:0
    system:
      decision_model:
        deployment: primary
```

- The CLI saves this deployment choice in the active configuration as a new
  version, which `vllm-sr config versions` lists and `vllm-sr config rollback`
  undoes; later starts keep it, and `vllm-sr status` shows it.
- The Helm chart's `decisionModel` value and the operator's
  `spec.config.decision_model` set the same decision model.
- Deployment keys are exact and case-sensitive. Model identity, device and
  engine arguments belong to the deployment, not to `decision_model`.
- `--platform` selects the runtime platform, while explicit deployment
  placement remains authoritative.

Measured through the Router on the router signal suite, against the Vela 1.0
specialists, and for the latency record's five request signals
([record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-decision-model-sizes.md)):

| Decision model | Hardware | Held-out accuracy against Vela 1.0 | p50 on a GPU | p50 on 12 CPU cores |
| --- | --- | --- | ---: | ---: |
| `vela3-fast` (default) | CPU or GPU | Ahead on prompt guard, domain, modality and feedback; behind on safety and fact check ([record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela3-router-signals.md)) | Not measured | Not measured |
| `Vela-2.0-0.3B` | CPU or GPU | Ahead on prompt guard and safety, behind on domain, modality and feedback | 6.6 ms | 79 ms |
| `Vela-2.0-0.8B` | CPU or GPU | Ahead on domain, prompt guard, safety, modality and hallucination; behind on PII | 40.1 ms | about 3 s |
| `Vela-2.0-4B` | GPU, about 17 GB | Ahead on every signal but fact check | 55.2 ms | Not measured |
| `Vela-2.0-9B` | GPU, about 32 GB | Ahead on every signal | 76.5 ms | Not measured |

- **GPU:** one AMD Instinct MI325X, sequential requests. At concurrency 16 a GPU
  serves about 154 (0.3B), 25 (0.8B), 18 (4B) and 13 (9B) requests per second.
- **Use a GPU for the 4B and 9B in this workload.** Their CPU latency was not
  measured in this record. Placement follows the deployment and runtime
  requirements, not a CLI ban based on the model name; explicit GPU devices
  must match the selected platform and be available on the host.
- **The 0.8B on a CPU** is a decoder: a request takes seconds, as the table
  shows. Serve it on a GPU, or keep the 0.3B on a CPU.
- **Every size** is behind Vela 1.0 on user feedback's fresh file (CrossWOZ)
  and on PII in distribution. Per-signal numbers with intervals are in the
  record.

Each size has its own calibrated thresholds, which a rule without a `predicate`
takes when you switch:

| Decision model | Jailbreak | Domain | PII | Fact check | User feedback |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vela3-fast` | 0.88 | 0.41 | 0.18 | 0.9999 | 0.47 |
| `Vela-2.0-0.3B` | 0.75 | 0.28 | 0.01 | 0.93 | 0.37 |
| `Vela-2.0-0.8B` | 0.71 | 0.38 | 0.07 | 0.994 | 0.34 |
| `Vela-2.0-4B` | 0.63 | 0.45 | 0.05 | 0.9984 | 0.33 |
| `Vela-2.0-9B` | 0.42 | 0.46 | 0.14 | 0.998 | 0.35 |

Predicates a configuration sets, such as the built-in recipes', stay; the
record maps each to every size (for example `mom-v1`'s `prompt_attack` 0.75 is
0.71 on the 0.8B, 0.63 on the 4B and 0.42 on the 9B). A binding keeps that one
signal on its own model. Vela 2.0 0.3B, the previous default, is compared with
the Vela 1.0 specialists in
[Vela 2.0](../supported-models/vela-2/overview.md#vela-20).

## Hardware {#hardware}

| Hardware | Status | Use |
| --- | --- | --- |
| CPU | Validated | Every Router image runs models on CPU out of the box. |
| AMD Instinct MI300X, MI325X | Validated | Set `device: rocm:0`. `vllm-sr serve --platform rocm` and the `vllm-sr-rocm` image ship PyTorch for ROCm. |
| NVIDIA GPUs | Validated | Set `device: cuda:0`. `vllm-sr serve --platform cuda` ships PyTorch for CUDA. Records: [index](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/cuda-h200.md), [image smoke](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/cuda-router-image.md), [goldens](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/cuda-golden-answers.md). Fused kernels without bit-exact evidence stay off the exact profile. |
| Intel GPUs | Available, not yet validated | `device: xpu:0`, with the runtime installed next to an XPU build of PyTorch. |
| Ascend NPUs (910B, 950) | Works, not yet validated | `device: npu:0`, with the runtime installed next to torch_npu and the host CANN toolkit. See [Ascend NPU](../../installation/ascend-npu.md). |
| Apple silicon | CPU only in this release | On macOS the docker target runs the CPU image, because Docker's Linux VM gets no GPU. Host GPU support is tracked in [#4636](https://github.com/vllm-project/semantic-router/issues/4636). |

`device: auto` (the default) picks the first validated GPU with enough free
memory and otherwise the CPU:

- A GPU you name explicitly must exist, or the model fails to load with a clear
  reason instead of quietly running on the CPU.
- Accelerators that are not yet validated (`xpu`, `npu`) are never picked by
  `auto`; on a host that has one, name the device explicitly to use it.

<details>
<summary>The AMD software stack the models are checked on</summary>

On AMD GPUs the Router images ship the stack the runtime is validated on:
PyTorch 2.12 for ROCm 7.2, FLA 0.5.2, and `causal-conv1d` 1.7.0 built for ROCm.
Its `causal-conv1d` also carries code for MI200 and MI350 GPUs, so the models
that use it run there too, though only MI300X and MI325X are validated.

The built-in models' GPU reference answers are checked on that stack, and every
model compares itself with them when it loads. With another PyTorch, ROCm or
kernel build, a model can fail that check or report
[`unverified`](../deploy/troubleshooting.md#a-ready-models-self-check-says-unverified).
If a model's reference answers had to be recorded again on this stack, its
family's [record](https://github.com/vllm-project/semantic-router/tree/main/src/model-runtime/docs/records)
says so and gives how often it agrees with the released answers.

</details>

### How much memory {#how-much-memory}

Memory depends on the model family and its precision (`dtype`), not just the
device:

- FP32 weights use about 4 bytes per parameter, BF16 about 2, plus memory for
  inputs, activations and concurrency. A 307M FP32 task model needs roughly
  1.3 GB for its weights alone.
- Each replica has its own process and weight copy; place replicas on suitable
  devices (see [Place and scale workers](../deploy/placement-and-scaling.md#place-and-scale-replicas)).
- [Quantized packages](../quantization/overview.md) of Decision 3.0 hold the
  same model in about a third to three fifths of the memory.

Check the model's page and record, and measure peak memory for your workload.

## Your own models {#your-own-models}

Hugging Face ModernBERT and mmBERT embedding models load the same way as the
built-in Vela Embedding: give a Hub repository with `revision`, or an absolute
path to a local copy. A decision model of another architecture needs a family
plugin that answers System One questions; see
[Add your own model family](../extend/plugins.md).
