---
title: Vela 3.0
sidebar_label: Vela 3.0
description: The Router's own decision models, trained for every built-in question, with spans; vela3-fast is the default.
---

# Vela 3.0

Vela 3.0 is the Router's own family of decision models. Every size answers the
built-in routing questions on trained adapters and returns spans for PII and
hallucination checks. **vela3-fast** is the default decision model and runs on
a CPU; the larger sizes run a Decision 3.0 package with gated adapters on a
GPU.

```yaml title="The default, which you can leave out"
global:
  model_catalog:
    system:
      decision_model: {deployment: primary}  # vllm-sr/vela3-fast on the CPU
```

## Sizes

<!-- BEGIN GENERATED MODEL TABLE -->

| Model | Parameters | Answers | Spans | Reads | Device memory |
| --- | --- | --- | --- | --- | --- |
| [vela3-fast](./vela3-fast) | 1.52B | choice, noul, span | router | text | 12 GiB |
| [vela3-lite](./vela3-lite) | 0.903B | choice, noul, score, span | router | text, image, video | 4 GiB |
| [vela3-mini](./vela3-mini) | 4.79B | choice, noul, score, span | router | text, image, video | 16 GiB |
| [vela3-flash](./vela3-flash) | 8.83B | choice, noul, score, span | router | text, image, video | 32 GiB |
| [vela3-turbo](./vela3-turbo) | 27.6B | choice, noul, score, span | router | text, image, video | 72 GiB |

<!-- END GENERATED MODEL TABLE -->

Models are in the
[Vela 3.0 collection](https://huggingface.co/collections/vllm-sr/vela-30-6ad1c430eb1718eb4754edcd).

## vela3-fast

`vllm-sr/vela3-fast` is a 380M UltraBERT encoder, 0.4B with the adapter a
question runs on. Its ten adapters carry their own top layers and heads, so the
deployment loads 1.5B parameters.

- **Trained adapters** answer the Router's built-in questions; `choice` and
  `noul` questions of your own run on its decision backbone.
- **Spans** run on span adapters routed by label set and wording: personal
  data, unsupported claims, evidence, relations and entities, with the `pii`
  and `halu` presets.
- **It answers no `score` questions.** A `noul` question is read by its
  instructions alone, so one that describes `false` or `true` is invalid. A
  `set` question takes one of its trained router texts.
- **Precision and memory:** FP32 on every device, about 7 GB once loaded; give
  the Router 12 GiB.
- Its answers match the package's own runtime bit for bit on a CPU
  ([parity record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela3-parity.md)).

[Choose a model](../../get-started/choose-a-model.md#built-in-signals) has its
measured accuracy on the built-in signals and its calibrated thresholds.

## The larger sizes

`vllm-sr/vela3-lite`, `-mini`, `-flash` and `-turbo` run a pinned Decision 3.0
package (d3-lite, d3-mini, d3-flash and d3-turbo, at the revisions their
adapters were trained on) with gated LoRA adapters over it:

- The Router asks its built-in questions under the IDs these packages were
  trained on, so they run on trained adapters. Harm categories are asked as one
  `hazard:` noul per category, which only vela3-lite's hazard adapter answers.
- Every other `choice`, `noul` and `score` question answers as that d3 package
  does.
- They answer `span` questions, routed by label set and wording as
  vela3-fast's are. A span question reads a target of up to 32,768 tokens in
  windows; a longer target fails that question with `scan_budget_exceeded`.
- vela3-turbo answers each request on d3-turbo's state-once path first, then
  the router questions again with their adapter on.
- Requests with images run with every adapter off and refuse span questions.
- vela3-mini and vela3-flash run open span labels without a description by
  name, where their packages write a definition with the base first
  ([parity record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela3-d3-parity.md)).

:::note
Only vela3-fast and the Vela 2.0 sizes have calibrated thresholds. The larger
Vela 3.0 sizes use 0.5 for a built-in question without a `predicate`, with an
`uncalibrated_preset` warning.
:::
