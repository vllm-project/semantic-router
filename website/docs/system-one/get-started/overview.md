---
title: System One
sidebar_label: Overview
sidebar_position: 1
slug: /system-one/overview
description: The decision model answers every judgment the Router makes, through one API of typed questions.
---

# System One

System One is how the Router judges a request. A **decision model** reads the
request and answers typed questions about it with probabilities, so every
routing judgment, from "which subject is this about?" to "does this cached
request ask the same thing?", goes through one model and one API.

```json title="POST /v1/systemone"
{
  "state": {"request": "Prove that the square root of 2 is irrational."},
  "questions": {
    "math": {"type": "noul", "instructions": "Is this a mathematics request?"},
    "difficulty": {"type": "score", "instructions": "How hard is this request?",
                   "levels": ["Easy", "Medium", "Hard"]}
  }
}
```

The answer gives P(yes) for `math` and the expected level of `difficulty`. A
rule in your configuration matches on those numbers.

## Why it works this way

- **Zero-shot first.** A question works as soon as you write it. There is no
  classifier to train and no example bank to collect: every System One model
  answers from the question's instructions and its options' descriptions.
- **One model for every judgment.** Signals, the model selector, guards and
  the judges of the cache, RAG and tools all ask the same deployment, and
  the Router batches their questions into one call per stage.
- **Trained where it helps.** Some model families were trained on the
  Router's built-in questions, and some have calibrated thresholds. Neither is
  required; see [Trained and zero-shot answers](./concepts.md#trained-and-zero-shot-answers).

## The default decision model

With nothing configured, every question goes to the pre-declared deployment
`primary`: Vela 3.0 **vela3-fast**, a CPU-friendly encoder that the built-in
[model runtime](../../model-runtime/get-started/overview.md) serves next to the Router. It
starts when the first rule, selector, plugin or judge needs it, answers every
built-in question with trained adapters and calibrated thresholds, and returns
spans.

```yaml title="config.yaml (the default, which you can leave out)"
global:
  model_catalog:
    system:
      decision_model: {deployment: primary}
```

:::tip
Point `decision_model` at another deployment to change the model every
question asks. A signal type, a recipe or a single rule can still name its
own; see [Choose a deployment](../deploy/deployments.md).
:::

## What asks System One

| Feature | What it asks | Page |
| --- | --- | --- |
| Application signals | One built-in question each: domain, fact check, user feedback, jailbreak, safety, PII, hallucination, refusal, turn relation, complexity, output modality | [Signals](../../tutorials/signal/overview.md) |
| `systemone` signal | Your own question, in plain language | [Write custom questions](../use/custom-questions.md) |
| `systemone` selector | Which candidate model should answer | [System One selection](../../tutorials/algorithm/selection/systemone.md) |
| `response_guard` | Response-stage signals about the model's answer | [Response Guard](../../tutorials/plugin/response-guard.md) |
| `masking` | Where personal data is, in the final provider-bound request | [Masking](../../tutorials/plugin/masking.md) |
| `response_cache` | Whether a recalled cached request asks the same thing (`same_request`) | [Response Cache](../../tutorials/plugin/response-cache.md) |
| `context` | Whether a document or a segment helps answer the request (`relevance`) | [Context](../../tutorials/plugin/context.md) |
| `tools` | Whether the request needs a tool, and which ones (`tool_need`, `tool_rank`) | [Tools](../../tutorials/plugin/tools.md) |
| `escalate` | Whether an answer is acceptable (`answer_check`) | [Escalate](../../tutorials/algorithm/looper/escalate.md) |

[How the Router uses System One](../use/router.md) shows the call each feature
makes.

## Model families

| Family | Models | Answers | Runs on |
| --- | --- | --- | --- |
| Vela 3.0 | `vela3-fast` (the default), `vela3-lite`, `vela3-mini`, `vela3-flash`, `vela3-turbo` | choice, noul and span, with trained adapters for the built-in questions; the larger sizes also answer score and read images | CPU (vela3-fast) or GPU |
| Decision 3.0 (d3) | `d3-edge` to `d3`, `d3-turbo` and `d3-ultra`, which is also hosted | choice, noul and score, over text, images and video | GPU, or a hosted API |
| Decision 2.0 and 1.0 | 0.6B to 27B | choice, noul, score | CPU (0.6B) or GPU |
| Vela 2.0 | 0.3B to 9B | choice, noul, score, set, span; trained for the built-in questions | CPU (0.3B) or GPU |

Models differ in the question types they answer. Decision 1.0, 2.0 and 3.0
return no spans, and vela3-fast answers no `score` question. A rule, judge or
selector whose question type its deployment does not answer fails to load:
`span_required` for a span question, such as PII detection, and
`question_type_unsupported` for the other types.

Every model has its own page under
[Supported models](../../model-runtime/supported-models/overview.md), and the
[support matrix](../reference/support-matrix.md) shows which built-in questions
each family answers trained, zero-shot or not at all.

## Next steps

- [Concepts](./concepts.md): question types, the state, presets, trained and
  zero-shot answers, and span capability.
- [The System One API](../use/api.md): the request and response formats, model
  cards and errors.
- [Write custom questions](../use/custom-questions.md): turn a routing idea into
  a `systemone` rule.
- [Choose a deployment](../deploy/deployments.md): local, attached or remote
  models, and what the Router checks at load.
