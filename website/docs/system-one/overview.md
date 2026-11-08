---
title: System One
sidebar_label: Overview
description: The decision model answers every judgment the Router makes, through one API of typed questions.
---

# System One

System One is how the Router judges a request. A decision model reads a
context, called the state, and answers typed questions about it with
probabilities: which of these options fits, is this true, how much, which of
these labels apply, and where in the text is each label. The Router asks it
every question that needs judgment, from "which subject is this request about?"
to "does this cached request ask the same thing?", through one API.

```text
POST /v1/systemone
{"state": {"request": "Prove that the square root of 2 is irrational."},
 "questions": {"domain": {...}, "jailbreak": {...}, "complexity": {...}}}
```

## Zero-shot first

A question works as soon as it is written: there is no classifier to train
and no example bank to collect. Every System One model answers every question
it supports zero-shot, from its instructions and the descriptions of its
options. Two optimizations sit on top, per model family, and neither is ever
required:

- **Trained bindings.** Some families were trained on the Router's built-in
  questions. When the answering model has a trained form of a question, the
  Router asks it in that form and maps the answer back.
- **Calibrated thresholds.** For the families measured on the router signal
  suite, a built-in question's default threshold keeps a known operating
  point. A model without one uses 0.5, and the configuration load says so
  with an `uncalibrated_preset` warning.

Models differ in the question types they answer (the table below): Decision
1.0, 2.0 and 3.0 return no spans, and vela3-fast answers no `score` question.
A model that answers `noul` also answers a `set` question, asked as one `noul`
per label. A rule, judge or selector whose question type its deployment does
not answer fails to load: `span_required` for a span question, such as PII
detection, and `question_type_unsupported` for the other types. The load knows
what a built-in model answers, and what a `systemone` deployment declares in
`capabilities.question_types`; any other model is checked against its card, or
its API's listing, when the generation prepares.

## The default decision model

With nothing configured, the Router's decision model is the pre-declared
deployment `primary`: Vela 3.0 vela3-fast, a CPU-friendly encoder that the
built-in [model runtime](../model-runtime/overview) serves next to the
Router. It starts when the first rule, selector, plugin or judge resolves to
it, answers every built-in question with trained adapters and calibrated
thresholds, and returns spans.

```yaml
global:
  model_catalog:
    system:
      decision_model: {deployment: primary}
```

That line is the default and can be left out. Point it at another deployment
to change the model every question asks; a type, a recipe or a single rule
can still name its own. [Choose a deployment](./deployments) covers local GPU
sizes, attached runtimes and remote System One APIs.

## What asks System One

| Feature | What it asks | Page |
| --- | --- | --- |
| Application signals | One built-in question each: domain, fact check, user feedback, jailbreak, safety, PII, hallucination, refusal, turn relation, complexity, output modality | [Signals](../tutorials/signal/overview) |
| `systemone` signal | Your own question, in plain language | [Write custom questions](./custom-questions) |
| `systemone` selector | Which candidate model should answer | [System One selection](../tutorials/algorithm/selection/systemone) |
| `response_guard` | Response-stage signals about the model's answer | [Response Guard](../tutorials/plugin/response-guard) |
| `masking` | Where personal data is, in the final provider-bound request | [Masking](../tutorials/plugin/masking) |
| `response_cache` | Whether a recalled cached request asks the same thing (`same_request`) | [Response Cache](../tutorials/plugin/response-cache) |
| `context` | Whether a document or a segment helps answer the request (`relevance`) | [Context](../tutorials/plugin/context) |
| `tools` | Whether the request needs a tool, and which ones (`tool_need`, `tool_rank`) | [Tools](../tutorials/plugin/tools) |
| `escalate` | Whether an answer is acceptable (`answer_check`) | [Escalate](../tutorials/algorithm/looper/escalate) |

[How the Router uses System One](./router) shows the call each feature makes.

## Model families

| Family | Models | Answers | Runs on |
| --- | --- | --- | --- |
| Vela 3.0 | `vela3-fast` (the default), `vela3-lite`, `vela3-mini`, `vela3-flash`, `vela3-turbo` | choice, noul and span, with trained adapters for the built-in questions; the larger sizes also answer score and read images | CPU (vela3-fast) or GPU |
| Decision 3.0 (d3) | `d3-edge` to `d3`, `d3-turbo` and `d3-ultra`, which is also hosted | choice, noul and score, over text, images and video | GPU, or a hosted API |
| Decision 2.0 and 1.0 | 0.6B to 27B | choice, noul, score | CPU (0.6B) or GPU |
| Vela 2.0 | 0.3B to 9B | choice, noul, score, set, span; trained for the built-in questions | CPU (0.3B) or GPU |

The [support matrix](./reference#support-matrix) shows which questions each
family answers trained, zero-shot or not at all.

## In this chapter

- [Concepts](./concepts): question types, the state and its parts, presets,
  trained and zero-shot answers, span capability and limits.
- [The System One API](./api): the standard request, the batch form, model
  cards, errors and limits.
- [Choose a deployment](./deployments): local, attached or remote models,
  where a question's deployment comes from, and what the Router checks at load.
- [How the Router uses System One](./router): the call behind every feature,
  how calls are merged, caps on judged candidates, and recall against judging.
- [Write custom questions](./custom-questions): turn a routing idea into a
  `systemone` rule, test it directly and tune it.
- [Performance and cost](./performance): what a request costs, and the knobs
  that bound it.
- [Reference](./reference): the preset catalog, the support matrix,
  configuration fields and load-time codes.
