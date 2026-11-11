---
title: System One concepts
sidebar_label: Concepts
sidebar_position: 2
description: Question types, the state and its parts, presets, trained and zero-shot answers, span capability and the protocol's limits.
---

# System One concepts

A System One call has two halves: the **state**, the context the questions are
about, and the **questions**, each with a name and a type. The model answers
every question with probabilities, and a rule matches on those numbers.

```json title="One state, two questions"
{
  "state": {"request": "My card was charged twice.", "history": "Where is my order?"},
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "choices": [{"key": "billing", "description": "Payments and refunds"},
                         {"key": "other", "description": "Anything else"}]},
    "urgent": {"type": "noul", "instructions": "Does the user report a problem that blocks them?"}
  }
}
```

## Question types

| Type | Asks | You write | The answer | A rule reads |
| --- | --- | --- | --- | --- |
| `choice` | Which of these options fits? | 2 to 255 `choices`, each a `key` and a `description` | The chosen option and every option's probability | P(chosen option), or P(option) for a label |
| `noul` | Is this true? | The `instructions` only | P(yes) | P(yes) |
| `score` | How much, on an ordered scale? | 2 to 10 `levels` | The expected level (counted from 0) and each level's probability | The expected level |
| `set` | Which of these labels apply? | 1 to 255 `labels` | Each label's probability, and the labels the model selects | P(label) |
| `span` | Where in the text is each label? | 1 to 255 `labels` | Labelled character spans, each with its probability | The most probable span of a label |

- `noul` is System One's yes/no type.
- A model with no native `set` answers a set question as one `noul` per label,
  in the same call.
- Some models declare more types: d3-turbo and d3-ultra answer `box` (where
  something is in an image) and `track` (where it is in each moment of a
  video) on the [API](../use/api.md). Routing rules do not ask them.

:::tip
Write instructions the way you would ask a colleague, and describe each option
in a few words. [Write custom questions](../use/custom-questions.md) shows what
reads well to a model.
:::

## The state and its parts

The state is the text a question is about. For a question about the current
user turn alone, it is that text. When a question reads more, the state is an
object of named parts: the model reads `request` as the user's turn, `answer`
as the model's answer and every other part as context.

| Part | Holds | Asked by |
| --- | --- | --- |
| `request` | The current user turn: the routing text, after `global.router.routing_input` removes what it is told to | Every request-stage question |
| `history` | The earlier user turns | Rules with `input.history` |
| `previous_turns` | The last N turns, user and assistant, as a transcript | Rules with `input.previous_turns`; `turn_relation` reads 4 by default |
| `previous_answer` | The assistant answer the current turn replies to | `user_feedbacks` (the `feedback` preset) |
| `tool_results` | The request's tool results | Rules with `input.tool_results` |
| `candidate` | The item a judge scores: a document, a segment, a cached request | `relevance` and `same_request` judges |
| `answer` | The model's answer | Response-stage rules; `answer_check` |
| `context` | The answer's grounding context: retrieved documents and tool results | Response-stage rules with `input.context`; `hallucination` |
| `images`, `videos` | The current turn's image and video parts | Rules with `input.image` or `input.video` |

A part with nothing in it is left out, so a rule that reads the history of a
first turn asks about the request alone.

### Stages

A question is asked at one of two stages:

- **request** (the default): before routing, about the request. Decisions,
  projections and request plugins read request-stage rules.
- **response**: after the model has answered, about its answer. Only plugins
  read response-stage rules.

:::note
A decision routes the request before an answer exists, so a decision whose
rules read a response-stage signal is refused at load.
:::

## Presets

A preset is a canonical zero-shot question the Router owns: its instructions,
its options with descriptions, the stage and parts it reads, how much of a long
input it covers, and whether it needs spans. Presets are internal; you reach
them through the features that ask them:

- the eleven **application signals**, each of which compiles a rule into one
  question of its preset;
- the **`match` judges** `relevance`, `same_request` and `tool_rank`, and the
  tools plugin's `tool_need` gate;
- **`escalate`**, whose `answer_check` and `refusal` checks judge answers.

A question of your own is a [`systemone` rule](../../tutorials/signal/systemone.md).
The [preset catalog](../reference/presets.md#preset-catalog) lists every
preset.

Rules that ask the same preset of the same deployment about the same parts
share one question per call. Each rule then applies its own `labels` and
`predicate` to the shared answer.

## Trained and zero-shot answers

Every model answers a preset in its canonical form. Where a model family was
trained on a preset, the Router asks it in the trained form instead:

- **Trained bindings.** The Vela 3.0 d3 sizes turn a trained adapter on by
  exact question ID (`domain`, `attack`, `factcheck`, ...); vela3-fast and
  Vela 2.0 pick it by exact wording, which for every preset they were trained
  on is the canonical wording. Trained option keys are mapped back: the
  modality question's trained `AR`, `DIFFUSION` and `BOTH` come back as
  `text`, `image` and `text_image`.
- **Router-owned IDs.** A question without a trained form goes out under an ID
  with a `:` in it, which cannot collide with a trained ID and so never turns
  on an adapter by accident.
- **Calibrated thresholds.** A preset's default predicate on vela3-fast and
  Vela 2.0 is the threshold that keeps the Vela 1.0 specialists' operating
  point on the router signal suite.

:::note
Every other model uses 0.5 as a preset's default predicate, and the
configuration load warns with `uncalibrated_preset`, naming the rule. Write
`predicate` on the rule to set your own.
:::

The [support matrix](../reference/support-matrix.md) shows, for each preset and
family, whether answers are trained and calibrated, trained, zero-shot or
impossible.

## Span capability

Span questions return labelled character offsets. The Vela 3.0 and Vela 2.0
families answer them; the d3 family, local or hosted, and Decision 1.0 and 2.0
do not. Three things need spans:

- the `pii` and `hallucination` presets;
- every `systemone` rule whose question type is `span`;
- the `masking` plugin, through the `pii` rule it names.

Each of them must resolve to a span-capable deployment. With the default
decision model nothing is needed. With a decision model that has no spans,
bind the span-needing types to one that does.

:::warning
Otherwise the configuration fails to load with `span_required`:

```text
recipes[default].signals.pii[personal_data]: span_required: resolves to deployment "d3-ultra-remote", which cannot answer span questions; bind pii to a span-capable deployment (rule deployment, recipes[].model_bindings.pii or global.model_catalog.bindings.pii)
```

:::

## Media inputs

A rule reads the current turn's images with `input.image` and its videos with
`input.video`. The answering deployment must declare that it reads the medium,
or the configuration fails to load with `input_modality_unsupported`.

- The d3 family and the Vela 3.0 d3 sizes read images.
- `input.audio` and `input.video` are part of the schema, and a model that
  declares them answers such rules without a schema change.
- No current decision model reads audio, so `input.audio` is refused on every
  deployment until one does.

## Limits

| Limit | Value |
| --- | --- |
| Options of a `choice` | 2 to 255 |
| Labels of a `set` or `span` | 1 to 255 |
| Levels of a `score` | 2 to 10 |
| Questions in one call | No protocol limit |
| Candidates of a choice judge (`tool_rank`) | At most 255, the option limit |
| Candidates of a per-candidate judge (`relevance`, `same_request`) | `judge.max_candidates`, 64 by default, at most 4096 |

A judge over more candidates than its cap truncates; it never fails the
request. Input length is a property of the model: vela3-fast reads 32,768
tokens per decision and 8,192-token windows for router and span questions, and
the larger Vela 3.0 sizes read up to 262,144 tokens per decision and 32,768 per
span target. [Performance and cost](../deploy/performance.md#long-inputs)
covers long inputs.
