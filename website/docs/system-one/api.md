---
title: The System One API
sidebar_label: The API
description: The standard request, the batch form, model cards, errors and limits of the System One API.
---

# The System One API

Every System One model the Router asks answers the same HTTP API, whether the
built-in model runtime serves it, you run it yourself, or a hosted service
does. The Router speaks it for you; this page is for checking a model by hand,
for writing a client, and for understanding what the Router sends.

| Endpoint | Does |
| --- | --- |
| `POST /v1/systemone` | Answers a request. A question that cannot be answered fails the request as a whole |
| `POST /v1/decisions` | The model runtime's form of the same call: the same bodies, and a failed question carries its own error in a 200 |
| `GET /v1/models` | Lists the models a runtime answers with, with their cards; a Router or Engine frontend lists its System One models at `GET /v1/systemone/models` |

The [model runtime reference](../model-runtime/reference#decisions) documents
every field the runtime reads; its OpenAPI contract is the source of truth.
The two forms below are contract 4.0.0, which a runtime reports as
`api_version` in `GET /health`. The Router refuses a runtime of any other
contract: a Router and its model runtime come from one release. The Router's
own public System One API takes the same two forms.

## The standard request

A standard request is one state and the questions about it:

```json title="POST /v1/systemone"
{
  "model": "vllm-sr/vela3-fast",
  "state": {
    "request": "Our export button fails with a 500 since this morning. Open a ticket.",
    "history": "How do I export my invoices as CSV?"
  },
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "choices": [
        {"key": "billing", "description": "Payments, invoices and refunds"},
        {"key": "engineering", "description": "Outages, errors and bugs"},
        {"key": "other", "description": "Anything else"}
      ]
    },
    "urgent": {
      "type": "noul",
      "instructions": "Does the user report a problem that blocks their work?"
    }
  }
}
```

- `state` is text, or an object of named parts: `request`, `history`,
  `previous_turns`, `previous_answer`, `tool_results`, `candidate`, `answer`
  and `context`
  (see [the state and its parts](./concepts#the-state-and-its-parts)).
- `questions` holds any number of named questions. Answers come back under
  the same names, in the same order.
- `images` and `videos` carry the turn's media to a model whose card lists
  that modality.
- `model` is optional while a server answers with one model.
- `options.deadline_ms` bounds the call; the Router sets it from the rule's
  `timeout_ms`.

To check a model by hand, serve it with the
[model runtime](../model-runtime/quickstart) and send it a request:

```bash
vllm-srun serve vllm-sr/vela3-fast --device cpu --port 8100
curl -s localhost:8100/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"request": "Our export button fails with a 500 since this morning. Open a ticket."},
  "questions": {"urgent": {"type": "noul",
    "instructions": "Does the user report a problem that blocks their work?"}}
}'
```

## The batch request

A batch is a set of named standard requests in one call, for questions about
different states:

```json title="POST /v1/systemone"
{
  "model": "vllm-sr/vela3-fast",
  "states": {
    "doc_1": {
      "state": {
        "request": "How long is the refund window?",
        "candidate": "Refunds are accepted within 30 days of purchase."
      },
      "questions": {
        "relevance": {
          "type": "noul",
          "instructions": "Does the candidate passage contain information that helps answer the request?"
        }
      }
    },
    "doc_2": {
      "state": {
        "request": "How long is the refund window?",
        "candidate": "Our offices close on public holidays."
      },
      "questions": {
        "relevance": {
          "type": "noul",
          "instructions": "Does the candidate passage contain information that helps answer the request?"
        }
      }
    }
  }
}
```

- Every entry of `states` is a standard request: its own `state`, its own
  `questions`, and its own `images` and `videos`. Question names are unique
  within an entry, and may repeat across entries.
- The batch's `model`, `options` and `assets` apply to every entry.
- The two forms are mutually exclusive. A batch body has no top-level
  `state`, `questions`, `images` or `videos`, and a body that mixes the forms,
  or a batch with an empty `states`, is refused with 400 `invalid_request`.
- The response is `{"model", "states", "usage"}`: each entry's answers under
  the same name, as that entry's own request would get them, and a `usage`
  that sums every entry. It has no top-level `answers`.
- An entry whose own request would be refused for its images, videos, assets
  or tokens instead fails each of its questions with `invalid_input`, and the
  other entries are answered. Only a batch whose every entry is refused is
  refused, with the first entry's error.

## Questions

| Field | Type | Meaning |
| --- | --- | --- |
| `type` | all | `choice`, `noul`, `score`, `set` or `span`; `box` and `track` where a card declares them |
| `instructions` | all | The question, in plain language |
| `choices` | `choice`, `noul` | The options in order, each a `key` and a `description`; a `noul`'s two keys are `false` and `true` |
| `levels` | `score` | The level descriptions, lowest first |
| `labels` | `set`, `span` | The labels, each a `key` and an optional `description` |
| `threshold` | `set`, `span` | Replaces the model's own decision threshold; the applied value comes back in `thresholds` |
| `head` | `span` | `router` or `broad`, on models with a broad span head |
| `preset` | all | A question the model itself defines (for example `pii`), listed in its card's `presets` |
| `over` | all | The state part, or parts, the question reads |
| `overflow` | all | `window` (read a long part whole, in windows) or `truncate` (its first tokens), on models with a scan budget |

`choices`, `levels` and `labels` are ordered forms of System One `criteria`;
a question takes one or the other. A question takes only its type's fields,
and any other field is refused with `invalid_question`.

## The response

```json title="Response"
{
  "model": "vllm-sr/vela3-fast",
  "answers": {
    "team": {
      "type": "choice",
      "choice": "engineering",
      "probabilities": {"billing": 0.04, "engineering": 0.93, "other": 0.03},
      "confidence": 0.9
    },
    "urgent": {"type": "noul", "noul": 0.88}
  },
  "usage": {"input_tokens": 212, "output_tokens": 0}
}
```

| Type | Answer fields |
| --- | --- |
| `choice` | `choice`, `probabilities` of every option, `confidence` |
| `noul` | `noul`: P(yes) |
| `score` | `score` (the expected level), `probabilities` of every level, `confidence`, `legend` |
| `set` | `sets.<question>`: each label's probability and the selected labels; a composed set also answers one `noul` per label under `<question>.<label>` |
| `span` | `spans.<question>`: each span's `label`, `start` and `end` (characters, end exclusive), `text` and `probability` |

A batch response carries these fields per entry, under `states.<name>`.

## Model cards

A runtime's `GET /v1/models`, and a remote API's `GET <endpoint>/models`, list
each model with a card. The Router reads it when the configuration loads, or
when an attached runtime starts, and checks every rule against it:

| Card field | The Router checks |
| --- | --- |
| `question_types` | The deployment answers every question type a rule resolving to it asks |
| `presets` | A span-needing preset (`pii`, `halu`) is answered by the model's own question where the card lists it |
| `modalities` | A rule with `input.image` or `input.video` resolves to a model that reads the medium |
| `limits` | Options, labels and levels within the model's ranges; input token limits |

A remote API that lists no card for the model is described by the
deployment's `capabilities` instead (see
[Choose a deployment](./deployments#remote-system-one-apis)).

## Errors and limits

On `POST /v1/systemone`, the first question that cannot be answered, in
request order, answers the whole request. The body's `code` is
`question_failed`, with the failed `question`, its `question_error`, a
`reason` and, in a batch, the name of the question's entry as `state`:

| Question error | Status | Means |
| --- | --- | --- |
| `invalid_question` | 422 | A malformed question, or a type, field or preset the model does not answer |
| `max_length_exceeded`, `scan_budget_exceeded`, `invalid_input` | 400 | The input is longer than the model reads, or malformed |
| `invalid_model_output` | 500 | The model produced no valid answer |
| `unavailable` | 503 | The model is not ready |
| `deadline_exceeded` | 504 | `options.deadline_ms` passed |

A request that is too large as a whole is refused with 413
`request_too_large`, and an overloaded server answers 429 `overloaded`. A
body that is not a standard request or a batch, such as one that mixes
`state` or `questions` with `states` or has an empty `states`, is refused
with 400 `invalid_request`.
Per question, choices number 2 to 255, labels 1 to 255 and levels 2 to 10;
there is no limit on the number of questions.

When the Router gets an error, the questions of that call are unknown: each
rule's or feature's failure policy decides what that means, and a request is
never failed by a judge that has too many candidates.
