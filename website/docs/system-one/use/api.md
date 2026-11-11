---
title: The System One API
sidebar_label: The API
sidebar_position: 1
description: A single request, the batch endpoint, roles, media, model cards, errors and limits of the System One API (vllm-sr/5.0).
---

# The System One API

Every System One model answers the same HTTP API, whether the built-in model
runtime serves it, you run it yourself, or a hosted service does. The Router
speaks it for you; use this page to check a model by hand, to write a client,
or to understand what the Router sends.

vllm-sr speaks `vllm-sr/5.0`: the published System One standard,
`systemone/0.2`, plus a set of extension fields in their natural places. Any
standard client or SDK works unchanged; SDK users pass the extension fields
with `extra_body`. A server that must keep serving clients of the earlier
contract speaks `vllm-sr/4.0`, which accepts both forms; see
[System One schemas](./schemas.md).

## Try it

Serve a model with the [model runtime](../../model-runtime/get-started/quickstart.md)
and ask it one question:

```bash
vllm-srun serve vllm-sr/vela3-fast --device cpu --port 8100
curl -s localhost:8100/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Our export button fails with a 500 since this morning. Open a ticket.",
  "questions": {"urgent": {"type": "noul",
    "instructions": "Does the user report a problem that blocks their work?"}}
}'
```

The answer to `urgent` is P(yes), for example `{"type": "noul", "noul": 0.88}`.

## Endpoints

| Endpoint | Does |
| --- | --- |
| `POST /v1/systemone` | Answers one request. A question that cannot be answered fails the request as a whole |
| `POST /v1/systemone/batch` | Answers named requests in one call, each as it would be answered alone |
| `POST /v1/decisions` | The native form of the same call, which the Router uses: the internal request form, and a failed question carries its own error in a 200 |
| `GET /v1/models` | Lists the models a server answers with, each with its `version`, `features` and `card` |

A runtime reports its contract as `api_version` in `GET /health` (`5.0.0`).

:::note
The Router refuses a runtime of any other contract: a Router and its model
runtime come from one release. The [HTTP API reference](../../model-runtime/reference/http-api.md#decisions)
documents every field the runtime reads; its OpenAPI contract is the source of
truth, and the [extension reference](../reference/extension-reference.md) lists
every field vllm-sr adds to the standard.
:::

## A single request

A request is one state and the questions about it:

```json title="POST /v1/systemone"
{
  "model": "vllm-sr/vela3-fast",
  "state": {
    "request": "Our export button fails with a 500 since this morning. Open a ticket.",
    "history": "How do I export my invoices as CSV?"
  },
  "roles": {"request": "user", "history": "context"},
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {
        "billing": "Payments, invoices and refunds",
        "engineering": "Outages, errors and bugs",
        "other": "Anything else"
      }
    },
    "urgent": {
      "type": "noul",
      "instructions": "Does the user report a problem that blocks their work?"
    }
  },
  "options": {"deadline_ms": 1500}
}
```

| Field | Meaning |
| --- | --- |
| `state` | Text, an object or an array. Its keys carry no meaning of their own |
| `roles` | Extension. Which state keys hold the `user`'s content, `context`, or the `answer` under check, for role-aware families (Vela). Without `roles`, the whole state is the user's content on every family |
| `questions` | Any number of named questions. Answers come back under the same names, in the same order |
| `assets` | Images and videos by placeholder (see [Media](#media)) |
| `model` | Optional while a server answers with one model |
| `options` | Extension. `deadline_ms` bounds the call (the Router sets it from the rule's `timeout_ms`), `return_meta` adds `meta`, and `max_tokens` is the scan budget of a model that reads a long part in windows |

A request that reads past the model's validated length sends the standard
header `TypeSafe-Danger-Zone-Long-Context: true`. A server answers any member
the schema does not define with 400, naming it.

How a family turns an object state into model input is the family's own, like
a chat template: Decision 3 reads it as `json.dumps` text, Vela reads its parts
by role. Send `roles` for a role-aware family; when the Router builds a
multi-part state itself, it sends the roles its part names (`request`,
`history`, `previous_turns`, `previous_answer`, `tool_results`, `candidate`,
`answer`, `context`) have always had.

## The batch request

A batch is a set of named requests in one call, for questions about different
states:

```json title="POST /v1/systemone/batch"
{
  "model": "vllm-sr/vela3-fast",
  "requests": {
    "doc_1": {
      "state": {"request": "How long is the refund window?", "candidate": "Refunds are accepted within 30 days of purchase."},
      "roles": {"request": "user", "candidate": "context"},
      "questions": {"relevance": {"type": "noul", "instructions": "Does the candidate passage help answer the request?"}}
    },
    "doc_2": {
      "state": {"request": "How long is the refund window?", "candidate": "Our offices close on public holidays."},
      "roles": {"request": "user", "candidate": "context"},
      "questions": {"relevance": {"type": "noul", "instructions": "Does the candidate passage help answer the request?"}}
    }
  },
  "options": {"deadline_ms": 3000}
}
```

- Every entry of `requests` is a complete request: its own `state`, `roles`,
  `questions` and `assets`. It may repeat the batch's `model`; `options` belong
  to the batch.
- The response is `{"model", "responses", "usage"}`: each request's response
  under the same name, as the request alone would get it, and a `usage` that
  sums every request.
- A failed question answers the batch as a whole, with its request's name as
  the error's `state`. `/v1/systemone` takes exactly one request.

## Questions

| Field | Type | Meaning |
| --- | --- | --- |
| `type` | all | `choice`, `noul`, `score` and `box` (the standard's), and the extension types `set` and `span`, and `track` where a card declares it |
| `instructions` | all | The question, in plain language |
| `criteria` | `choice`, `noul`, `score`, `set`, `span` | A `choice`'s options (an object of keys to descriptions), a `noul`'s `true` and `false`, a `score`'s levels (an array, lowest first), or a `set`'s or `span`'s labels |
| `preset` | all | Extension. A question the model was trained on (for example `pii`), listed in its card's `presets`; the question still carries its own `instructions` and `criteria`, which any other server answers |
| `threshold` | `set`, `span` | Extension. Replaces the model's own decision threshold; the applied value comes back in `thresholds` |
| `head` | `span` | Extension. `router` or `broad`, on models with a broad span head |
| `over` | all | Extension. The state keys the question reads |
| `overflow` | all | Extension. `window` (read a long part whole, in windows) or `truncate` (its first tokens), on models with a scan budget |
| `require_full_input` | all | Extension. Fails the question unless the whole input is read; the answer then carries `input_coverage: complete` |

## Media

Images and videos are `assets`, each under a placeholder string that stands
for it where it appears. A placeholder may sit inside the state, or in a
question's `instructions` or `criteria` on models whose card allows it.

The Router places a turn's media in front of the state:
`"state": ["⟦img-1⟧", "⟦vid-1⟧", "What happens here?"]`. A vllm-sr/5.0 server
reads such a state, whose items are image placeholders, then video
placeholders, then exactly one other item, as that item with the media in front
of it, as Decision 3 reads a turn's images.

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

## Model cards

`GET /v1/models` lists each model with the standard's `name`, `description`
and `release_date`, and the extensions `version` (`5.0`), `features` (the
question types it answers) and `card`. The Router reads the card when the
configuration loads, or when an attached runtime starts, and checks every rule
against it:

| Card field | The Router checks |
| --- | --- |
| `question_types` | The deployment answers every question type a rule resolving to it asks |
| `presets` | A span-needing preset (`pii`, `halu`) is answered by the model's own question where the card lists it |
| `modalities` | A rule with `input.image` or `input.video` resolves to a model that reads the medium |
| `limits` | Options, labels and levels within the model's ranges; input token limits |

A remote API that lists no card for the model is described by the deployment's
`capabilities` instead (see
[Choose a deployment](../deploy/deployments.md#remote-system-one-apis)).

## Errors and limits

An error is the standard `detail`, with the extension's flat `code`,
`question`, `state` (in a batch, the request's name), `reason` (a fixed word)
and `limit`. On a 422 `detail` is the standard's list, each item locating the
failed question:

```json title="422"
{"detail": [{"loc": ["body", "questions", "team"], "msg": "question \"team\" cannot be answered: the question is malformed", "type": "invalid_question"}],
 "code": "question_failed", "question": "team", "reason": "question_malformed"}
```

On `POST /v1/systemone`, the first question that cannot be answered, in request
order, answers the whole request with `code` `question_failed`:

| Question error | Status | Means |
| --- | --- | --- |
| `invalid_question` | 422 | A malformed question, or a type, field or preset the model does not answer |
| `max_length_exceeded`, `scan_budget_exceeded`, `invalid_input` | 400 | The input is longer than the model reads, or malformed |
| `invalid_model_output` | 500 | The model produced no valid answer |
| `unavailable` | 503 | The model is not ready |
| `deadline_exceeded` | 504 | `options.deadline_ms` passed |

- A request that is too large as a whole is refused with 413
  `request_too_large`; bodies may be up to 32 MiB.
- An overloaded server answers 529 with `Retry-After`; a server that is
  starting answers 503 with `Retry-After`.
- A body with a member the schema does not define is refused with 400
  `invalid_request`, naming it.
- Per question, choices number 2 to 255, labels 1 to 255 and levels 2 to 10;
  there is no limit on the number of questions.

When the Router gets an error, the questions of that call are unknown: each
rule's or feature's failure policy decides what that means. A judge that has
too many candidates never fails a request.
