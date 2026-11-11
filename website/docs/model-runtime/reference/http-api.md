---
title: HTTP API reference
sidebar_label: HTTP API
sidebar_position: 2
description: "The worker API of the model runtime: endpoints, responses and errors, long inputs, classify, embeddings, rerank, bundle, decisions, and System One compatibility."
---

# HTTP API reference

Every `vllm-srun` worker serves one HTTP API. The Router frontend publishes
its System One part to clients; the rest is a worker API for runtimes you run
yourself.

## Endpoints {#http-api}

The following table describes the **worker** API. The frontend exposes
`POST /v1/systemone` and `/v1/decisions`, and discovers published native models
at `GET /v1/systemone/models`; requests use each deployment's public model
name and the listener's API key and native grants. Chat model discovery remains
`GET /v1/models`. Worker-only classification, embedding, rerank and bundle
endpoints are not automatically published by the frontend.

Every request may name its `model`; it is required when a process serves
several models. The contract is the OpenAPI document served at
`GET /openapi.yaml`.

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/decisions` | Answer Choice, Noul, Score, Box, Set and Span questions about a state; a failed question is answered in place. It reads the runtime's internal request form, the forms below plus `roles`, whatever the process's System One schema. |
| `POST /v1/systemone` | One request in the process's System One schema, answered as System One answers it: a request with a failed question fails as a whole ([System One compatibility](#system-one-compatibility)). |
| `POST /v1/systemone/batch` | Named requests in one call, each answered as it would be alone, under `responses`. |
| `POST /v1/classify` | Run a classification head over texts, text pairs or grounded answers: label probabilities, independent label scores or token spans. |
| `POST /v1/embeddings` | OpenAI-compatible embeddings of text, images and audio, with `dimensions` and `layer`. |
| `POST /v1/rerank` | Score documents against a query. |
| `POST /v1/bundle` | Several of the above, for one or more models, in one call. |
| `GET /v1/models` | What each model is, what it serves, its labels, limits, device and self-check status, and the process's `limits`: the most tasks one `/v1/bundle` may carry and the largest request body. With `/health` and `/health/live`, it reports the contract's `api_version` (`5.0.0`). `models` lists the decision models as `{name, description, release_date}`, the shape the System One SDKs read, with `version`, `features` and `card` under `vllm-sr/5.0`. |
| `GET /health` | Readiness: 200 when every model is ready, 503 otherwise. `status` is `starting`, `loading`, `warming`, `ready`, `degraded` or `failed`, and a process that serves several models lists each model's state. |
| `GET /health/live` | Liveness: 200 with `"status": "alive"` while the process serves requests, whether or not its models are ready; 503 with `"status": "stalled"` and a `reason` while a forward has run longer than `--forward-timeout-seconds`. |
| `GET /metrics` | Prometheus metrics. |

A response carries the answers and `usage`. Add
`"options": {"return_meta": true}` to a request to also get `meta`: the
revision, model digest, `dtype`, `numerics`, engine, device and timings that
answered it. `numerics` is `exact` when the model's answers never depend on
batching (its rows answer the same alone or in any batch, at its published
precision), and `approximate` when the rows a forward holds can move them by
rounding. `GET /v1/models` reports a model's `engine_args` and `numerics`.

The responses of `/v1/decisions`, `/v1/systemone`, `/v1/classify`,
`/v1/embeddings`, `/v1/rerank` and `/v1/bundle`, errors included, carry the
runtime's own time for the request in a `Server-Timing` header, in
milliseconds:

```text
Server-Timing: parse;dur=0.021, tokenize;dur=0.153, queue;dur=0.008, forward;dur=4.871, post;dur=0.034, serialize;dur=0.019, total;dur=5.141
```

| Phase | Time spent |
| --- | --- |
| `parse` | Reading and decoding the request body. |
| `tokenize` | Validating, rendering and tokenizing the request. |
| `queue` | Waiting for the model, before its first forward and between its forwards. |
| `forward` | Running its forwards, readout included. |
| `post` | Assembling the answers. |
| `serialize` | Encoding the response. |
| `total` | From the handler's start until the response is ready to send. What it holds beyond the phases is the server's own overhead. |

A bundle whose tasks go to several models reports the `queue`, `forward` and
`post` of the model that answered last. A client's own time for the call minus
`total` is the transport: its encoding and decoding, the connection and the
HTTP exchange. The router records it as a metric for every call.

A request that cannot be served at all returns an HTTP error with
`{"error": {"code", "message"}}`: 400 `invalid_request`, 404
`model_not_found`, 413 `request_too_large`, 415 `unsupported_media_type` (a
`Content-Encoding` other than one layer of gzip or zstd), 422
`unsupported_surface` (the model does not serve that endpoint), 429
`overloaded`, 503 `not_ready`. A rejected image, video, asset or body adds its
`reason` (for example `video_too_long`, `image_too_large`, `asset_unreferenced`
or `encoding_malformed`) and, when it broke a limit, the `limit`. A forward
that runs out of device memory answers 413 `request_too_large` when the
request had the forward to itself and 429 `overloaded` when it shared it with
other requests. These fixed words never carry request data, so the Router's public
System One API answers with them and keeps the runtime's message private: it
says `video longer than 300 s` where the runtime says why in its own words.
The Router answers a deployment whose workers are all busy the same way, 429
`overloaded`, and one with no ready worker 503 `not_ready`; both carry
`Retry-After` (1 and 5 seconds). In Engine mode a call that runs past the
Router's 30-second budget returns 504 `systemone_deadline_exceeded`, with a
message that names `options.deadline_ms`; a request whose `options.deadline_ms`
is longer waits that long instead, and a long-context request without one
waits the listener's whole `apis.systemone.max_deadline_ms` (10 minutes by default),
which bounds both.
A failed item of a request (one input or one question) carries its own error
code and never fails the others; a failed question also carries a fixed
`reason`. `POST /v1/systemone` is the exception: there a failed question fails
the request ([System One compatibility](#system-one-compatibility)).

## Long inputs

A model reads at most its token budget of an input: `options.max_tokens`, at
most the model's `max_input_tokens`. The runtime tokenizes a longer input only
as far as that budget decides the answer, so the memory and time an input
costs follow its budget, not its length:

- It cuts the text before a word boundary (whitespace, punctuation, a symbol,
  or a Chinese, Japanese or Korean character) and keeps the tokens of the
  words before the cut. In a run with no boundary the tokenizer splits at,
  such as base64, hex, or Chinese without spaces in a tokenizer that splits
  words only at spaces, it cuts before a letter or digit and keeps the tokens
  that end 1,024 characters before the cut.
- Under `reject`, a text longer than its budget times the most characters one
  of the model's tokens covers fails without being tokenized.

These are the tokens that reading the whole text gives, so the answer is the
same. The runtime checks each kind of cut on probe texts the first time it
uses a tokenizer and reads whole texts where a cut fails the check. A
tokenizer that drops or folds characters, as WordPiece drops whitespace and
reads a word of more than 100 characters as one unknown token, never cuts
inside a run: it reads such a run whole, which costs little because the run is
a few tokens. The usage of an input read in part counts the tokens read, which
exceed the budget, and adds `"tokens_lower_bound": true`.

A model that reads an input in windows reads it up to a scan budget, in
tokens. Classify with `overflow: window` reads at most `max_tokens`. A
Vela 2.0 question that supports windowing reads a long state part whole up to its card's
`limits.max_scan_tokens`: four inputs on a CPU (32,768 tokens for Vela 2.0
0.3B) and 32 on a GPU. A decisions request's `options.max_tokens`, or the
models-file option `max_scan_tokens`, sets another. A question with
`overflow: truncate` reads only the part's first `limits.truncate_tokens`
instead: one input on a CPU, one forward. A Vela 3.0 evidence span question
truncates to its own 32,768-token window instead. Questions of both kinds share their
model input, as they would without `overflow`, unless a part has to be read
in windows. An input over its scan budget fails instead of being read:

| Item error | Meaning |
| --- | --- |
| `max_length_exceeded` | The input has more tokens than its budget under `reject`. |
| `scan_budget_exceeded` | The input is one the model reads in windows, and it has more tokens than its scan budget. None of it was read. |

The only byte limit is `--max-request-bytes`, on a request as a whole.

The router reads a long request in two ways before the model does (see
[Long inputs](../../system-one/deploy/performance.md#long-inputs)). A routing
question (domain, fact check, feedback, modality, `systemone` questions
without `coverage: full`, and the `systemone` selector) reads a bounded view,
the head and the tail within the deployment's `routing_view`. A guard question
(jailbreak, safety and PII) reads it whole, in windows within the deployment's
`scan_window`, each of which the model reads whole, up to its scan budget or
the deployment's `input.max_tokens` with `overflow: window`. When the signals'
deadline (`global.model_catalog.signal_timeout_ms`) ends a scan early, a
jailbreak or PII rule decides on the windows read
(`on_unscanned: best_effort`); when no window answers, it matches unless it
sets `on_unknown: no_match`. A hallucination question reads its input whole,
up to the model's scan budget; content past it was not read.

Token budgets describe input coverage, not response time. Complete long-input
embeddings and scans can exceed the signal deadline on a CPU with few cores.
Measure latency at your input lengths and concurrency; choose sufficient CPU
capacity or GPU placement and tune [worker threads](../deploy/placement-and-scaling.md#cpu-threads)
for that traffic. Increasing a deadline does not make inference faster.

A deadline bounds the caller's wait. It does not immediately interrupt a model
forward already running, which can continue to occupy the worker and delay
queued requests. One API call may contain several questions or scan windows
and require multiple forward passes.

## Classify

```json title="POST /v1/classify"
{
  "model": "vela-pii",
  "input": ["Hi, I'm Tom Baker (tom.baker@example.com)."],
  "options": {"overflow": "window", "window": {"tokens": 512, "overlap": 64}}
}
```

`input` is a string, a list of strings, a list of `{"text", "text_pair"}`
pairs, or a list of `{"context", "question", "answer"}` items for
hallucination checks. `options.overflow` is `reject`, `truncate` or `window`.
Span offsets count Unicode code points, end exclusive.

## Embeddings

```json title="POST /v1/embeddings"
{"model": "vela-embedding", "input": ["How do I reset my password?"], "dimensions": 256, "layer": 11}
```

`dimensions` and `layer` select a smaller vector or an earlier layer when the
model declares them. With `return_meta`, `meta.representation` identifies the
vector space, so you can keep vectors of different settings apart.

## Rerank

```json title="POST /v1/rerank"
{"model": "vela-reranker", "query": "How do I reset my password?",
 "documents": ["Open Settings, then Security.", "Our offices are closed on Sunday."], "top_n": 1}
```

## Bundle

```json title="POST /v1/bundle"
{
  "tasks": [
    {"id": "domain", "classify": {"model": "vela-domain", "input": ["What is 2 + 2?"]}},
    {"id": "kind", "decisions": {"model": "decision-edge", "state": "What is 2 + 2?",
      "questions": {"math": {"type": "noul", "instructions": "Is this a math question?"}}}}
  ],
  "options": {"deadline_ms": 80}
}
```

Results come back in task order, each with the status it would have had on
its own.

## Decisions

A request names a `state` and its `questions`. A question's options are
`criteria`, or the ordered list the Router config uses for the same thing:

| Type | `criteria` | Or, as in the Router config |
| --- | --- | --- |
| `choice` | `{key: description}`, 2 to 255 options | `choices: [{key, description}]` |
| `noul` | `{false: description, true: description}`, both optional | `choices: [{key, description}]` with keys `false` and `true` |
| `score` | `[description, ...]`, 2 to 10 levels | `levels: [description, ...]` |
| `set`, `span` | `{label: description}`, 1 to 255 labels | `labels: [{key, description}]` |

A question takes one form or the other, and only its type's fields. Vela 2.0
answered these two requests on CPU; the numbers are rounded.

```json title="POST /v1/decisions"
{
  "model": "Vela-2.0-0.3B",
  "state": "Write a Python function that merges two sorted lists and explain its running time.",
  "questions": {
    "domain": {
      "type": "choice",
      "instructions": "Which domain does this request belong to?",
      "choices": [
        {"key": "code", "description": "Programming"},
        {"key": "math", "description": "Mathematics"},
        {"key": "other"}
      ]
    },
    "reasoning": {"type": "noul", "instructions": "Does answering this request need multi-step reasoning?"},
    "difficulty": {
      "type": "score",
      "instructions": "How difficult is this request?",
      "levels": ["Trivial", "Moderate", "Hard"]
    }
  }
}
```

```json title="Response"
{
  "model": "Vela-2.0-0.3B",
  "answers": {
    "domain": {
      "type": "choice",
      "choice": "code",
      "confidence": 1.0,
      "probabilities": {"code": 1.0, "math": 0.0, "other": 0.0},
      "abstain_probability": 0.01
    },
    "reasoning": {"type": "noul", "noul": 0.34},
    "difficulty": {
      "type": "score",
      "score": 1.3,
      "confidence": 0.0,
      "legend": {"0": "Trivial", "1": "Moderate", "2": "Hard"},
      "probabilities": {"0": 0.13, "1": 0.43, "2": 0.43}
    }
  }
}
```

A Set answers each label as a Noul under `<id>.<label>`, and lists the labels
at or above its threshold in `sets`. A Span returns the text it found in
`spans`, with Unicode code point offsets, end exclusive, and a Noul under its
ID. `thresholds` gives the threshold each question applied; set `threshold` on
the question to choose another.

```json title="POST /v1/decisions"
{
  "model": "Vela-2.0-0.3B",
  "state": "My card was charged twice and the parcel never arrived. Write to me at jane.doe@example.com.",
  "questions": {
    "problems": {
      "type": "set",
      "instructions": "Which problems does the customer report?",
      "labels": [
        {"key": "billing", "description": "A payment or charge problem"},
        {"key": "shipping", "description": "A delivery problem"},
        {"key": "account", "description": "A login or account problem"}
      ]
    },
    "contacts": {
      "type": "span",
      "instructions": "Find the personal contact details.",
      "labels": [{"key": "EMAIL_ADDRESS", "description": "An email address"}]
    }
  }
}
```

```json title="Response"
{
  "model": "Vela-2.0-0.3B",
  "answers": {
    "problems.billing": {"type": "noul", "noul": 0.95},
    "problems.shipping": {"type": "noul", "noul": 0.7},
    "problems.account": {"type": "noul", "noul": 0.03},
    "contacts": {"type": "noul", "noul": 1.0}
  },
  "spans": {
    "contacts": [
      {"label": "EMAIL_ADDRESS", "start": 71, "end": 91, "text": "jane.doe@example.com", "probability": 1.0}
    ]
  },
  "sets": {
    "problems": {
      "selected": ["billing", "shipping"],
      "probabilities": {"billing": 0.95, "shipping": 0.7, "account": 0.03}
    }
  },
  "thresholds": {"problems": 0.3, "contacts": 0.65}
}
```

`confidence` is System One's published statistic. For a Choice with n options
it is `(p_max - 1/n) / (1 - 1/n)`: how far the chosen option sits above an even
split. For a Score with levels 0 to n - 1 it is
`max(0, 1 - sum_i p_i |i - m| / MAD_unif)`, where m is the most likely level and
`MAD_unif = sum_i |i - (n - 1)/2| / n` is the same spread for an even
distribution: probability on a neighbouring level lowers it less than the same
probability further away. Above, Vela is torn between Moderate and Hard, so the
Score's confidence is 0. A Noul has none: its `noul` already says how sure it
is.

An invalid question is answered `{"type", "error": "invalid_question",
"reason", "message"}`, where the message names the field, such as `set
questions do not take ['colour']`; the other questions are answered as usual.
`reason` is a fixed word, such as `question_malformed` or `box_unsupported`, that
never carries request data. A Choice or Score may omit `instructions` (the
model chooses the best matching option), and so may a Noul whose criteria
describe an outcome. Vela 3.0 reads a Noul by its instructions alone and
answers one that describes `false` or `true` with `invalid_question`. A request in which no question is valid is answered 400
`invalid_request`, with each question's reason in the message and the first
one's `reason` and `question`.

A request is a standard request, one `state` and its `questions` as above, or
a batch: named standard requests under `states`, answered in one call. Every
entry holds a `state` and its own `questions`, and may hold `images` and
`videos`; it is read exactly as a standard request with them and the batch's
`model`, `options` and `assets` would be, and is answered under the same name
in the response's `states`. An entry whose images, videos, assets or tokens
are refused fails each of its questions with `invalid_input` (reason
`input_malformed`) and the refusal's message, and the other entries are
answered; a batch every entry of which is refused is answered with the first
refusal. The two forms are mutually exclusive: a batch has
no top-level `state`, `questions`, `images` or `videos`, and a body that mixes
the forms is answered 400 `invalid_request`. The Router sends a batch when one
request stage asks a deployment about several texts: the candidates a
`relevance` or `same_request` judge checks, one entry each, or the earlier
messages a history-aware jailbreak or PII rule reads. Questions about one text
share one entry.

```json title="POST /v1/decisions"
{
  "model": "Vela-2.0-0.3B",
  "states": {
    "doc_1": {
      "state": {"request": "How long is the refund window?", "candidate": "Refunds are accepted within 30 days of purchase."},
      "questions": {"relevance": {"type": "noul", "instructions": "Does the candidate passage contain information that helps answer the request?"}}
    },
    "doc_2": {
      "state": {"request": "How long is the refund window?", "candidate": "Our offices close on public holidays."},
      "questions": {"relevance": {"type": "noul", "instructions": "Does the candidate passage contain information that helps answer the request?"}}
    }
  }
}
```

The response is `{"model", "states": {"doc_1": {"model", "answers", "usage",
...}, "doc_2": {...}}, "usage"}`: each entry's answers as its own request gets
them, and a `usage` that counts every entry. It has no `answers` of its own;
with `return_meta`, its `meta` is the one every entry carries.

A model whose card lists the `image` modality (Decision 3.0) also reads
`images`: base64 PNG, JPEG or WebP data URLs, each of at most
`limits.image_max_bytes` bytes and `limits.image_source_max_pixels` pixels.
Every question of the state sees them, in front of its text and in this order;
the model resizes each to at most `limits.image_max_pixels` pixels. A model
whose card lists the `video` modality also reads `videos`: base64 MP4, WebM,
QuickTime or Matroska data URLs, each of at most `limits.video_max_bytes`
bytes, `limits.video_max_seconds` seconds and `limits.video_source_max_pixels`
pixels per frame, after the images. The model reads `limits.video_fps` frames
per second, at least 4 and at most `limits.video_max_frames` spread over the
whole video, each at up to `limits.video_max_pixels` pixels; every pair of
frames is one group of input tokens after its timestamp, and the videos of a
request take at most `limits.video_max_tokens` input tokens. In a batch, each
entry carries its own `images` and `videos`. Their tokens count in `usage`. A malformed or over-limit image or video, or one sent to a model
without its modality, fails the request with 400 `invalid_request` and its
`reason`. A worker
started directly reads bodies of at most `--max-request-bytes`, so raise it to
send large videos; the Router gives the workers it manages 96 MiB.

A Decision 3.0 package whose `decision_config.json` declares a state-once
`attention_mode` (`prefix_bidir`, `prefix_causal` or `causal`) reads each
state once: its images, videos and text form one state block, and every
question branches from it. Asking more questions about one state then costs
only their own tokens, and `usage.input_tokens` counts the state once and each
question's block. Its answers are those of each question asked alone, up to
rounding. `vllm-sr/d3-turbo` is such a package.

Two more `decision_config.json` fields change how a package reads a request,
and its identity covers both. With `prompt_layout` `header-v1`, a text-only
request lists its valid questions in front of its state, numbered in request
order, Choice and Score questions with their options: the list is part of the
state block, so every question sees what else is asked. A request with images
or videos, or with assets or Box and track questions, keeps the layout above.
`temperature_by_questions` (`{"max_questions": [n_1, ..., n_k], "temperature":
[t_1, ..., t_k, t_rest]}`) sets a request's temperature from its number of
Choice, Noul and Score questions n: the `t_i` of the first `n <= n_i`, else
`t_rest`. Box and track answers keep the package's `temperature`.
`vllm-sr/d3-ultra` declares both: a one-question request answers at 1.046, any
other at 1.0.

```json title="POST /v1/systemone"
{
  "model": "d3-turbo",
  "state": "Is the attached receipt complete, and does the clip show it being paid?",
  "images": ["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."],
  "videos": ["data:video/mp4;base64,AAAAIGZ0eXBpc29t..."],
  "questions": {
    "complete": {"type": "noul", "instructions": "Does the receipt show a total, a date and a merchant?"},
    "paid": {"type": "noul", "instructions": "Does the clip show the receipt being paid?"}
  }
}
```

`src/model-runtime/tools/reference_examples.py --url <runtime>` sends this
section's requests to a runtime that serves Vela 2.0 0.3B and checks that the
answers match.

## System One compatibility

`POST /v1/systemone` and `POST /v1/systemone/batch` speak the process's System
One schema, which `vllm-srun serve --systemone-schema` sets:

- **`vllm-sr/5.0`** (the default): the System One standard plus the vllm-sr
  extensions. A request is decoded strictly, any other member refused with 400
  naming it; an object state without `roles` gives no part a role; assets whose
  placeholders lead the state (images, then videos, then one other item) are
  read in front of that item; `TypeSafe-Danger-Zone-Long-Context: true` reads a
  long context. Errors are the standard `detail` (a list on a 422) with flat
  `code`, `question`, `state`, `reason` and `limit`, and an overloaded process
  answers 529 with `Retry-After`. See [System One schemas](../../system-one/use/schemas.md).
- **`vllm-sr/4.0`**: the forms below as they are, and every 5.0 form; an
  error adds `detail` and the flat fields beside `error`. Use it for a runtime
  that serves clients of the earlier contract directly.

A typed question that names a `preset` beside its own `instructions` or
`criteria` (the 5.0 form) asks the model's trained preset when its card lists
it, and the question as written otherwise.

The internal form, which `/v1/decisions`, `/v1/bundle` and a `vllm-sr/4.0`
process read, accepts System One requests as JSON, compressed JSON or multipart
bodies, and answers with System One fields. It also accepts the extensions
above (`images`, `videos`, batches in `states`, Set and Span,
`choices` / `levels` / `labels`, `options`) and `roles`: absent, an object
state's keys name its parts; `null`, no part has a role; an object, the role of
each key. A model the process does not serve is answered with an error that
names the served models.

**Bodies.** A body may be compressed with one layer of `Content-Encoding`
`gzip` (or `x-gzip`) or `zstd` with a window of at most 8 MiB; the size limit
applies to the body as sent and again once decoded. Another coding, or more
than one, is 415. A body may be `multipart/form-data`: the first part,
`request`, holds the JSON request, and every other part holds one image's
bytes and is claimed by exactly one asset.

**Assets.** A top-level `assets` object holds images by reference, as
`{"type": "image", "data_base64": "..."}` or, in a multipart body,
`{"type": "image", "part": "<part name>"}`. Each reference is replaced
wherever it occurs in a string of the state, of a question's `instructions`
or of its options: a string equal to a reference is that image, a longer one
is split at each occurrence, and object keys are never replaced. Every asset
must be used, a reference has 1 to 1,024 bytes (65,536 for all of them), and
each use adds the image again: its tokens count where it sits. The format is
read from the bytes (PNG, JPEG or WebP; animated images are refused), and an
image may have up to 16 MiB. One rejected image fails the request.

```json title="POST /v1/systemone"
{
  "model": "d3-lite",
  "state": {"receipt": "<receipt>", "note": "Uploaded by the customer."},
  "assets": {"<receipt>": {"type": "image", "data_base64": "/9j/4AAQSkZJRgABAQ..."}},
  "questions": {
    "complete": {"type": "noul", "instructions": "Does <receipt> show a total, a date and a merchant?"}
  }
}
```

Where a model reads images is its card's `image_placements`. Every model with
the `image` modality reads the state's images: a use in the state is read as
one more image in front of the state, after `images`, and the state's text
names it `[image k]`. Where the card also lists `instructions` or `criteria`,
an image in a question's instructions or options belongs to that question
alone; on other models that question fails with `question_images_unsupported`
rather than being moved into the state, where it would change the other
questions' answers. Put an image several questions share in the state: it is
read once there.

**Box.** A `box` question locates what its instructions describe in an image:
the one image its instructions reference, else the state's only image
(`box_needs_image` and `box_target_ambiguous` otherwise). With `at`, a number
of seconds, it locates it at that moment of the state's only video, and a
`track` question at every sampled moment of it, `rate` times per second: 1
(the default) or 2. Answers:

```json title="Box answers"
{"type": "box", "box": {"top": 0.13, "left": 0.25, "bottom": 0.23, "right": 0.35}, "confidence": 0.81}
{"type": "box", "box": {"top": 0.2, "left": 0.1, "bottom": 0.6, "right": 0.5}, "confidence": 0.77, "time": 4.5}
{"type": "track", "track": [{"time": 0.0, "box": {"top": 0.2, "left": 0.1, "bottom": 0.6, "right": 0.5}, "confidence": 0.9, "visible": 0.98}]}
```

Edges are fractions of the image's height and width from its top-left corner,
as the image was sent. A box is always given, so pair it with a Noul when the
target may be absent; `visible` is the probability that the target is visible
at that moment. `confidence` follows the model card's formula and may be
`null`. A model answers `box` and `track` only where its card's
`question_types` list them; otherwise the question fails with `box_unsupported`
or `track_unsupported`. Where the card has `limits.box_video_max_seconds`, a
`box` question with `at` and a `track` question read a clip of at most that
many seconds: a longer clip, or one whose length cannot be read from its
container, fails that question with `box_video_too_long`. Every other question
about a video keeps `limits.video_max_seconds`.

**Errors.** On `/v1/systemone` a request any of whose questions fails is
answered as a whole, as System One answers it, so that every answer a client
receives has its question's type: `{"error": {"code": "question_failed", "question_error",
"reason", "question", "message"}}` (in a batch, with the question's entry as `"state"`),
422 for `invalid_question`, 400 for
`max_length_exceeded`, `scan_budget_exceeded` and `invalid_input`, 500 for
`invalid_model_output`, 503 for `unavailable` and 504 for `deadline_exceeded`.
The first failed question decides, and the message lists them all.
`/v1/decisions` answers each failed question in place.

**Long context.** A model whose card has `limits.validated_input_tokens` caps
each question (the state and that question) at that length, the one it was
validated at. A request with `"options": {"long_context": true}` may have up
to `limits.max_input_tokens` per question, slower and less accurate; a longer
question fails with `max_length_exceeded`, reason `question_tokens`. A model
without `validated_input_tokens` reads up to `max_input_tokens` either way. A
state-once model (`usage: state_once`) also caps each request at
`limits.max_request_tokens`: its state once and every question's own tokens,
with or without long context (the frame windows a video Box or track question
reads are bounded by the video limits instead); a longer request fails as a
whole with 400 `invalid_request`, reason `request_tokens`, and the cap as
`limit`.

In summary:

| Area | vLLM Semantic Router |
| --- | --- |
| Image formats | PNG, JPEG, WebP |
| Image size | up to 16,000,000 pixels and an aspect ratio of 200, read at the model's `image_max_pixels` (resized) |
| Images in questions and options | models whose `image_placements` list them; otherwise 422 `question_images_unsupported` |
| Box and track | models whose `question_types` list them; otherwise 422 `box_unsupported` or `track_unsupported` |
| A malformed question | 422 for the request on `/v1/systemone`; on `/v1/decisions` that question fails alone |
| Context | `validated_input_tokens` per question by default, `max_input_tokens` with long context; state-once models: `max_request_tokens` per request (the state once and every question) |
| Models | the served model IDs; any other model is answered with the served models |
| `GET /v1/models` | `{models}` and the runtime's cards (`data`) |
| Usage | the state once on models whose card says `usage: state_once`; each question's copy otherwise |
| Request size | 48 MiB on the Router's public API; `--max-request-bytes` on a worker |
