# System One Signal

## Overview

`systemone` asks a System One model a typed question about the request and turns the answer into a routing fact.
You write the question in plain language: pick one of a few options
(`choice`), yes or no (`noul`), or a level on a scale (`score`). Models that
answer them, such as Vela 2.0, also take `set` questions (which of these
labels apply?) and `span` questions (where in the text is each label?). The
model runs in the [built-in model runtime](../../../model-runtime/overview.md),
which the router starts for you, or behind a remote
[System One API](#ask-a-remote-system-one-model).

## Key Advantages

- A new question works as soon as you write it; there is no classifier to train.
- Answers come with probabilities, so routes can require a confident answer.
- A rule reads what its question needs: the current turn, the earlier turns,
  tool results, or the request's images and videos.
- Questions sharing a deployment and compatible input can be grouped in one
  request-stage call, including the [`pii` signal](pii.md#vela-20).
- A late or failed answer makes the signal unknown; the decision's failure
  policy determines whether routing continues or rejects the request.

## What Problem Does It Solve?

Fixed-label classifiers answer only the questions they were trained for. A
System One model answers "does this need step-by-step reasoning?" or "is this
about our product?" from the question text alone.

## When to Use

Use it for questions that need judgment about the whole request. Prefer
heuristic signals for structural facts (length, keywords, modality) and the
specialized learned signals (domain, PII, jailbreak) where they already answer
your question.

## Configuration

A question that names no `deployment` asks the Router's decision model,
`global.model_catalog.system.decision_model` (Vela 3.0 vela3-fast unless you
[choose another model](../../../model-runtime/choose-a-model.md#choose-a-size)). It joins the
request-stage batch with compatible built-in questions. Reusing a deployment
avoids loading another copy of the model; it does not guarantee one forward
pass for the complete routed request:

```yaml
routing:
  signals:
    systemone:
      - name: needs_tools
        question:
          type: noul
          instructions: Does answering this request need a tool call?
        predicate:
          gte: 0.7
```

The decision model is `{deployment: primary}` by default and can be a Vela or
Decision 1.0/2.0 deployment. The model must support every requested question
type. The [`systemone` selection algorithm](../../algorithm/selection/systemone.md)
asks the same decision model by default, so it can also choose a backend.

To ask another model, such as a Decision 2.0 model, name it as a
`model_runtime` deployment and give each question its `deployment`:

```yaml
global:
  model_catalog:
    deployments:
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: auto

routing:
  signals:
    systemone:
      - name: needs_reasoning
        deployment: decision-kai
        question:
          type: noul
          instructions: Does answering this request need multi-step reasoning?
        predicate:
          gte: 0.7
        timeout_ms: 1000
      - name: request_kind
        deployment: decision-kai
        question:
          type: choice
          instructions: What kind of request is this?
          choices:
            - key: code
              description: Writing, reviewing or debugging code
            - key: math
              description: Mathematics or quantitative reasoning
            - key: chat
              description: Anything else
        labels: [code, math]
      - name: difficulty
        deployment: decision-kai
        question:
          type: score
          instructions: How difficult is this request?
          levels: [Trivial, Moderate, Hard]
        predicate:
          gte: 1.5

  decisions:
    - name: hard-code
      priority: 200
      rules:
        operator: AND
        on_unknown: no_match
        conditions:
          - type: systemone
            name: request_kind
            label: code
          - type: systemone
            name: needs_reasoning
      modelRefs:
        - model: large-coder
```

Configurations written for earlier releases name this signal `decision`
(`routing.signals.decision` and `type: decision` conditions). They still
load, with a `renamed_signal_type` warning, and `vllm-sr config migrate`
rewrites them to `systemone`.

### Rule fields

| Field | Meaning |
| --- | --- |
| `name` | The rule's name, which conditions and projections use. |
| `question` | The typed question: `type`, `instructions`, and its `choices`, `levels` or `labels`. |
| `labels` | The options a `choice`, `set` or `span` rule matches. A condition can still name any option. |
| `ignore_labels` | `set` and `span` only: labels that never count toward the rule. |
| `predicate` | The probability or level the answer must meet (`gt`, `gte`, `lt`, `lte`). |
| `deployment` | The deployment that answers; without one, the decision model. |
| `stage` | `request` (default) or `response`. |
| `input` | What the rule reads besides the current user turn; see [Read more than the current turn](#read-more-than-the-current-turn). |
| `coverage` | `truncate` (default) reads what fits the model's input; `full` reads all of it. |
| `on_unscanned` | `coverage: full` only: what an answer that could not read the whole input means: `match`, `no_match` or `unknown` (default). |
| `timeout_ms` | The answer's deadline, up to 60000 ms; else the deployment's, else 1000 ms. |

### What a rule matches

A condition `{type: systemone, name: <rule>}` matches when the rule does; a
condition with a `label` matches when that option or label does:

| Question type | The rule matches when | A condition's `label` matches when |
| --- | --- | --- |
| `noul` | the probability of yes meets `predicate` (default `gte: 0.5`) | (takes no label) |
| `score` | the expected level meets `predicate` (required; levels count from 0) | (takes no label) |
| `choice` | the chosen option is one of `labels`, and its probability meets `predicate` when one is set; a rule without `labels` matches only through a label | the label is the chosen option, and its probability meets `predicate` when one is set |
| `set` | a counted label matches | the label's probability meets `predicate`; without one, the model selected the label |
| `span` | a counted label matches | a span with the label has a probability that meets `predicate`; without one, the model found a span with the label |

A `set` or `span` rule counts `labels`, or every label of the question
without them, less `ignore_labels`. A condition may add its own `predicate`;
for a condition with a `label`, it reads that label's value. The
configuration is refused when a `label` or an entry of `labels` or
`ignore_labels` is not an option of the question, when a condition names an
ignored label, and when an unlabelled condition names a `choice` rule
without `labels`.

The rule publishes these values, which a condition's `predicate` and a
[projection score](../../projection/scores.md) read:

| Question type | `systemone:<name>` | `systemone:<name>:<key>` |
| --- | --- | --- |
| `noul` | P(yes) | — |
| `score` | the expected level | — |
| `choice` | P(chosen) | P(key) for every option |
| `set` | the highest P among the counted labels | P(key) for every label |
| `span` | the highest probability among the counted labels | the label's most probable span (0 when none) |

A projection score reads them with `value_source: raw`: `name: <question>`
reads `systemone:<name>`, and `name: <question>:<key>` reads one option or
label of a `choice`, `set` or `span` question. The question is asked whenever
a used projection reads it:

```yaml
routing:
  signals:
    systemone:
      - name: difficulty
        question:
          type: score
          instructions: How much reasoning does a strong expert need to answer well?
          levels: [none, a little, multi-step, expert]
        predicate:
          gte: 2
      - name: needs
        question:
          type: set
          instructions: What does a good answer need?
          labels:
            - key: deliberation
              description: a derivation, proof or careful step-by-step check
            - key: tools
              description: calling external tools or functions
  projections:
    scores:
      - name: effort
        method: weighted_sum
        inputs:
          - type: systemone
            name: difficulty
            weight: 0.3
            value_source: raw
          - type: systemone
            name: needs:deliberation
            weight: 0.4
            value_source: raw
    mappings:
      - name: effort_band
        source: effort
        method: threshold_bands
        outputs:
          - name: effort_high
            gte: 0.9
  decisions:
    - name: deliberate
      priority: 200
      rules:
        operator: AND
        conditions:
          - type: projection
            name: effort_high
      modelRefs:
        - model: large-reasoner
```

### Read more than the current turn

A rule reads the current user turn. `input` adds named parts of the
conversation to what the model reads:

```yaml
routing:
  signals:
    systemone:
      - name: escalation
        question:
          type: noul
          instructions: Has the user asked for the same fix more than once?
        input:
          history: true
          previous_turns: 6
          tool_results: true
```

- `history`: the earlier user turns.
- `previous_turns` (0 to 32): the last turns of the conversation, user and
  assistant, as a transcript.
- `tool_results`: the request's tool results.
- `image` and `video`: the current turn's images and videos. The deployment
  must read that input: a model runtime model whose card lists the modality,
  or a `systemone` deployment that declares it in `capabilities.modalities`.
  Otherwise the configuration is refused with `input_modality_unsupported`.
  No deployment reads audio yet, so `input.audio` is always refused.
- `answer` and `context`: the model's answer and its grounding context, for
  a `stage: response` rule only.

Rules that read the same parts of the request share a call; a rule that
reads other parts travels as a further state of the same call.

### Read the whole input

A rule reads what fits the model's input and drops the rest
(`coverage: truncate`). With `coverage: full` the model reads all of it, in
windows when it is longer than one input. When the answer could not cover the
whole input (the scan budget ran out, the model could not tell its coverage,
or the input exceeds the model's length), `on_unscanned` decides what the
rule reports: `match`, `no_match`, or `unknown` (default), which leaves the
decision's failure policy to decide:

```yaml
routing:
  signals:
    systemone:
      - name: account_ids
        question:
          type: span
          instructions: Which spans are account or order numbers?
          labels:
            - key: account_number
              description: a customer account or order number
        coverage: full
        on_unscanned: match
```

### Questions about the answer

`stage: response` asks the question about the model's answer, with
`input.answer` and `input.context` naming what it reads. A response plugin
reads such a rule; a decision's rules cannot, because a decision routes the
request before an answer exists, and the configuration is refused if one
does.

### Ask a remote System One model

A question can also ask a System One API that the router does not manage: a
hosted Decision API, another vLLM-SR Engine, or a gateway in front of one.
Declare it as a `systemone` deployment in `global.model_catalog.deployments`
and name it as the question's `deployment`, as you would a `model_runtime`
deployment. It answers System One questions only: it is not one of the
`providers.models` that requests are routed to, and no model listing shows it.

```yaml alternative
global:
  model_catalog:
    deployments:
      d3-ultra-remote:
        provider: systemone
        endpoint: https://app.vllm-sr.ai/v1/systemone
        model: vllm-sr/d3-ultra
        auth:
          api_key_env: VSR_APP_KEY
        timeout_ms: 3000

routing:
  signals:
    systemone:
      - name: support_team
        deployment: d3-ultra-remote
        question:
          type: choice
          instructions: Which team should handle this ticket?
          choices:
            - key: billing
              description: Payments and refunds
            - key: engineering
              description: Outages and bugs
        labels: [billing]
```

`endpoint` is the URL the API answers `POST` requests on, with any path
prefix, and `model` is the model it answers with. The key comes from
`auth.api_key_env` or from `auth.api_key_file` (an absolute path, read when a
call needs it), never from the configuration itself, and goes as
`Authorization: Bearer <key>`. For a gateway that takes the key in its own
header, set `auth.header`, and `auth.prefix: ""` to send the key without a
prefix:

```yaml
global:
  model_catalog:
    deployments:
      gateway-d3:
        provider: systemone
        endpoint: https://gateway.example.com/Unified/v1/systemone
        model: vLLM-SR-d3-27B
        capabilities: {question_types: [choice, noul, score], modalities: [text, image]}
        auth:
          header: Ocp-Apim-Subscription-Key
          prefix: ""
          api_key_env: GATEWAY_KEY
```

- A request stage's questions to the deployment, including those of the
  built-in signals that ask it, go in one System One call: their options as
  System One `criteria`, any other text they read as a further `states`
  entry, and the call's deadline as `options.deadline_ms`. A System One API
  fails a call with a failed question as a whole and names the question, so
  the router asks the other questions once more within the same deadline:
  only the failed question's signal is unknown. Any other failure leaves
  every signal of the call unknown. Answers to the same questions about the
  same input are reused for up to 10 minutes, until the configuration
  changes.
- The question types, ready-made questions (`pii`, `halu`) and input
  modalities the model answers and reads come from `GET <endpoint>/models`,
  read when the configuration loads, when the API lists the model; else from
  `capabilities.question_types`, `capabilities.presets` and
  `capabilities.modalities`; else they are the System One types `choice`,
  `noul` and `score`, on text.
  A `set` question is composed from `noul` questions when the model has no
  native `set` support.
- A network round trip takes longer than a local model. A question without
  its own `timeout_ms` takes the deployment's `timeout_ms`, else the 1000 ms
  default. A slow, failing or unreachable API leaves its signals unknown; it
  never fails the request. The router logs each kind of failure once, without
  the key.
- `global.model_catalog.system.decision_model.deployment` can name a
  `systemone` deployment too. Questions and
  [System One selectors](../../algorithm/selection/systemone.md) that name no
  deployment then ask it, and so do the built-in signals that no binding
  gives another deployment. The hazard part of a safety rule, embeddings and
  reranking keep their own deployments. Module thresholds the configuration
  does not write are 0.5 on such a model, with an `uncalibrated_preset`
  warning. On a model that answers no span questions, PII and hallucination
  detection report a yes/no verdict without positions, with a `span_fallback`
  warning when the configuration loads; a configuration that masks PII is
  then refused. [Choose a model for a signal](../../../installation/choose-a-model-for-a-signal.md#s2-everything-on-a-remote-system-one-model)
  shows this setup.
- A `systemone` deployment takes no device, engine arguments, replicas or
  input budget.

### Set and span questions

A `set` question names labels and asks which apply; a `span` question asks
where in the request each label occurs. Both take `labels` instead of
`choices`, each with a `key` and an optional `description`, and conditions on
them name a label or, to match any counted label, none:

```yaml alternative
global:
  model_catalog:
    deployments:
      vela2:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-0.3B
        device: cpu

routing:
  signals:
    systemone:
      - name: support_topics
        deployment: vela2
        question:
          type: set
          instructions: Which topics does the request mention?
          labels:
            - key: billing
              description: payments, invoices or refunds
            - key: shipping
              description: deliveries, tracking or returns
            - key: other
              description: anything else
        ignore_labels: [other]
      - name: account_ids
        deployment: vela2
        question:
          type: span
          instructions: Which spans are account or order numbers?
          labels:
            - key: account_number
              description: a customer account or order number
          head: router

  decisions:
    - name: billing-with-account
      priority: 200
      rules:
        operator: AND
        conditions:
          - type: systemone
            name: support_topics
            label: billing
          - type: systemone
            name: account_ids
      modelRefs:
        - model: support-model
```

- `threshold` (0 to 1) replaces the model's own decision threshold for the
  question. Without it the model applies its calibrated threshold, which is
  what decides `selected` labels and found spans.
- `head` (`span` only) names the span head that answers on models with two:
  `router` (trained on PII, unsupported claims and toxic spans) or `broad`
  (open extraction). Without it the model chooses by its own rule.
- A rule's `predicate` replaces the model's selection: `gte: 0.8` on a `set`
  rule matches the labels whose probability is at least 0.8.
- A native `set` question is used when the model supports it. Otherwise the
  Router can compose the set from one `noul` question per label. The task
  catalog reports this as `composed_noul`, rather than native Set support.

The configuration loads only when the rule's deployment can answer it. A
`span` rule on a deployment that answers no span questions is refused with
`span_required`, as is a rule whose `input` reads a modality the deployment
does not (`input_modality_unsupported`). A model runtime deployment whose
model is known only once it is running (an attached runtime) is checked when
the runtime prepares the questions, with the same message. Decision 1.0/2.0
can therefore answer a routing `set` task through their Noul capability, but
cannot produce `span` locations. Structural support does not establish task
accuracy: evaluate your questions and thresholds with the selected model. The
public native System One API still accepts only the question types that model
serves.

On a Vela 3.0 D3 model, a question with no trained form asks with the
model's decision adapter when the package has one; other packages ignore the
request for it.

While the model is loading, overloaded or slower than `timeout_ms`, the signal
is unknown. `rules.on_unknown` on the decision, or `on_error: match | no_match`
on a condition, decides what an unknown answer means. Compatible questions
share a batch; distinct input states and later selection or response stages
can require additional calls and forwards. A shared call can remain active
while another caller still needs it. A deadline ends the caller's wait but
does not guarantee that an active model forward stops immediately.

Truncating to the deployment's scan budget shortens only what the rule reads;
it does not shorten the request sent to the selected Chat backend. Matched
System One rules are listed in the `x-vsr-matched-decision-model` response
header.

To choose a model, size and hardware, or to run the model on your own GPU
server, see [Decision models](../../../model-runtime/guides/decisions.md).
