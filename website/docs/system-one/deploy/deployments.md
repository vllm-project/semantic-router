---
title: Choose a deployment
sidebar_label: Choose a deployment
sidebar_position: 1
description: Run the decision model locally, attach a runtime you run, or ask a remote System One API, and what the Router checks at load.
---

# Choose a deployment

Every System One question is answered by a **deployment**: a named entry of
`global.model_catalog.deployments` that says which model answers and where it
runs. Declare a deployment once, then name it wherever it is used.

| You want | Use | Section |
| --- | --- | --- |
| The Router to run the model for you, on its CPU or a GPU | A local `model_runtime` deployment | [Local models](#local-models) |
| One GPU server shared by several Routers | A `model_runtime` deployment with an `endpoint` | [Attach a runtime you run](#attach-a-runtime-you-run) |
| A hosted Decision API or another vLLM-SR engine | A `systemone` deployment | [Remote System One APIs](#remote-system-one-apis) |

## Where a question's deployment comes from

The most specific level wins:

| Level | Where you write it | Applies to |
| --- | --- | --- |
| Rule | `deployment` on a rule, a selector (`algorithm.systemone`), a judge (`match.judge`) or a check (`escalate.accept.systemone`) | That rule or feature |
| Recipe | `recipes[].model_bindings.<type>` | The rules of one application signal type in one recipe |
| Global | `global.model_catalog.bindings.<type>` | The rules of one application signal type in every recipe |
| Default | `global.model_catalog.system.decision_model` | Every question the levels above leave open |

The binding keys are the application signal types: `domain`, `fact_check`,
`user_feedback`, `jailbreak`, `safety`, `pii`, `hallucination`, `refusal`,
`turn_relation`, `complexity` and `modality`.
[Choose a model for a signal](../../installation/choose-a-model-for-a-signal.md)
works through each level with examples the Router's tests load.

## Local models

A `model_runtime` deployment runs in the built-in
[model runtime](../../model-runtime/get-started/overview.md), which the Router
starts and supervises. The pre-declared `primary` deployment is vela3-fast on
the CPU; declare `primary` again to change its model or device:

```yaml title="config.yaml"
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/vela3-lite
        device: rocm:0
```

| You want | Model | Runs on |
| --- | --- | --- |
| The default: every built-in question trained, with spans | `vllm-sr/vela3-fast` | CPU; the loaded deployment holds about 7 GB |
| Lower latency on a small GPU, multilingual | `vllm-sr/vela3-lite` (0.8B) | GPU |
| More accuracy, with the built-in questions still trained | `vllm-sr/vela3-mini` (4B), `-flash` (9B), `-turbo` (27B) | GPU |
| Questions over images and video | The Decision 3.0 sizes, `vllm-sr/d3-edge` to `vllm-sr/d3` | GPU; no spans |
| Your own questions on a CPU | `vllm-sr/Decision-2.0-Kai-0.6B` | CPU or GPU; no spans |

Every built-in artifact is pinned to an exact Hugging Face revision, and a
deployment that nothing resolves to never starts.
[Supported models](../../model-runtime/supported-models/overview.md) has a page
per model with its size, memory, question types and a deployment snippet.

### Attach a runtime you run

A `model_runtime` deployment with an `endpoint` attaches to an engine the
Router does not manage, such as a GPU server shared by several Routers:

```yaml title="config.yaml"
global:
  model_catalog:
    deployments:
      gpu-decider:
        provider: model_runtime
        artifact: vllm-sr/vela3-flash
        endpoint: http://gpu-runtime.example.com:8100
        model: vela3-flash
    system:
      decision_model: {deployment: gpu-decider}
```

`model` is the ID the runtime lists in `GET /v1/models`. The Router reads the
runtime's model card when the runtime is up, and checks the rules against it
then. [Connect the Router to the model runtime](../../model-runtime/deploy/connect-the-router.md#attach-to-a-runtime-you-run)
covers the runtime side, and [Run on Kubernetes](../../model-runtime/deploy/kubernetes.md)
a shared runtime in a cluster.

## Remote System One APIs

A `systemone` deployment asks a System One API the Router does not run: a
hosted Decision API, another vLLM-SR engine, or a gateway in front of one. It
answers System One questions only; it is not one of the `providers.models`
requests are routed to.

```yaml title="config.yaml"
global:
  model_catalog:
    deployments:
      d3-remote:
        provider: systemone
        endpoint: https://systemone.example.com/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: SYSTEMONE_API_KEY}
        timeout_ms: 1500
    system:
      decision_model: {deployment: d3-remote}
    bindings:
      pii: primary
      hallucination: primary
```

- `endpoint` is the URL the API answers `POST` requests on, with any path
  prefix; `model` is the model it answers with.
- The API must take both [request forms](../use/api.md): the Router sends each
  stage's questions in one call, a batch with as many entries as the stage asks
  about (up to `judge.max_candidates` per judge), and never splits it.
- The question types, presets and input modalities come from the API's
  `GET <endpoint>/models` when it lists the model; otherwise from the
  deployment's `capabilities` (`question_types`, `presets`, `modalities`);
  otherwise they are `choice`, `noul` and `score` over text.
- A question without its own `timeout_ms` takes the deployment's, else
  1000 ms. A slow, failing or unreachable API leaves its questions unknown; it
  never fails the request by itself, except a delegated one the request carries
  no credential for.

:::warning
The d3 family answers no spans, so this example keeps `pii` and
`hallucination` on the local `primary` deployment. Without those two bindings,
a `pii` rule would fail to load with `span_required`.
:::

### Authentication

- The key comes from `auth.api_key_env` or `auth.api_key_file`, never from the
  configuration, and goes as `Authorization: Bearer <key>`. A gateway that
  takes the key in its own header sets `auth.header`, and `auth.prefix: ""` to
  send it without a prefix.
- An API that authorizes each user takes `auth: {credential: delegated}`
  instead of a key: each call carries the credential the authenticator issued
  the requesting user (`global.services.authz.delegated_credential`), in the
  same header and prefix. A request without one is answered `401` before the
  call, the Router's own listing read is refused (the `capabilities` apply),
  and no answer is kept for another request. See
  [Security Hardening](../../installation/security-hardening.md#delegate-the-users-credential-to-an-upstream).

## What the Router checks at load

`vllm-sr config validate --config config.yaml` runs the same checks. Nothing
falls back silently:

| Check | Error or warning |
| --- | --- |
| Every deployment a rule, binding or feature names is declared | The name, and the declared names |
| The deployment answers the rule's question type | The rule, the deployment and the types it answers |
| A span-needing rule or plugin resolves to a span-capable deployment, or to one whose `capabilities.presets` lists its `pii` or `halu` preset | `span_required` |
| A rule's `input.image`, `input.video` or `input.audio` is read by its deployment | `input_modality_unsupported` |
| A preset answered without a calibrated threshold | `uncalibrated_preset` (a warning; the rule uses 0.5) |
| A key written inline (`auth.api_key`, an `Authorization` header) | Keys come from `api_key_env` or `api_key_file` |

An attached runtime whose model is known only once it runs is checked when it
prepares the questions, with the same messages.

## What leaves the host

A question to a remote deployment sends the parts of the request it reads.
Choose bindings with that in mind:

| Asked by | Sends |
| --- | --- |
| `domains`, `fact_check`, `complexity`, `modality` | The current user turn |
| `user_feedbacks` | The current user turn and the assistant answer it replies to |
| `jailbreak`, `safety` | The current user turn, and the earlier user turns with `input.history`; at the response stage, the model's answer |
| `pii` | The current user turn, and the earlier user turns and tool results it is told to read |
| `turn_relation` | The current user turn and the last turns of the conversation |
| `hallucination` | The model's answer and its grounding context |
| `refusal`, `answer_check` | The request and the model's answer |
| `relevance`, `same_request` | The request and each candidate document, segment or cached request |
| `tool_need`, `tool_rank` | The request and the candidate tools' names and descriptions |
| `systemone` rules | The parts their `input` names, and the images with `input.image` |

:::note
A remote model has no calibrated thresholds, so its presets use 0.5 with the
`uncalibrated_preset` warning until a rule writes its own `predicate`. An
embedding deployment is separate from all of this: a remote embedding provider
receives the text it embeds, and only recall features use it.
:::
