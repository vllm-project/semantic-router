---
title: Model catalog reference
sidebar_label: Model catalog reference
description: Every field of global.model_catalog.deployments, the providers and what each answers, bindings, embeddings, resolution order, thresholds and admission.
---

# Model catalog reference

`global.model_catalog` declares every model the Router asks: the models that
answer routing signals and embedding models. Request routing
targets, the models that answer users, stay in
[`providers.models`](model-configuration.md).
[Choose a model for a signal](choose-a-model-for-a-signal.md) explains which
deployment answers each signal.

## Deployments

A deployment is a model the Router can ask, under a name. `provider` says which
API the Router speaks to it.

| `provider` | API the Router calls | Can answer |
| --- | --- | --- |
| `model_runtime` | The built-in model runtime: managed (the Router starts it) or attached (`endpoint` set) | Anything its model card serves |
| `systemone` | `POST <endpoint>` and `GET <endpoint>/models` of a System One API | Every question-style signal and selector |
| `openai_chat` | `POST <endpoint>/chat/completions` | The memory query rewrite (`global.stores.memory.query_rewrite`) |
| `openai_embeddings` | `POST <endpoint>/embeddings` | Embeddings |

The Router refuses at load a deployment given a signal it cannot answer.

Two deployments are always declared and start only when something resolves
to them:

| Name | Model | Used by default for |
| --- | --- | --- |
| `primary` | `vllm-sr/vela3-fast` on CPU | `system.decision_model` |
| `vela-embedding` | Vela 1.0 Embedding | `embeddings.default` |

Embeddings serve recall only: `vela-embedding` starts when the response cache,
memory or a vector store recalls with it, never for a decision (see
[Embeddings](../model-runtime/guides/embeddings.md)).

## Field reference

```yaml title="field-reference"
global:
  model_catalog:
    deployments:
      # Managed model_runtime: no endpoint; the Router starts and supervises it.
      managed-example:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Hazard  # Hub repository or absolute package path
        revision: 5dd25f2cc3c98f338e6a79b667662d60f936a28d  # 40-character commit
        device: cpu                      # cpu | auto | cuda | rocm, or cuda:N / rocm:N
        input: {max_tokens: 32768, overflow: reject}  # reject | truncate | window
        public_name: hazard-public       # its name on the inference API
      replicated-example:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        revision: cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764
        replicas: [{device: cpu}, {endpoint: http://worker-2.example.com:8100, model: kai}]
      # Connection block: an attached model_runtime and every remote provider.
      remote-example:
        provider: systemone              # or openai_chat, openai_embeddings
        endpoint: https://api.example.com/v1/systemone  # full URL; unix:///abs.sock for an attached runtime
        model: vllm-sr/d3-ultra          # the remote model; an attached runtime's served model
        auth: {header: Authorization, prefix: Bearer, api_key_env: EXAMPLE_KEY}  # or api_key_file: /abs/path
        headers: {X-Org: acme}           # non-secret headers only
        timeout_ms: 1500                 # per attempt
        max_retries: 1                   # transport retries within the caller's deadline
        tls: {ca_file: /etc/ssl/ca.pem, server_name: api.example.com, insecure_skip_verify: false}
        limits: {max_request_bytes: 262144, max_response_bytes: 4194304}
        capabilities: {question_types: [choice, noul, score], presets: [pii, halu]}
    system:
      decision_model: {deployment: primary}
    bindings: {pii: primary}
    embeddings:
      default: {deployment: vela-embedding}  # recall embeds with it
```

### Managed `model_runtime` fields

| Field | Meaning |
| --- | --- |
| `artifact` | A Hugging Face repository ID, or an absolute path to a model package |
| `revision` | The 40-character commit of a repository artifact |
| `device` | `cpu`, `auto` (the best device the host has), `cuda`, `rocm`, or one device such as `cuda:0` |
| `input` | The deployment's input budget: `max_tokens`, and `overflow` (`reject`, `truncate` or `window`) |
| `replicas` | Workers of one logical deployment: each a `device`, or an `endpoint` and `model` of a worker started elsewhere |
| `public_name` | The deployment's name on the inference API |
| `max_num_seqs`, `max_num_batched_tokens`, `batch_window_ms`, `dtype`, `enable_shared_context`, `max_model_len` | Engine arguments; see [engine arguments](../model-runtime/engine-args.md) |

### Connection block

The same fields connect an attached model runtime (a `model_runtime`
deployment with `endpoint`) and every remote provider.

| Field | Meaning | Limits |
| --- | --- | --- |
| `endpoint` | The full URL of the API, path prefix included. The Router appends the operation path (`/chat/completions`, `/embeddings`); a systemone endpoint is the URL questions are posted to. An attached runtime may use `unix:///abs/path.sock` | `http` or `https`; no credentials or fragment |
| `model` | The model the remote API answers with; on an attached runtime, the served model | Required for `systemone`, `openai_chat`, `openai_embeddings` |
| `auth` | `header` (default `Authorization`), `prefix` (default `Bearer`; `""` sends the bare key), and exactly one of `api_key_env` or `api_key_file` (an absolute path, read on each call so a rotated key applies), or `credential: delegated`: each call made for a client request carries the credential the authenticator issued that user ([Security Hardening](security-hardening#delegate-the-users-credential-to-an-upstream)) | A key is never written in the configuration; `credential: delegated` takes no key, and not on `model_runtime` |
| `headers` | Further non-secret headers | `Authorization`, `Proxy-Authorization`, `X-Api-Key`, `Api-Key`, `Ocp-Apim-Subscription-Key`, `Cookie` and the `auth` header are refused |
| `timeout_ms` | Bound of one attempt | systemone ≤ 60000; others ≤ 300000 |
| `max_retries` | Transport retries after the first attempt, within the caller's deadline | 0–10; not on an attached runtime |
| `tls` | `ca_file`, `server_name`, `cert_file` and `key_file` (a client certificate), `insecure_skip_verify` | `https` only; files are absolute paths |
| `limits` | `max_request_bytes`, `max_response_bytes` | Not on an attached runtime |
| `capabilities` | The remote model's facts, when its API lists nothing: `question_types`, `presets` and `modalities` (systemone); `dimensions` and `modalities` (openai_embeddings: `text`, `image`, `audio`) | An embedding deployment recall uses embeds `text` |
| `generation` | `max_tokens`, `temperature` and `reasoning` (`{family, use_reasoning, reasoning_effort}`, where `family` names a built-in reasoning family) of an `openai_chat` deployment's completions | `max_tokens` ≤ 32768; `temperature` 0–2; `reasoning` needs `family` and `use_reasoning` |

Redirects are refused, so a credential never follows a redirect to another
host.

## Bindings and rule deployments

```yaml
global:
  model_catalog:
    deployments:
      local-vela: {provider: model_runtime, artifact: vllm-sr/vela3-fast, device: cpu}
    bindings: # every recipe
      pii: local-vela
recipes:
  - name: default
    model_bindings: # this routing profile; recipes[].model_bindings for one recipe
      domain: d3-remote
```

Binding keys are the application signal types: `domain`, `fact_check`,
`user_feedback`, `jailbreak`, `safety`, `pii`, `hallucination`, `refusal`,
`turn_relation`, `complexity` and `modality`. A binding answers the rules of
its type that name no deployment; a rule writes `deployment:` to ask another.
`recipes[].signals.systemone[]` rules and `algorithm.systemone` of a decision
write `deployment:` too. The `jailbreak`, `pii` and `hallucination` bindings
also choose the model behind `response_guard` and `masking`, which read those
signals' rules.

A binding names a deployment and nothing else (`pii: primary`).

Resolution, for every signal: the rule's `deployment`, then the recipe's
binding, then the global binding, then `system.decision_model` for a
question-style signal. Embeddings answer no signal: recall embeds with
`embeddings.default`, or with the deployment a store's or a RAG source's
`embedding` block names.

## Thresholds

A rule's predicate you do not write comes from the model that answers: the
threshold calibrated for `vela3-fast` or a Vela 2.0 size; 0.5 for any other
model, local or `systemone`, which has none (the load warns
`uncalibrated_preset`, naming the rule). A `predicate` you write is used as
written.

## Admission

`global.model_catalog.admission` bounds concurrent calls. A key is a declared
deployment's name, and its gate is shared by every user of that deployment.

```yaml
global:
  model_catalog:
    admission:
      local-vela: {max_concurrency: 4, max_queue: 64, queue_timeout_ms: 250, on_overflow: shed}
      d3-remote: {max_concurrency: 16}
```

## Examples

### R1. Remote embedding model

```yaml title="r1-remote-embedding"
global:
  model_catalog:
    deployments:
      remote-embed:
        provider: openai_embeddings
        endpoint: https://embedding.example.com/v1
        model: BAAI/bge-m3
        auth: {api_key_env: EMBEDDING_API_KEY}
        timeout_ms: 5000
        max_retries: 2
        capabilities: {dimensions: 1024, modalities: [text]}
    embeddings:
      default: {deployment: remote-embed}
```

The Router posts to `https://embedding.example.com/v1/embeddings`: the
operation path is appended to the endpoint's path, never assembled from a host
and port.

### R3. Remote decision model for jailbreak and hallucination

```yaml title="r3-remote-judge"
global:
  model_catalog:
    deployments:
      d3-judge:
        provider: systemone
        endpoint: https://judge.example.com/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: JUDGE_KEY}
        timeout_ms: 5000
recipes:
  - name: default
    model_bindings:
      jailbreak: d3-judge # request- and response-stage rules
      hallucination: d3-judge
```

Every `jailbreak` and `hallucination` rule of this scope asks `d3-judge`; the
other signals stay on `primary`. A binding names only the deployment: the
signal decides the question, the provider how the Router asks it. A
hallucination rule asks a span question, so its deployment must answer spans;
a `systemone` deployment that lists `capabilities.question_types` must include
`span`.

### R4. System One behind a gateway with its own auth header

```yaml title="r4-systemone-gateway"
global:
  model_catalog:
    deployments:
      d3-gateway:
        provider: systemone
        endpoint: https://gateway.example.com/Unified/v1/systemone
        model: vllm-sr/d3-ultra
        auth:
          header: Ocp-Apim-Subscription-Key
          prefix: ""                     # no Bearer prefix
          api_key_file: /run/secrets/systemone-key
        timeout_ms: 3000
        capabilities: {question_types: [choice, noul, score], presets: [pii, halu]}
    system:
      decision_model: {deployment: d3-gateway}
```

### R5. Attached model runtime

With `endpoint`, the Router connects to a model runtime that is already
running; without it, the Router starts and supervises the runtime.

```yaml title="r5-attached-runtime"
global:
  model_catalog:
    deployments:
      decision-kai:
        provider: model_runtime
        endpoint: https://decision-worker.example.com:8100
        model: decision-kai                      # the served model
        auth: {api_key_env: WORKER_KEY}          # optional
        tls: {ca_file: /etc/ssl/internal-ca.pem}
        timeout_ms: 1000
    system:
      decision_model: {deployment: decision-kai}
```

### R6. Request backends use the same connection shape

```yaml title="r6-request-backends"
providers:
  defaults: {model: claude}
  models:
    - name: claude
      backend_refs:
        - provider: anthropic
          endpoint: https://gateway.example.com/anthropic
          auth: {header: x-api-key, prefix: "", api_key_env: ANTHROPIC_KEY}
          headers: {X-Org: acme}
    - name: native-d3              # a native System One alias: no second endpoint or key
      api_format: systemone
      deployment: d3-remote
global:
  model_catalog:
    deployments:
      d3-remote:
        provider: systemone
        endpoint: https://app.vllm-sr.ai/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: VSR_APP_KEY}
```

`providers.models` stays a separate concept: the models that answer users,
which routing chooses among. Only the connection shape (`endpoint`, `auth`,
`headers`) is shared. A backend reads its key from `auth.api_key_env`, or
takes the requesting user's with `auth.credential: delegated`. A
deployment, the decision model included, is never a routing target and never
appears in `/v1/models`. The native System One API answers for one only under
its `public_name`, or through a native alias such as `native-d3` that names it.
