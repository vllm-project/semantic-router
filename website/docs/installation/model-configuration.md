---
title: Configure Models
description: Choose a catalog-backed or custom model, bind it to providers, and use it from routing decisions.
---

# Configure models

A configured Model joins a stable Router name to one model identity and one or
more physical backends. Configure Models under `providers.models`, then refer to
their `name` from `providers.defaults.model` and
`recipes[].decisions[].modelRefs[].model`.

## Understand the model identifiers

The fields answer different questions and are not interchangeable:

| Field | Meaning |
| --- | --- |
| `name` | Router-local alias used by defaults, decisions, and direct model calls. |
| `catalog` | Optional canonical Model Card ID from the built-in catalog. Omit it for a private or newly released model. |
| `provider_model_id` | Optional model or deployment name sent to the upstream provider. |
| `backend_refs[].provider` | Stable Provider contract that supplies endpoint, authentication, path, protocol, and model-mapping defaults. |
| `api_format` | Optional wire-protocol override: `openai`, `responses`, or `anthropic`. It never selects a Provider. |

For example, `production-reasoner` can be your Router alias,
`openai/gpt-5.6-sol` its catalog identity, and a provider-specific deployment
name its `provider_model_id`.

## Choose a configuration path

| Starting point | Configure | Reasoning behavior |
| --- | --- | --- |
| Model exists in the catalog | Set `catalog` and a Provider on each backend. | Inherited from the catalog and selected Provider mapping. |
| Private or unlisted model with a known built-in family | Omit `catalog`; set `reasoning.family`. | Reuses the built-in family contract. |
| Private or unlisted model with unique controls | Omit `catalog`; define `reasoning` inline. | Uses the model-local contract you define. |
| Custom pass-through model | Omit both `catalog` and `reasoning`. | Router does not synthesize reasoning controls. |

See [Catalog-backed models](catalog-backed-models),
[Custom models](custom-models), and [Reasoning configuration](model-reasoning)
for each path. [Model configuration patterns](model-configuration-patterns)
shows the supported combinations and invalid mixtures.

## Start with one custom model

```yaml
version: v0.5
providers:
  defaults:
    model: local-chat
  models:
    - name: local-chat
      provider_model_id: served-model
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: host.docker.internal:8000
          protocol: http
recipes:
  - name: default
```

The Router creates a sparse local Model Card for `local-chat`. Add a matching
`providers.models` entry only when routing needs metadata such as context
window, capabilities, tags, or LoRAs. Add benchmark measurements independently
under top-level `evaluation.records[]`.

## Check task capabilities before dispatch

The selected model must satisfy both the provider protocol's capabilities and
its declared task capabilities. The same checks apply when the Router selects
a fallback from the matched decision. A fallback cannot reintroduce a model
that was excluded by the request's context-window check.

Model Card capability metadata accepts protocol names such as `image_input`
and `image_generation`, and these catalog aliases:

| Alias | Means |
| --- | --- |
| `vision`, `audio`, `video` | Image, audio and video **input**; they don't grant media generation |
| `structured_output` | `structured_json` |
| `tool_use` | `tools` |

Descriptive labels such as `long_context` or `coding` don't erase the
recognized capability declarations beside them.

A model without any recognized capability declaration retains protocol-only
compatibility. For an annotated model, a protocol that can encode a request
does not override missing task capabilities. When no eligible decision model
can serve the requested task, the Router returns `unsupported_capability`.

## Configure Models in the Dashboard

Open **Build → Routing → Models → Add Model**. You can then:

1. Choose a Provider, connect it, and select discovered or built-in model IDs.
2. Enter a model ID when the Provider cannot list models.
3. Use **Advanced settings** for a name prefix, a built-in reasoning family,
   routing metadata, pricing, and delivery settings.
4. Use **Manual setup** from the Provider chooser for complete control over
   `catalog`, inline reasoning fields, `provider_model_id`, `api_format`, and
   backend references.

The Dashboard edits the same configuration document as YAML. Use one authoring owner for
a deployment and review the exported YAML before serving it. See
[Configuration workflows](configuration-workflows).

## Routing capabilities

Author-declared model capabilities act as an eligibility filter before model
selection: candidates whose declared capabilities cannot express a request are
filtered out before the configured selection algorithm scores the survivors, so
a compatible model with a higher score can win over one that merely appears
first. Final dispatch only validates the chosen model: a late capability
mismatch returns an error rather than restarting the selection or replacing the
chosen candidate. Declare them on a model card:

```yaml
recipes:
  - name: default
providers:
  models:
    - name: image-backend
      api_format: images
      capabilities: [image_generation]
      modalities: {input: [text], output: [image]}
```

A model's `modalities` name the media it reads and writes; a model that
declares none reads and writes text. A decision's
[output contract](../tutorials/decision/output) checks them, with its
capabilities, when the configuration loads.

The protocol capability vocabulary is defined by the `llmprotocol` package,
and a declaration reaches capability-aware dispatch in one of these states:

1. **No recognized declaration** — the model is unannotated and stays eligible
   on wire expressibility alone.
2. **Task/modality declaration** — names like `image_input`, `image_output`,
   `image_generation`, `audio_input`, `audio_output`, `video_input`,
   `video_output`, `file_input`, and `file_output` steer the declared-task
   filter: an annotated model must declare every task bit the request requires,
   otherwise the dispatch is rejected with `unsupported_capability`.
3. **Transport/accounting declaration** — names like `tools`, `reasoning`,
   `streaming`, and `structured_json` are recognized, yet they carry no task
   bit. Such a model is annotated, so it does not qualify for a media task it
   never declared; text requests require no task bit and are unaffected.
4. **Catalog aliases** — `vision`, `audio`, and `video` project onto
   `image_input`, `audio_input`, and `video_input`; `structured_output` maps to
   `structured_json` and `tool_use` maps to `tools`. Projection runs before the
   filter, so a card declaring `[image_input, vision]` is filtered on
   `image_input` alone: it stays eligible for image requests and is rejected
   for audio, which it never declared.
5. **Descriptive labels** — names outside the protocol vocabulary and its
   aliases (e.g. `long_context`, `coding`) are metadata: they contribute no
   task bit and do not void recognized names in the same declaration, so a
   declaration of descriptive labels alone is treated as unannotated.

The pre-scoring filter and the final validation apply the same qualification,
so validation cannot admit a model the filter would have denied.

## Tune timeouts, retries, and endpoint health

The `reliability` block of a provider model controls how the Router talks to
that model's backends: load balancing, timeouts, retries, circuit breaking,
outlier ejection, and active health checks. Every field is optional, and a
field you leave out keeps the default shown below.

```yaml
providers:
  models:
    - name: production
      reliability:
        lb_policy: least_request
        connect_timeout: 3s
        total_timeout: 120s
        per_try_timeout: 30s
        retry_count: 2
        retry_on: 5xx,retriable-status-codes
        retriable_status_codes: [429]
        retry_after_max: 10s
        consecutive_5xx: 5
        health_check_path: /health
      backend_refs:
        - name: replica-a
          provider: vllm
          endpoint: 10.0.0.1:8000
        - name: replica-b
          provider: vllm
          endpoint: 10.0.0.2:8000
```

| Field | Default | Effect |
| --- | --- | --- |
| `lb_policy` | `round_robin` | `round_robin` follows the backend weights; `least_request` prefers the replica with the fewest requests in flight. |
| `connect_timeout` | `10s` | Time to open a connection, TLS handshake included. |
| `total_timeout` | the listener's `timeout` | Time for the whole call: every attempt and the streamed response. `0s` turns it off. |
| `idle_timeout` | the listener's `timeout` | Longest wait for more of a streamed response. `0s` turns it off. |
| `per_try_timeout` | none | Time for each attempt until its response starts. |
| `first_byte_timeout` | none | Time for the first response byte, so a stream that stalls before it starts can still be retried. Standalone mode only. |
| `retry_count` | `0` | Retries after the first attempt, up to 5. |
| `retry_on` | `connect-failure,refused-stream` | When to retry, with Envoy's condition names. Both gateways act on `5xx`, `gateway-error`, `reset`, `connect-failure`, `refused-stream`, `retriable-status-codes`, `retriable-4xx`, `reset-before-request`, and `envoy-ratelimited`. Envoy's other names (`retriable-headers`, `http3-post-connect-failure`, and the gRPC `cancelled`, `deadline-exceeded`, `internal`, `resource-exhausted`, `unavailable`) load, and the standalone gateway ignores them. A name Envoy does not know fails the configuration load. |
| `retriable_status_codes` | none | Statuses retried under `retriable-status-codes`, such as `429`. |
| `retry_back_off_base`, `retry_back_off_max` | `25ms`, ten times the base | Bounds of the randomized exponential wait between retries. |
| `retry_after_max` | none | Wait the `Retry-After` seconds a response asks for, up to this bound. |
| `retry_budget_percent`, `retry_budget_min_concurrency` | three retries at once | Limit concurrent retries to a share of the requests in flight instead (20% and 3 once either is set). |
| `consecutive_5xx` | none | Failures in a row that take a replica out of rotation. Needs two or more replicas. |
| `base_ejection_time`, `max_ejection_percent` | `30s`, `50` | How long an ejected replica stays out (longer each time in a row), and the share that can be out at once. |
| `health_check_path`, `health_check_interval`, `health_check_timeout` | none, `10s`, `2s` | Active `GET` checks; only a `200` passes. |

Retries and failures behave the same way whichever gateway serves the
traffic:

- A connection failure, a reset, or a per-try timeout counts as no response
  at all, so `5xx`, `gateway-error`, and `reset` all retry it.
- A retry prefers a replica the request has not tried yet.
- Nothing is retried once any part of the response has reached the client.
- A replica that fails a health check, or is ejected, leaves the rotation. If
  fewer than half of the replicas remain healthy, the Router spreads requests
  over all of them rather than overloading the rest.
- When no attempt gets a response, the client receives `503` (or `504` for a
  timeout) as Envoy's plain-text local reply, such as `upstream connect error
  or disconnect/reset before headers. reset reason: remote connection
  failure`. The response phases do not run on it, so it is not in the error
  format of the client's API.

The Envoy data plane has no first-byte timeout, so a configuration that sets
`first_byte_timeout` fails when the Router renders its Envoy configuration.

Active health checks run in the gateway that carries the client traffic. In
standalone mode the Router probes each replica. Behind Envoy, Envoy probes
them, and the model calls the Router makes itself, such as an executor's,
leave the rotation to outlier ejection (`consecutive_5xx`).

### Override timeouts and retries for one decision

A decision can carry its own `reliability` block for the requests it routes,
for example a long-context route that needs more time than its provider
model's default:

```yaml
recipes:
  - name: default
    decisions:
      - name: long_context_route
        priority: 100
        rules:
          operator: AND
          conditions:
            - type: context
              name: long_context
        modelRefs:
          - model: production
        reliability:
          total_timeout: 600s
          per_try_timeout: 300s
          retry_count: 1
          retry_on: reset
```

The block takes the provider block's timeout and retry fields, and merges
with the provider model's settings the same way in every gateway:

- `total_timeout`, `per_try_timeout`, `idle_timeout`, and `first_byte_timeout`
  replace the provider model's; `0s` turns one off.
- `retry_count` replaces the provider model's. `retry_on` and
  `retriable_status_codes` add to the provider model's conditions. Retries on
  a provider model that has none use `connect-failure,refused-stream` unless
  the decision names its own conditions.
- `retry_back_off_base`, `retry_back_off_max`, and `retry_after_max` replace
  the provider model's.
- Connection timeouts, load balancing, retry budgets, ejection, and health
  checks stay with the provider model, because they belong to its backends.

Behind Envoy, the Router passes the override to Envoy's router as per-request
headers: `x-envoy-upstream-rq-timeout-ms`,
`x-envoy-upstream-rq-per-try-timeout-ms`, `x-envoy-max-retries`,
`x-envoy-retry-on`, and `x-envoy-retriable-status-codes`. Every Envoy
configuration the project ships lets the Router set exactly these headers. If
you run the Router behind your own Envoy, allow them in the ext_proc filter's
`mutation_rules`:

```yaml
mutation_rules:
  allow_expression:
    regex: "^x-envoy-(upstream-rq-timeout-ms|upstream-rq-per-try-timeout-ms|max-retries|retry-on|retriable-status-codes)$"
```

The reliability block is the only way to set these headers: the Router drops
them from a decision's `header_mutation` plugin when it loads the
configuration, and logs a warning.

Envoy cannot apply an idle timeout, a first-byte timeout, a back-off, or a
`Retry-After` bound to a single request. A decision that sets `idle_timeout`,
`first_byte_timeout`, `retry_back_off_base`, `retry_back_off_max`, or
`retry_after_max` fails validation unless the Router serves with
`--gateway standalone`.

## Check prompts against the context window

The Router checks a request against the models' `context_window_size` twice:
from the moment it arrives, against every model the request can reach, and
after the decision, against the decision's candidates. It counts the prompt
itself, beside the signals, with each model's own tokenizer, and asks no
backend to count:

```yaml
providers:
  models:
    - name: deepseek-v4.1-flash
      context_window_size: 1048576
      tokenizer: deepseek-ai/DeepSeek-V4.1-Flash   # optional @revision
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: deepseek.example:8000
          protocol: http
global:
  router:
    context_window:
      margin: 0.03
      proxy_margin: 0.30
      proxy_tokenizer: o200k_base
```

| Field | Default | Effect |
| --- | --- | --- |
| `providers.models[].tokenizer` | the model's Hugging Face repo, else none: the proxy counts | The model's own tokenizer: a Hugging Face repo id, `org/name` or `org/name@revision`; a built-in encoding (`o200k_base`, `cl100k_base`); or a local `tokenizer.json` or the directory that holds one (a path that is absolute or starts with `./` or `../`). |
| `global.router.context_window.margin` | `0.03` | The band around a window, as a fraction of it, for a count with the model's own tokenizer. |
| `global.router.context_window.proxy_margin` | `0.30` | The band for a count with the proxy tokenizer. |
| `global.router.context_window.proxy_tokenizer` | `o200k_base` | Counts the prompts of models without a tokenizer, or whose tokenizer has not loaded. |

- **Tokenizers.** Most models need no `tokenizer`. A model whose catalog card
  publishes open weights on Hugging Face (`distribution.source`), or that a
  self-hosted engine such as vLLM or SGLang serves under its repo id
  (`provider_model_id: Qwen/Qwen3-8B`), counts with that repo's tokenizer.
  Name one only for another model, or to pin a revision. The Router
  downloads only the repo's `tokenizer.json`, with the Hugging Face CLI and
  settings of its other model downloads (`HF_ENDPOINT`, `HF_TOKEN` for a
  gated repo, `HF_HOME`, the revision), into the same models directory:
  `models/tokenizers/<org>/<name>`, or `models/tokenizers/<org>/<name>@<revision>`.
  It downloads in the background when it starts, and neither startup nor a
  request waits; with `HF_HUB_OFFLINE=1` it downloads nothing. The Router
  carries `o200k_base` and `cl100k_base`, and reads byte-level BPE
  `tokenizer.json` files, the kind DeepSeek, Qwen, GLM, MiniMax and gpt-oss
  publish.
- **Until a tokenizer loads.** While a model's tokenizer downloads, or when
  it cannot load (no Hugging Face access, a gated repo without `HF_TOKEN`, a
  SentencePiece tokenizer), the proxy tokenizer counts for it, with its wider
  band. The log says `tokenizer_pending`, `tokenizer_download_failed` or
  `tokenizer_unavailable`, and `llm_context_window_tokenizer_loaded` is `0`
  for that tokenizer; it is `1` once the tokenizer has loaded, which the log
  says as `tokenizer_loaded`.
- **The byte bound.** A byte-level tokenizer never produces more tokens than
  a prompt has UTF-8 bytes. A request whose bytes, plus a reserve for the
  chat template and message framing, fit the widest window it can reach is
  never counted.
- **The count.** A longer request is counted from the moment it arrives,
  beside the signals, the guard included: neither waits for the other. It is
  counted once per tokenizer, in parallel parts, and the count serves the
  check after the decision too. It covers the text, tool calls and results,
  tool definitions and the output schema, plus a few tokens per message for
  the chat template. A 4 MB prompt takes about 50 ms on one core. Requests
  that carry images, audio, video or files are not counted.
- **The bands.** A count with the model's own tokenizer at or below the
  window less `margin` fits, and at or above the window plus `margin` does
  not. A proxy count can be well off the model's own count: `o200k_base`
  counts Chinese text up to a quarter higher than some models' tokenizers,
  and English and code a little lower. So a proxy count uses `proxy_margin`,
  and refuses a prompt only at or above the window plus 30%. When the counts
  rule out every model the request can reach, the Router cancels the signals
  still running, so that the guard stops reading the request, and answers
  `400` with `context_length_exceeded` at once. The message names the widest
  window and the estimated count; see
  [Routing errors](../api/router#routing-errors). A count in between goes on
  to the backend, which decides; its refusal reaches the client as
  `context_length_exceeded`, and a fallback goes on only to candidates with
  wider windows. When a decision of the recipe compresses context, the
  refusal waits for the check after the decision. When one replaces the
  client's system prompt or drops its tools (a tool policy of `none`), the
  request is refused only if it is still beyond the band without them.
- **What decided.** Each model's check is logged once per request as
  `context_window_checked`, with the kind of count (`exact` or `proxy`), its
  band and the verdict (`fits`, `in_band` or `exceeds`), and counted in
  `llm_context_window_checks_total`; refusals are counted in
  `llm_context_window_refusals_total`, and `llm_context_window_margin` shows
  both bands. `context_window_counted` logs how long the count took and how
  long the request waited for it after its signals, which
  `llm_context_window_wait_seconds` records: about zero.
- **Without a count.** A request with media is never refused for its length.
  Its token estimate still steers it away from a candidate whose window looks
  too small, and when it rules out every candidate, the widest ones stay.

The token estimate learns a bytes-per-token ratio from the prompt usage
backends report. That ratio places a request in a recipe's
[context bands](../tutorials/signal/heuristic/context), and nothing else: it
never refuses a request.

### Tokenizers without Hugging Face access

A Router without Hugging Face access reads tokenizers from its models
directory, as it does its other models. Fill the directory where there is
access, then ship it:

1. Download each tokenizer to where the Router looks for it, or run the
   Router once with `--download-only`, which fetches the tokenizers along
   with the other models:

   ```bash
   hf download deepseek-ai/DeepSeek-V4.1-Flash --include tokenizer.json \
     --local-dir models/tokenizers/deepseek-ai/DeepSeek-V4.1-Flash
   hf download Qwen/Qwen3-8B --include tokenizer.json --revision <commit> \
     --local-dir models/tokenizers/Qwen/Qwen3-8B@<commit>
   ```

2. Ship `models/` as the Router's model volume (`/app/models` in the image),
   and set `HF_HUB_OFFLINE=1`.
3. Keep the configuration as it is: the repo ids, named or inferred, find the
   files. A tokenizer kept elsewhere can be named by its path instead, as
   `tokenizer: /opt/tokenizers/glm/tokenizer.json`.
4. Check the Router's log when it starts: `tokenizer_loaded` names each
   tokenizer and its vocabulary size, and `tokenizer_pending` names one whose
   file is missing, whose models then count with the proxy and its wider
   band.

## Fall back to another model

When a model fails, the Router can send the request to the decision's next
ranked candidate instead. This cross-model fallback is off by default. Turn it
on globally, for a recipe, or for one decision:

```yaml
global:
  router:
    fallback:
      enabled: true
      max_attempts: 3
      total_timeout: 30s
      per_attempt_timeout: 10s
      retryable_status_codes: [502, 503, 504]
      circuit_breaker:
        consecutive_failures: 3
        cooldown_period: 30s
        half_open_probes: 1
recipes:
  - name: default
    decisions:
      - name: escalate
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: small
          - model: large
        fallback:
          max_attempts: 2
          retryable_status_codes: [429, 503]
```

| Field | Default | Effect |
| --- | --- | --- |
| `enabled` | `false` | Whether a failed attempt moves on to the next candidate. |
| `max_attempts` | `3` | Attempts across candidates, the first one included. |
| `total_timeout` | `30s` | Time from the start of the request in which another candidate is tried. A candidate gets at most what is left, and none starts after it runs out. |
| `per_attempt_timeout` | `10s` | Time for each candidate after the first. It cannot exceed `total_timeout`. |
| `retryable_status_codes` | `[502, 503, 504]` | Statuses that move on to the next candidate. |
| `circuit_breaker` | 3 failures, `30s`, 1 probe | Skips a backend that keeps failing, for a cool-down. Global and recipe policies only. |
| `forward_request_headers` | none | Client request headers that a fallback attempt the Router sends itself copies; see [Carry client headers on a fallback attempt](#carry-client-headers-on-a-fallback-attempt). Global and recipe policies only. |

The policy a request runs under is merged field by field, the most specific
layer first:

1. a request-graph step's `fallback`, for the model calls a request graph
   makes;
2. the decision's `fallback`;
3. the recipe's `recipes[].fallback`;
4. `global.router.fallback`.

- **Layers.** A field a layer leaves out keeps the layer below, and `enabled`
  is only set where you write it.
- **Decisions.** A decision's block takes every field except
  `circuit_breaker`, which belongs to the backends.
- **Candidates** are always the decision's ranked models that remain eligible
  for the request. There's no separate fallback list, and a provider model has
  no fallback of its own: other replicas of the same model are tried through
  its retries, ejection and health checks instead.
- **Reasoning.** A candidate is sent with its own `modelRefs` reasoning
  settings (`use_reasoning`, `reasoning_effort`, `reasoning_mode`), written for
  its backend the way the first attempt's are. A candidate the Router cannot
  prepare, such as one its rate limit refuses, is passed over for the next one
  and logged as `fallback_candidate_skipped` with the reason.
- **Streaming.** A streamed request falls back too, whatever `stream_options`
  it carries. Behind Envoy the Router calls the candidate without streaming and
  streams the answer to the client with the client's own `stream_options`, so a
  client that sets `include_usage` still gets the usage chunk last. In
  standalone mode the candidate streams, and a Chat Completions backend is
  asked for its usage as the first attempt's is.
- **The answer.** A chain that ends without a success answers with its last
  attempt. A candidate's error keeps its status and reaches the client in the
  client's protocol, so a candidate's `400` stays a `400`, and a refusal of a
  prompt over its window becomes `context_length_exceeded`. When no backend
  answered the last attempt, the client gets `503` with
  `upstream_unavailable`, or `504` with `upstream_timeout`, and none of the
  transport's detail.
- **Long prompts.** A candidate whose window the Router's
  [count](#check-prompts-against-the-context-window) exceeds beyond its band
  is never tried. After a backend refuses the prompt for its length, the chain
  goes on only to candidates with wider windows. Both are logged as
  `fallback_candidate_skipped`. Behind Envoy the first model's error body is
  not seen, so its `400` or `413` counts as a refusal for length when the
  count was in the band of that model's window.

Both gateway modes run the same policy, with one deliberate difference:
standalone mode also falls back when no backend answered at all, such as a
refused connection or a timeout, while behind Envoy the client receives
Envoy's reply. Such a failure counts as a `503` or `504`, so keep those
statuses in `retryable_status_codes` to fall back on it. Nothing falls back
once any part of the response has reached the client.

Neither fallback timeout shortens the first attempt, which keeps its provider
model's and decision's `reliability` timeouts. To fall back from a model that
never answers, give that provider model a `reliability.per_try_timeout` in
standalone mode:

- The timeout counts as a `504`, and the next candidate is tried.
- The candidate gets at most `per_attempt_timeout` and what is left of
  `total_timeout`, so make `total_timeout` longer than the `per_try_timeout`.
- A decision's own `per_try_timeout` bounds every candidate it tries, so a
  slower candidate fails the same way.

### Carry client headers on a fallback attempt

Behind Envoy, the first attempt goes through Envoy with the client's
headers. A fallback attempt is a call the Router makes itself, and it carries
only the backend's `headers` and its credential. To pass a client header
through as well, such as the `x-user-id` an upstream gateway attributes usage
by, list it in `forward_request_headers`:

```yaml
global:
  router:
    fallback:
      enabled: true
      forward_request_headers: [x-user-id]
```

- **Matching.** Names match in any letter case, and an attempt copies a
  header only when the client sent it with a value.
- **Precedence.** The client's value replaces a backend's `headers` value of
  the same name, which stays the default for a client that sends none. The
  credential header is always the Router's own.
- **Gateway modes.** In standalone mode every attempt, fallback attempts
  included, starts from the client's request and already carries its
  headers, so the list changes nothing there. An attempt that starts from the
  client's request, which is also the first attempt behind Envoy, sends a
  backend's `headers` value in place of the client's.
- **Layers.** Only global and recipe policies take the list, as with
  `circuit_breaker`. A recipe's list replaces the global one.
- **Logs.** Each attempt in the fallback record names the headers it copied
  (`forwarded_headers`), never their values.

The configuration fails to load, naming the header, when a list names one
that a call the Router builds must not copy from the client:

- a pseudo-header, such as `:authority`;
- a hop-by-hop or framing header: `connection`, `keep-alive`, `te`,
  `trailer`, `transfer-encoding`, `upgrade`, any `proxy-*` header, `host`,
  `content-length`, and the body headers the Router sets for its own request
  (`content-type`, `content-encoding`, `accept`, `accept-encoding`, `expect`);
- a credential: `authorization`, `proxy-authorization`, `cookie`,
  `x-api-key`, `api-key`, `ocp-apim-subscription-key`, the per-user key
  headers (`x-user-*-key`), and any backend's auth header;
- the Router's own `x-vsr-*` headers, Envoy's `x-envoy-*` controls, and the
  identity headers (`x-authz-*` and the names `global.services.authz.identity`
  sets), which the Router keeps from every upstream.

A remote deployment takes the same setting for the calls the Router makes to
it; see
[Forward client headers](model-catalog.md#forward-client-headers).

On Kubernetes, a decision takes the same `reliability` and `fallback` blocks,
with the same field names: in an `IntelligentRoute`'s `spec.decisions[]` and in
a `SemanticRouter`'s `spec.config.decisions[]`. The CRDs check the durations,
retry count and status codes when you apply the resource.

## Validate the result

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

Validation resolves catalog entries, Provider mappings, reasoning controls,
Model Card identities, and backend-pool compatibility before startup.

## Configure models used by Router tasks

For the decision and embedding models the Router uses itself, start with the
[model runtime](model-runtime/get-started/overview.md). It covers local and remote models,
their configuration, and operations.
