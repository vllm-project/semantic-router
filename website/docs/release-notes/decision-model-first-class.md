---
title: "Release note: v0.5, the decision model is first-class"
description: v0.5 makes the decision model the one model that answers every routing judgment, gives the configuration one structure at schema v0.5, and reduces the response headers to a public set.
---

# Release note: v0.5, the decision model is first-class

The Router already answered most routing questions with one model family,
the System One decision models. Its configuration still described an older
system, in which each signal had its own small specialist model or an
embedding-similarity trick. v0.5 makes the decision model first-class:

- **Signals have three layers:** the `systemone` primitive for questions you
  write; eleven application signals, each compiled into one `systemone`
  question; and seven deterministic heuristics.
- **Zero-shot everywhere.** Every System One model answers every question it
  supports zero-shot; trained adapters and calibrated thresholds are
  per-family optimizations. Models differ in the question types they answer:
  Decision 1.0, 2.0 and 3.0 return no spans, and vela3-fast answers no score
  question. The configuration fails to load when a question resolves to a
  deployment that cannot answer its type: a rule with `span_required` or
  `question_type_unsupported`, and a judge, the `systemone` selector or an
  escalate check with a message naming the deployment and the types it
  answers.
- **Similarity has one `match` block**, with an optional embedding or BM25
  `recall` stage and an optional System One `judge` stage. Every judge has a
  hard cap on its candidates, and overflow is truncated, never an error.
- **Embeddings are for recall only, and start lazily.** No decision path uses
  embeddings.
- **One way to call System One:** a standard request, or a batch of named
  standard requests.
- **Algorithms are selectors and executors**, and **plugins only act**, naming
  the signals they act on.
- **The configuration has one structure**, at schema `v0.5`.
- **The response headers are a public set**, with one gated `x-vsr-trace`.

The new [System One](../system-one/overview) chapter explains the model
behind all of it.

## Breaking changes

This is one breaking release. Run `vllm-sr config migrate` once, then review
the notes it prints; the [migration guide](../installation/migrate-to-v0-5)
lists every removed form, its load error and its rewrite.

- **Config schema v0.5.** `version: v0.3` fails at load with
  `invalid_value`, and migrate writes `version: v0.5`. There is no top-level
  `routing`: routing lives in flat `recipes`, model facts on
  `providers.models[]`, and default plugins in `global.plugins`.
- **Removed forms fail at load with one line**, and migrate rewrites them.
  The only alias is `decision`, which loads as `systemone` with a warning.
- **Removed signal types:** `classifiers`, `preferences`, `embeddings` and `kb`
  (`systemone` questions; migrate drops a `kb` rule, which never matches, and
  names its question), `reasks` and `topic_continuity` (`turn_relation`),
  `events` and `structure` (`keywords`), and `actions` (a `systemone`
  question). `global.model_catalog.modules.*` is removed; a module's threshold
  is a rule's `predicate`.
- **Removed algorithms:** `prompt`, `router_dc`, `knn`, `kmeans`, `svm` and
  `mlp` become `systemone`; `automix`, `hybrid` and `latency_aware` become
  `multi_factor`; the Looper `confidence` algorithm and the native `cascade`
  become `escalate`. The ML selectors' training pipeline and the Dashboard ML
  pipeline are removed.
- **Removed plugins:** `tool_selection` (the `tools` plugin's `select`),
  `rag`, `memory` and `context_compression` (the `context` plugin),
  `hallucination` and `response_jailbreak` (`response_guard`),
  `shadow_dispatch` (`router_replay.shadow`), and the cache aliases
  (`response_cache`).
- **Prompt compression is the `context` plugin's `compress.signals`.**
  `global.model_catalog.modules` is removed: `prompt_compression` compresses
  the text the routing signals read from `global.plugins` or a recipe's
  plugins, never a decision's, and a `compress` block with only `signals`
  leaves the request whole. Migrate moves it.
- **The Router's embeddings API and `embeddings.multimodal` are removed:**
  `global.services.api.embeddings`, `POST /api/v1/diagnostics/embeddings`,
  `/api/v1/diagnostics/similarity` and `/api/v1/diagnostics/similarity/batch`,
  and `global.model_catalog.embeddings.multimodal`; migrate drops both. A
  decision about an image asks a `systemone` question that reads it
  (`input.image`). The model runtime's worker `/v1/embeddings` stays.
- **`global.model_catalog.embeddings.default` names a deployment**,
  `embeddings.default: {deployment: vela-embedding}`, the way
  `system.decision_model` does. A bare name fails with `invalid_value`, and
  migrate rewrites it.
- **The previous release's model-resource forms no longer load:**
  `external[]` models, binding objects (`contract`, `adapter`, `head`,
  `mapping_path`, `operating_point`), bindings for one rule,
  `question_types` and `served_name` on deployments, `classify` and `http`
  deployments, `candle`, `ort` and `openvino` providers, and a backend's
  `base_url`, `api_key_env`, `auth_header`, `auth_prefix` and
  `extra_headers`. A binding names its deployment, and a backend connects
  with `endpoint`, `auth` and `headers`; migrate rewrites them. A backend's
  inline `api_key` still loads, but not together with `auth.api_key_env`.
- **`vllm-sr config migrate` needs a Router** (a binary or an image), and
  `--offline` is removed. It takes a configuration in the layout before
  config v0.3 to v0.5 in one run.
- **Looper names and endpoint:** `global.integrations.looper.*.model_names`
  is refused at load; migrate writes the names as an entrypoint of a recipe it
  writes, the default recipe with only its decisions of that algorithm.
  `global.integrations.looper.endpoint` is refused, and migrate drops it: the
  Router makes a Looper's model calls in process. The Dashboard drops the
  External Models card and its Looper card edits only the timeout and
  headers; the operator refuses `external_models`.
- **Model output facts are `capabilities` and `modalities`.** The model
  `modality` keyword (`ar`, `diffusion`, `omni`), image routes on
  `DIFFUSION` or `BOTH`, the `image_gen` plugin and audio embedding rules are
  removed; migrate rewrites what it can, drops a route no decision executes,
  and names the rest.
- **The System One batch form is `states` only.** A batch body carries
  named standard requests under `states` and no top-level `state`,
  `questions`, `images` or `videos`; a body that mixes the two forms, or an
  empty `states`, is refused with `invalid_request`. A batch's response has
  no top-level `answers`, and its `usage` counts every entry. See
  [the API](../system-one/api).
- **Model runtime contract 4.0.0.** A Router of this release refuses a model
  runtime of any other contract; a Router and its runtime come from one
  release.
- **One System One call per request stage, for remote deployments too.** The
  Router no longer splits a stage's questions to a remote System One API into
  calls of at most eight texts. A System One API you run must accept a batch
  with as many entries as the stage asks about, up to `judge.max_candidates`
  per judge.
- **The Kubernetes operator's `SemanticRouter` resource follows the new
  structure.** `spec.config.routing` is refused; its fields are
  `spec.config.recipes` (each recipe's signals, projections, decisions,
  model bindings and plugins), with `spec.config.entrypoints`,
  `spec.config.plugins` and `spec.config.models` (the facts of
  `providers.models[]`).
- **`IntelligentRoute` resources that use the removed signal types need a
  hand rewrite.** `vllm-sr config migrate` rewrites configuration files, not
  cluster resources: rewrite `events` and `structure` signals as `keywords`
  rules, and the other removed types as the migration guide shows, before
  you upgrade the CRDs.
- **The response headers are a public set and one trace**, with no aliases
  for the old names (see [Headers](#headers)). A custom gateway that routes on
  `x-selected-model` must match `x-vsr-route-model`, and clients that read the
  matched-signal or decision headers read the `x-vsr-trace` members instead.

## Behavior to expect

- **`jailbreak` and `pii` rules whose model gives no answer do not match.**
  The default `on_unknown` is `no_match`, as it was: a guard whose decision
  model is down lets traffic through. `on_unknown: match` is the explicit
  opt-in to fail closed, and migrate writes it for the `on_error: block` of
  earlier releases. Text the model could not read still counts as an attack
  or as personal data.
- **Thresholds were re-fitted** where the old value belonged to another
  model. A rule without a `predicate` takes the answering model's calibrated
  threshold, or 0.5 with an `uncalibrated_preset` warning, when it is a
  domain, user feedback, modality or PII rule or reads its preset's positive
  option (`FACT_CHECK_NEEDED`, `unsafe`, `refusal`, `jailbreak`). Any other
  rule without one matches on the model's own selection, and a hallucination
  rule on the model's own span thresholds.
- **A `response_guard` that can refuse or rewrite an answer holds a streamed
  one.** The Router buffers the stream of a decision whose guard blocks or
  writes notes until the guard has read the whole answer, then sends its
  events or the refusal as the protocol's stream error. Time to first token
  grows only for those decisions, and a held answer is bounded at 64 MiB. A
  guard that only annotates or records keeps true streaming.
- **Span-needing rules must resolve to a span-capable deployment**, or the
  configuration fails with `span_required`. The default decision model,
  vela3-fast, returns spans.
- **RAG relevance and tool selection are judged by the decision model**, and
  semantic cache hits are confirmed by its `same_request` question. The
  response cache's semantic match defaults to embedding recall (`top_k: 3`,
  `min_similarity: 0.8`) confirmed by `same_request` (`min_score: 0.8`);
  `mode: exact` is unchanged.
- **Every `match.judge` takes `max_candidates`.** A choice judge reads up to
  255 candidates (the default); a judge asked about each candidate reads 64 by
  default. More candidates never fail a request: the judge reads the first by
  recall rank, or the most recent without recall, records a
  `match.judge.truncated` span event and counts
  `vsr_match_judge_truncated_total`. A tool catalog of more than 255 tools is
  cut to its first 255, or recalled with `bm25`.
- **RAG and memory sources may judge without recall**, and a RAG source over
  the in-memory vector store may recall with `bm25`. Context compression
  judges each segment against the latest user message, by default the 64
  that `bm25` ranks first.
- **The embedding model is for recall only and starts lazily.** The
  pre-declared `vela-embedding` starts only when the response cache, memory, a
  vector store or a RAG source recalls with it. A remote `openai_embeddings`
  deployment declares `capabilities.dimensions` and `capabilities.modalities`,
  and the Router refuses at load a recall deployment that does not embed text
  or a dimension the index does not have.
- **Prompt cache affinity moved into `multi_factor`.** With the `hybrid`
  selector gone, the weighted `multi_factor` objective favors the candidate
  that served a session's earlier turns (a later turn, carried history, a
  `previous_response_id` or a `prompt_cache_key`), as a bounded tie-breaker.
- **Decisions declare their output with `output: {kind, action}`.** An image
  decision answers a plain-text request with a generated image on the
  Responses API: the Router adds the hosted `image_generation` operation and
  selects only a model that can generate it. The load fails with
  `output_capability_missing` when a model does not declare what the output
  needs, and with `output_not_executable` for what this release cannot
  execute (image edits, audio, video) or an API that cannot carry it. A Chat
  Completions or Messages client of an image decision is refused with
  `unsupported_capability`.
- **The Router has an Images API entry.** `POST /v1/images/generations` is
  routed on its prompt and answered only by a decision whose output is an
  image; any other request there gets `400 image_route_required` before a
  model is called. `api_format: images` reaches OpenAI-compatible
  image-generation backends.
- **`systemone` questions with `input.image: true` route on what a request's
  images show**, answered by a decision model that reads images.
- **Identity headers are trusted only where configured.** A standalone Router
  drops client-sent identity headers unless a listener sets
  `identity.trust_headers: true`; policies that read the gateway's identity
  (authz signals, rate limits by user or group, per-user keys, and
  `global.router.headers.debug: authenticated`) fail to load in standalone
  mode without an identity source. Debug headers reach only the callers
  `global.router.headers.debug` allows.
- **Delegated credentials are an auth mode of every upstream.** An upstream's
  `auth` is a static key (`api_key_env`, `api_key_file`, or a backend's
  `api_key`), `credential: delegated`, or neither, for provider backends,
  decision-model and remote embedding deployments, `escalate` judges and
  `api_format: systemone` targets alike. `global.services.authz.delegated_credential`
  names the header the authenticator writes the user's credential to (default
  `x-vsr-delegated-credential`). A call made for a client request carries that
  request's credential, or the request is answered `401` before the call; a
  call the Router makes on its own behalf never does, and a delegated upstream
  refuses it. `backend_refs[].trusted_gateway` and
  `global.services.authz.gateway_credential` are removed: the Router and
  migrate refuse them and name the new form. See
  [Security Hardening](../installation/security-hardening#delegate-the-users-credential-to-an-upstream).

## Headers

Every response of the routing pipeline carries one public set, on cache hits,
fast responses, blocks, Looper answers and routing errors as well as upstream
answers: `x-vsr-schema-version` (now 3), `x-vsr-config-version`,
`x-vsr-response-path`, `x-vsr-selected-recipe`, `x-vsr-selected-decision`,
`x-vsr-selected-model`, `x-vsr-replay-id`, `x-vsr-warnings` and the configured
cost (`x-vsr-cost`, `x-vsr-cost-currency`). Everything else the Router reports
rides in `x-vsr-trace`, an RFC 8941 dictionary of at most 4 KB, and only for a
caller `global.router.headers.debug` allows: past 4 KB the lowest-priority
members are left out and `truncated` ends it. The renames keep no aliases.

| Old | New |
| --- | --- |
| `x-vsr-matched-*`, `x-vsr-context-token-count` | The trace's `signals` member |
| Decision details, retention, prompt cache, Looper, protocol and effective-token headers, cache similarity | Trace members (`ranking`, `confidence`, `algorithm`, `cache_similarity`, ...) |
| `x-vsr-selected-modality` | The trace's `modality` member, the selected decision's output kinds |
| `x-vsr-protocol-warnings`, `x-vsr-response-warnings` | `x-vsr-warnings`, an RFC 8941 list of codes |
| `x-vsr-cache-hit`, `x-vsr-fast-response` | `x-vsr-response-path: cache` or `fast_response` |
| `x-selected-model` (the Envoy route key) | `x-vsr-route-model` (internal); custom gateway routes must match it |
| `x-disable-router-memory` | `x-vsr-memory: off` |
| `x-sr-bench-*` request headers, and the benchmark receipt headers | `x-vsr-bench-*`; the receipts are trace members, so sr-bench needs debug access |
| Retired Looper hop headers; the unused KV transfer status | Removed |
| `x-session-id`, `x-claude-code-session-id` | `x-vsr-session-id`; `global.router.headers.session_id_headers` lists the client names also accepted, these two by default |

The Router drops its other `x-vsr-*` headers when a client sends them, and
every `x-vsr-*` header a provider returns. The one it reads from the
authenticator is `x-vsr-delegated-credential`, trusted like the identity
headers when `global.services.authz.delegated_credential` is enabled; no
upstream receives the header, and only a delegated upstream its value.

The [response headers reference](../troubleshooting/vsr-headers) lists every
header the registry defines.

## Removed test profiles

The end-to-end profiles of removed backends and of the Vela 1.0 encoders,
which cannot answer System One questions, are deleted:
`local-classifier-backend`, `category-remote-backend`,
`complexity-remote-backend` and `preference-external-routing` (remote
classifier backends); `ml-model-selection` (the trained selectors);
`multimodal-routing` (embedding image anchors) and `vela-omni` (audio
routing); `vela-halu` and `vela-shield` (Vela 1.0 Halu and Shield). The
baseline profiles run on a deterministic fake System One backend, and image
routing is tested again on `systemone` `input.image`.

## Measurements

The measurements behind the zero-shot presets (harm categories, RAG and
cache judges, tool selection, the turn relation and complexity questions,
the trained-ID fix and output intent) run on public data and land in the
calibration tables and this note as their results are published; they are
not gates for the release.

## Known limits

- **Multimodal output is phase 1:** image generation, and text with an image
  from one omni model. Image edits, speech and audio (phase 2) and video
  (phase 3) fail at load with `output_not_executable` until their codecs land.
- **No decision model reads audio yet**, so `input.audio` is refused on every
  deployment; `input.video` needs a model that declares it.
- **Remote decision models have no calibrated thresholds**; their presets use
  0.5 until a rule writes its own `predicate`.
- **Calibrated thresholds exist only for vela3-fast and Vela 2.0**; the
  other families use 0.5 with the `uncalibrated_preset` warning.
