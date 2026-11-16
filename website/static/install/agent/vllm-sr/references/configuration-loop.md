# Configure vLLM Semantic Router

A configuration is one document at schema `v0.5`. This reference shows where
each part goes, then the workflow that changes a running stack. Inspect only
the schema the change needs, from the CLI's bundled copy or the running Router:

```bash
vllm-sr config schema --section recipes.signals
vllm-sr config schema --endpoint "$ROUTER_ORIGIN" --surface algorithm:multi_factor
```

Follow section child paths; use `--expanded` or `--full` only when needed.
Installed help and the Router's `GET /api/v1` define the supported contract.

## Where things go

| Section | Holds |
| --- | --- |
| `listeners` | Public listeners: address, port, `timeout`, client `api_keys`, the native System One grants (`systemone.models`) and the identity headers a listener trusts (`identity.trust_headers`) |
| `providers` | The models requests are routed to: `backend_refs` (each with `endpoint`, `auth`, `headers`), and the facts routing reads (`capabilities`, `modalities`, context window, `pricing`) |
| `recipes` | Flat routing profiles: each one's `signals`, `projections`, `decisions`, `strategy`, `fallback`, `model_bindings` and default `plugins` |
| `entrypoints` | Public model names mapped to recipes; with several recipes, exactly one is `default: true` |
| `global` | Shared defaults: `global.plugins`, `global.router` (strategy, fallback, headers, learning), `global.stores`, `global.services` and `global.model_catalog` (the deployments the Router asks) |

Defaults live in `global`, and a recipe or a decision overrides them field by
field: `global.plugins` runs on every request, a recipe's `plugins` override
it, and a decision's `plugins` override the recipe's, merging mappings key by
key. `global.model_catalog.bindings.<signal type>` names the deployment of a
signal family, a recipe's `model_bindings` override it, and a rule's own
`deployment` overrides both. A request for a concrete provider model bypasses
recipes; `vllm-sr/auto` enters the default recipe.

With one recipe, no `entrypoints` are needed: it is published as
`vllm-sr/auto`. With several, map them:

```yaml
recipes:
  - name: balanced
    decisions: []
  - name: private
    decisions: []
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: balanced
    default: true
  - model_names: [acme/private]
    recipe: private
```

## Signals are questions to the decision model

Every model-judged signal is a question the decision model answers, by
default the deployment `primary` (vela3-fast). Three layers:

- **Heuristics** are deterministic code that asks no model: `keywords`,
  `context`, `conversation`, `input_modality`, `language`, `metadata` and
  `authz`.
- **Application signals** are presets: `domains`, `fact_check`,
  `user_feedbacks`, `jailbreak`, `safety`, `pii`, `hallucination`, `refusal`,
  `turn_relation`, `complexity` and `modality`.
- **`systemone` rules** are questions you write: `choice`, `noul` (yes or no),
  `score`, `set` or `span`.

A rule matches when the answer meets its `predicate` (`gt`, `gte`, `lt`,
`lte`) for its `labels`. A rule without a `predicate` takes the answering
model's calibrated threshold, or 0.5 with an `uncalibrated_preset` warning. A
rule that no decision, projection or plugin reads is never asked.

```yaml
recipes:
  - name: default
    signals:
      systemone:
        - name: needs_tools
          question:
            type: noul
            instructions: Does answering this request need a tool call?
          predicate: {gte: 0.7}
      jailbreak:
        - name: attack
          on_unknown: match
      complexity:
        - name: difficulty
    decisions:
      - name: refuse-attacks
        priority: 300
        rules:
          operator: AND
          conditions:
            - {type: jailbreak, name: attack}
        modelRefs: [{model: MODEL}]
      - name: hard-with-tools
        priority: 200
        rules:
          operator: AND
          conditions:
            - {type: complexity, name: difficulty, label: hard}
            - {type: systemone, name: needs_tools}
        modelRefs: [{model: MODEL}]
```

A late or failed answer leaves a rule unknown. The decision's
`rules.on_unknown` (or a condition's `on_error`) says what that means; by
default an unknown rule doesn't match, so a guard whose decision model is down
lets traffic through. `on_unknown: match` on a `jailbreak` or `pii` rule fails
closed. The [System One reference](https://vllm-sr.ai/docs/system-one/reference/presets)
lists every preset, its labels and the calibrated thresholds.

Each decision model answers some question types, and the load refuses a
question its deployment can't answer:

| Model | Answers |
| --- | --- |
| vela3-fast, the default `primary` | `choice`, `noul`, `span` (the `router` head); no `score` |
| vela3-lite, -mini, -flash, -turbo | `choice`, `noul`, `score`, `span` (the `router` head) |
| Decision 1.0 to 3.0 (the d3 models) | `choice`, `noul`, `score`; no `span` |
| Vela 2.0 | all five; only 0.8B, 4B and 9B have the `broad` span head |

A `set` question runs as one `noul` per label, so a model that answers `noul`
answers it. `pii`, `hallucination` and `span` questions need spans
(`span_required`), any other unanswerable type fails with
`question_type_unsupported`, and a span `head` the model lacks with
`span_head_unsupported`. Bind the rule or its signal type to a deployment
that answers it. A remote `systemone` deployment declares
`capabilities.question_types`; without them its API's listing decides when
the configuration prepares, and its span heads are unknown at load.

## Selectors, executors and plugins

A decision's `algorithm` chooses among its `modelRefs`. Selectors pick one:
`static` (the first), `random`, `multi_factor` (weighted quality, latency,
cost and load), and `systemone`, which asks the decision model which candidate
fits the request from their descriptions:

```yaml
recipes:
  - name: default
    decisions:
      - name: code
        priority: 100
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: small}, {model: large}]
        algorithm:
          type: systemone
          systemone:
            instructions: Which model should answer this request?
            candidates:
              small: Fast model for routine edits
              large: Strong model for hard refactors
```

When the decision model is not ready or answers invalidly, the first
`modelRef` answers. Executors call several models: `escalate` calls them in
order and stops at the first accepted answer (`escalate.accept.confidence`,
`accept.systemone` with the `answer_check` question, or `accept.refusal`),
and `fusion`, `remom` and `ratings` combine answers. `algorithm.on_error` is
shared by every executor.

Plugins act on the signals they name: `response_cache`, `context` (RAG
sources, memory, compression), `tools`, `response_guard`, `masking`,
`router_replay`, `system_prompt`, `header_mutation`, `request_params`,
`fast_response`. Discover one with
`vllm-sr config schema --surface plugin:response_cache`.

- **Compression for routing** is `compress.signals` of the `context` plugin,
  set in `global.plugins` or a recipe's `plugins`; the signals read the
  request before a decision is chosen, so a decision's own `context` plugin
  refuses it at load. A `compress` block with only `signals` leaves the
  request the model receives whole. `global.model_catalog.modules` no longer
  exists; migrate moves its `prompt_compression` here.
- **A guard that can refuse or rewrite an answer** (a `response_guard` with a
  `block` rule, an `annotate` rule with `details`, or `on_unknown: block`)
  also guards streamed answers: the Router holds the stream until the guard
  has read the whole answer, then sends it, or sends the protocol's stream
  error instead (HTTP 200, no answer bytes). A guard that only records keeps
  live streaming.
- **The response cache on a guarded decision** stores only what the guard
  released, and serves a semantic hit as is, without the guard; see
  [Optimization](https://vllm-sr.ai/install/agent/vllm-sr/references/optimization.md#caches-and-judge-caps).

## Similarity is a `match` block

Every feature that finds similar items (the response cache, RAG and memory
sources, context compression, tool selection) writes one `match` block, with
an optional `recall` stage (embedding or BM25 similarity) and an optional
`judge` stage (a System One question about each candidate, or a choice among
them):

```yaml
recipes:
  - name: default
    decisions:
      - name: default-route
        priority: 100
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: MODEL}]
        plugins:
          - type: response_cache
            configuration:
              enabled: true
              mode: semantic
              semantic:
                match:
                  recall: {method: embedding, top_k: 3, min_similarity: 0.8}
                  judge: {preset: same_request, min_score: 0.8, max_candidates: 3}
```

These are the response cache's defaults, with an explicit cap: recall the 3
most similar cached requests, then serve one that the `same_request` judge
scores at least 0.8. `judge.max_candidates` caps what a judge reads: 255 at most (and by default)
for a choice judge such as `tool_rank`, 64 by default for a per-candidate judge
such as `relevance` or `same_request`. More candidates are truncated, never an
error: the judge reads the first by recall rank and counts
`vsr_match_judge_truncated_total`. A block needs at least one stage.

## Embeddings serve recall only

No decision reads embeddings. The embedding deployment, `vela-embedding` by
default, starts only when a recall stage uses it. Name another one the way the
decision model is named, and give a store indexed with a different model its
own:

```yaml
global:
  model_catalog:
    deployments:
      remote-embed:
        provider: openai_embeddings
        endpoint: https://embed.example.com/v1
        model: BAAI/bge-m3
        capabilities: {dimensions: 1024, modalities: [text]}
    embeddings:
      default: {deployment: remote-embed}
```

A store's `embedding: {deployment, dimension}` overrides the default. The
Router's embeddings API, `embeddings.multimodal` and embedding signals are
gone; a decision about an image asks a `systemone` question with
`input.image: true`.

## Output contracts and images

A decision that answers with a generated image declares it, and every model it
names declares the capability:

```yaml
providers:
  models:
    - name: painter
      api_format: images
      capabilities: [image_generation]
      modalities: {input: [text], output: [image]}
      backend_refs:
        - {name: images, provider: vllm, endpoint: "images.example.com:8000"}
recipes:
  - name: default
    signals:
      modality:
        - {name: wants_image, labels: [image]}
    decisions:
      - name: image
        priority: 100
        rules: {operator: AND, conditions: [{type: modality, name: wants_image}]}
        output: {kind: image, action: generate}
        modelRefs: [{model: painter}]
```

`POST /v1/images/generations` reaches only a decision whose output is an
image; anything else there gets `400 image_route_required`. The load fails with
`output_capability_missing` when a model lacks what the output needs, and with
`output_not_executable` for image edits, audio or video.

## Headers and identity

Every routed response carries the public headers. The debug header
`x-vsr-trace` goes only to callers that `global.router.headers.debug` allows:
`none` (the default), `any` (every client that sends `x-vsr-debug: true`; for
loopback stacks, development and sr-bench) or `authenticated` (the
`debug_principals` an authenticating gateway asserts):

```yaml
global:
  router:
    headers:
      debug: any
```

A standalone Router drops client-sent identity headers unless a listener sets
`identity.trust_headers: true`. Policies that read the caller's identity
(authz signals, per-user keys, rate limits by user, `debug: authenticated`)
refuse to load without one.

### Upstream credentials

Every upstream the Router calls has one `auth` block: a provider's
`backend_refs[].auth` (which also covers `escalate` judges and `api_format:
systemone` targets) and a deployment's
`global.model_catalog.deployments.NAME.auth`. It takes one of three modes:

- **A static key**, `api_key_env: VAR` (or `api_key_file` on a deployment):
  the call acts as the Router.
- **Delegated**, `credential: delegated` and no key: the call carries the
  requesting user's own credential, which an authenticator in front of the
  Router issued for this request.
- **None**: neither.

`auth.header` and `auth.prefix` say how the credential is sent, in both
modes (by default the provider's own header for a backend, `Authorization:
Bearer` for a deployment; `prefix: ""` sends it bare).

Delegated credentials need three things, and the user's decision: ask first.

1. An authenticator in front (Envoy `ext_authz` or the user's gateway) that
   authenticates the client, strips client copies of the headers it writes,
   and writes the user's id (`global.services.authz.identity.user_id_header`)
   and the user's credential. A standalone Router keeps those headers only on
   a listener with `identity.trust_headers: true`, which only the
   authenticator may reach.
2. The contract that names that header:
   `global.services.authz.delegated_credential: {enabled: true}`. Its `header`
   defaults to `x-vsr-delegated-credential` and may not be `Authorization`,
   an API-key header or an identity header. It is off by default.
3. `auth: {credential: delegated}` on exactly the upstreams that must act as
   the user. A request that reaches one without a credential gets 401 with
   `delegated_credential_missing` before any upstream is called, and the
   Router never sends a user's credential on its own calls (probes, warmups,
   replay shadows, store building), so an upstream that only those reach
   needs a static key.

```yaml
global:
  services:
    authz:
      delegated_credential:
        enabled: true # the authenticator writes x-vsr-delegated-credential
providers:
  models:
    - name: team-llm
      backend_refs:
        - name: litellm
          endpoint: https://litellm.internal/v1
          provider: openai-compatible
          auth:
            credential: delegated
```

With any delegated upstream, every `response_cache` keeps `scope: user`, so
an answer obtained with one user's credential is served only to that user.
`backend_refs[].trusted_gateway`, `global.services.authz.gateway_credential`
and the `x-vsr-gateway-credential` header are removed with no rewrite: write
the forms above instead.

## Change a running stack

On the Docker target the Router serves `.vllm-sr/runtime-config.yaml`
(`runtime-config.NAME.yaml` for a stack named `NAME`), not `config.yaml`:
editing the source file changes nothing until it is applied. Derive the
candidate from the active document:

```bash
vllm-sr config get --format yaml --endpoint "$ROUTER_ORIGIN" > active.yaml
cp active.yaml candidate.yaml
# Edit the intended policy, then:
vllm-sr config validate --config candidate.yaml --endpoint "$ROUTER_ORIGIN"
vllm-sr config plan --config candidate.yaml --mode replace --endpoint "$ROUTER_ORIGIN"
vllm-sr config apply --config candidate.yaml --mode replace --endpoint "$ROUTER_ORIGIN"
vllm-sr config versions --endpoint "$ROUTER_ORIGIN"
```

Pass: `plan` prints `"valid": true` and `"changed": true`; `apply` exits 0 and
prints `"applied": true`; `config versions` lists the new version first. A
change that loads a model (a new decision-model rule, a new deployment) can
take minutes the first time: pass `--timeout 900`. A timeout doesn't cancel
the change; read `config versions` before retrying.

`apply` plans again and protects its write with an ETag; a separate earlier
`plan` does not pin it. Use the API's `If-Match` contract when the reviewed
revision must be fixed (the [HTTP planning example](https://vllm-sr.ai/install/agent/vllm-sr/references/http-configuration-plan.md)
does it step by step).

A change the running Router can't take (in standalone mode a listener's
address, port, timeout or TLS; with `--gateway extproc`, also the provider
topology) is `restart_required`: `config apply` saves it and prints
`Restart required: run vllm-sr serve to apply.`, and `vllm-sr status` reports
it until `vllm-sr serve --config config.yaml` applies it. Elsewhere, use the
deployment's own workflow, such as `helm upgrade`. To serve `config.yaml` again
and discard Dashboard edits: `vllm-sr serve --config config.yaml
--replace-active-config`. History and rollback are in
[Maintenance](https://vllm-sr.ai/install/agent/vllm-sr/references/maintenance.md#versions-and-rollback).

## Packaged recipes

Use `vllm-sr recipe builtin list`, then inspect `vllm-sr recipe builtin export
BUNDLE --output-dir DIR` or bind one to your models with
`vllm-sr recipe builtin init NAME --bundle BUNDLE --config config.yaml
--bindings bindings.yaml --model-name acme/auto --output recipe-config.yaml`.
Export preserves bundle bytes; init binds a recipe to actual providers and
refuses a decision without its declared `minimum_candidates`.
`--exclude-decision` creates a derivative with reduced coverage. A bundle, a
recipe name and a public entrypoint are different identities: `vllm-sr/auto`
cannot be rebound to a named recipe. Verify both
[routing and delivery](https://vllm-sr.ai/install/agent/vllm-sr/references/route-verification.md) afterwards.
