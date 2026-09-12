# Safety Models and Shared Policy

## Overview

`global.model_catalog` declares shared model assets and the modules that
use them. `global.services.authz` and `global.services.ratelimit` declare shared
identity and rate policy. Route-specific thresholds and actions still belong in
signals, decisions, and plugins.

## What Problem Does It Solve?

Jailbreak, PII, domain, fact-check, hallucination, and feedback capabilities
reuse model runtimes across routes. Defining those dependencies once keeps
route policy small and makes local versus remote processing visible.

## Key Advantages

- Reuses one model runtime across many route-local safety rules.
- Makes local and remote processing choices explicit.
- Separates shared identity/rate services from decision policy.

## When to Use

Override these settings when you need a different system model, execution
backend, threshold baseline, identity source, or rate-limit provider. Keep the
defaults when the bundled local models and policies meet your requirements.

## Configuration

### Local prompt guard

`variant` selects the local Candle-backed implementation. `mmbert32k` is the
canonical default; choose `candle` explicitly when that is the intended model.

```yaml
global:
  model_catalog:
    modules:
      prompt_guard:
        enabled: true
        variant: mmbert32k
        threshold: 0.7
```

### Remote prompt guard

Use `protocol` instead of `variant` for a remote guardrail. The two fields are
mutually exclusive. A remote guardrail also requires an entry under
`global.model_catalog.external` with `model_role: guardrail`.

```yaml
global:
  model_catalog:
    modules:
      prompt_guard:
        enabled: true
        protocol: http_classify
        threshold: 0.7
        positive_labels: [INJECTION]
    external:
      - name: guardrail-service
        model_role: guardrail
        llm_endpoint:
          address: guardrail.example.com
          port: 443
          protocol: https
        llm_model_name: prompt-guard
        llm_timeout_seconds: 5
```

`http_classify` expects the Router's supported classification contract;
`http_chat` uses a chat-completions prompt. Both send request text to the
configured service. Set `max_response_bytes` on the external model entry to
override the 1 MiB response limit.

The HTTP MCP classifier uses
`global.model_catalog.modules.classifier.mcp.max_response_bytes`. Its default
is 16 MiB.

### Content safety and prompt attacks

Use `routing.signals.safety` for content risks and `routing.signals.jailbreak`
for prompt injection, jailbreak and instruction hijacking. A harmful request
can contain no prompt attack, and an instruction hijack can ask for otherwise
harmless output.

The [Safety signal guide](../signal/learned/safety.md) covers two complementary
heads: **Safety** predicts `safe`/`unsafe`; **Hazard** predicts independent risk
categories. A category-specific rule first requires the Safety score to reach
its threshold, then checks whether any selected Hazard category reaches its
own threshold. The router skips Hazard inference when Safety is below threshold.

Omit a rule's `model` to use the native head configured under
`global.model_catalog.modules.safety`. Set `model` to an external classifier
name to use `POST /classify` instead. Native heads are independently owned by
the recipe, shared between its identical rule contracts, and released when the
recipe closes. Local artifact labels and activation are checked on loading.
External endpoints must return the complete declared label set. Safety scores
form a softmax distribution; Hazard scores are independent sigmoid values and
may sum to more than one.

The [content-safety fragment](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/safety/content-safety.yaml)
shows two HTTP heads with a privacy-specific policy and a general unsafe policy.
Replace `default-model` with a safety-capable provider alias and configure the endpoint addresses.
Thresholds are examples, not universal calibration. `rules.on_unknown:
fail_request` returns HTTP 503 when a required classification fails. A known
unsafe result selects the configured policy: the example refuses unsafe privacy
abuse and routes other risks to the backend. General content risk can include a
person in crisis who needs a supportive response, so it should not automatically
trigger a blanket refusal.

### Native classifier context

`max_sequence_length` on the domain, PII, prompt-guard, feedback, fact-check and
modality classifier modules is an explicit native mmBERT input budget. Zero
retains the historical 512-token budget. Larger values must fit the loaded
artifact's position capacity. Configure the separate Safety/Hazard head budgets
under `modules.safety.safety` and `modules.safety.hazard`.

For the supported local classifier paths, a budget above 512 also enables full
routing text instead of representative sampling or small security windows.
Over-budget inputs produce an inference error rather than a result computed
from an unseen truncation. Existing PII/jailbreak configurations with the
historical budget retain overlapping scans across the entire input. Choose a
budget supported by task-level quality and latency measurements; positional
capacity alone is not evidence of long-text accuracy.

For a prompt-guard checkpoint evaluated with token windows, configure
`modules.prompt_guard.window.size` and `window.overlap` alongside the total
`max_sequence_length` budget. This scans the original tokens and retains the
window with the highest combined positive-label probability. The model's
window size and decision threshold must be calibrated together; a larger
position capacity does not replace this inference policy. See
[Jailbreak Signal](../signal/learned/jailbreak.md#token-windows-for-a-local-classifier).

For native mmBERT embedding signals, set
`global.model_catalog.embeddings.semantic.embedding_config.full_context: true`
to use the loaded embedding model's complete context capacity. The default
keeps representative routing samples for latency. This affects the embedding
signal; unrelated semantic consumers keep their own policies.

### Remote category/domain classifier

Category/domain classification uses the shared `backend` block. The explicit
`model` is resolved by name in `global.model_catalog.external[]`; role and
response contract are validated at startup, so an unrelated classification
model cannot be selected accidentally. `protocol` describes the wire protocol
and `contract` describes the semantic response product. They are separate
axes. Category currently supports `http_classify` with
`label_distribution.v1`, which preserves the complete configured-label score
distribution used by domain matching and model selection. `deadline_ms` is
an optional per-backend request deadline and defaults to 5000.

```yaml
global:
  model_catalog:
    external:
      - name: domain-service
        model_role: classification
        llm_endpoint:
          address: domain-classifier.default.svc
          port: 8080
          protocol: http
        llm_model_name: domain-intent-v1
    modules:
      classifier:
        domain:
          category_mapping_path: models/mmbert32k-intent-classifier-merged/category_mapping.json
          fallback_category: other
          backend:
            protocol: http_classify
            contract: label_distribution.v1
            model: domain-service
            deadline_ms: 5000
```

Omit `backend` to retain local category inference. The deprecated
`use_modernbert` and `use_mmbert_32k` keys remain readable for local configs;
new canonical output uses `variant: candle`, `variant: modernbert`, or
`variant: mmbert32k`. An agreeing canonical and legacy selector is accepted,
while contradictory active selectors and both legacy selectors set to `true`
are rejected deterministically. `backend` is mutually exclusive with active
local selectors.

### On a classifier failure

An unreachable or invalid guardrail result is recorded as a signal error and
enters a decision tree as `Unknown`. Set root-level `rules.on_unknown` on the
consuming decision to resolve a terminal unknown as `no_match`, `match`, or
`fail_request`.

```yaml
global:
  model_catalog:
    modules:
      prompt_guard:
        enabled: true
        protocol: http_classify
        on_error: block
```

When `rules.on_unknown` is omitted, request-side jailbreak decisions retain
the existing `prompt_guard.on_error` behavior: `allow` (the default) tolerates
the failure and maps the terminal result to no match, so other content still
evaluates normally; `block` maps it to a match, treating the failure itself as
a positive detection, since an inference failure means the content could not
be verified safe.

The legacy `on_error` path applies to any prompt guard backend, local or
remote - not only the remote protocols above - and to both directions:
request-side jailbreak signal rules, including `method: contrastive` ones, and
the response-side `response_jailbreak` plugin, which scans LLM output with the
same backend. Response-side behavior is unchanged either way; the plugin's own
`action` decides: `block` returns a 403, `header` adds the response warning,
`none` stays silent.

Under the legacy path a failure is reported exactly as a real detection is. On
the request side that means the jailbreak signal fires at confidence `1.0`
with type `classification_error`, so `block` only closes a request if a
decision actually consumes the jailbreak signal (`type: jailbreak`) and acts
on it, typically with `fast_response` - without one it looks like a no-op. See
the `jailbreak-onerror` e2e profile's `block_on_classifier_error` decision for
a complete example.

:::note

This is not the same key as the `on_error` on a decision's classifier
condition, which takes `no_match` or `match`. That one answers "what should this
predicate evaluate to when the classifier fails"; `prompt_guard.on_error`
answers "was the content verified at all", for every rule the guardrail backend
serves. Both remain backward-compatible defaults only while the consuming rule
omits `rules.on_unknown`: setting `rules.on_unknown` disables every
condition-level `on_error` below it. See
[Classifier signals](../signal/learned/classifier.md).

:::

### Hallucination mitigation

The local detector uses `backend: candle`. An OpenAI-compatible remote detector
uses `backend: endpoint` with an absolute endpoint and model ID.

```yaml
global:
  model_catalog:
    modules:
      hallucination_mitigation:
        enabled: true
        detector:
          backend: endpoint
          endpoint: https://hallucination.example.com/v1
          model_id: KRLabsOrg/lettucedect-v2-qwen-2b
          include_explanation: true
```

The endpoint path does not provide the local NLI explainer used by some
cross-response checks. Configure route-local failure behavior accordingly.

### System model bindings

Signals and plugins resolve stable capability names through this catalog:

```yaml
global:
  model_catalog:
    system:
      prompt_guard: models/mmbert32k-jailbreak-detector-merged
      domain_classifier: models/mmbert32k-intent-classifier-merged
      pii_classifier: models/mmbert32k-pii-detector-merged
      fact_check_classifier: models/mmbert32k-factcheck-classifier-merged
      hallucination_detector: models/mom-halugate-detector
      hallucination_explainer: models/mom-halugate-explainer
      feedback_detector: models/mmbert32k-feedback-detector-merged
```

### Identity and rate limiting

```yaml
global:
  services:
    authz:
      fail_open: false
      identity:
        user_id_header: x-user-id
        user_groups_header: x-user-groups
      providers:
        - type: header-injection
          headers:
            openai: x-user-openai-key
    ratelimit:
      fail_open: false
      providers:
        - type: local-limiter
          rules:
            - name: premium-per-minute
              match:
                group: premium
              requests_per_unit: 120
              unit: minute
```

Only trust identity headers set or sanitized by an authenticated upstream.
`fail_open: true` trades availability for weaker enforcement and should be a
deliberate policy choice.

## Data and Security

- Local model variants keep inference in the Router process. Remote modules
  send the text they classify to their configured endpoints.
- Detector output is probabilistic. Calibrate thresholds on your corpus and
  keep least-privilege tool, provider, and storage controls in place.
- Store endpoint credentials in environment variables or Secrets. Do not place
  them in route descriptions or model IDs.
- See the
  [complete configuration example](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml)
  for all available model and policy groups.
