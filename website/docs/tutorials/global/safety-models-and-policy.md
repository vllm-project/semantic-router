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
configured service.

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
