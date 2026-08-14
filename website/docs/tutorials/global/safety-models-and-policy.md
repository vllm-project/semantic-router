# Safety Models and Shared Policy

## Overview

`global.model_catalog` declares Router-owned model assets and the modules that
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
- The exhaustive maintained examples are in
  [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml),
  and the canonical grouping is defined in
  [`canonical_global.go`](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/config/canonical_global.go).
