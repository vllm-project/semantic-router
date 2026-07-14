# Safety, Models, and Policy

## Overview

This page covers the shared runtime model and policy blocks inside `global:`.

These settings define shared safety behavior, shared runtime model settings, and router-wide policy defaults.

## Key Advantages

- Keeps shared policy separate from route-local safety plugins.
- Centralizes built-in classifier and embedding model overrides.
- Makes authz, ratelimit, and selection defaults consistent.
- Gives the router one place to override system model bindings.

## What Problem Does It Solve?

The router depends on shared runtime models and shared policy defaults that are not tied to one route. If those settings are scattered across routes, the resulting behavior is hard to reason about and hard to change safely.

These `global:` blocks solve that by collecting shared model and policy overrides in one layer.

## When to Use

Use these blocks when:

- built-in safety and classification models need shared runtime settings
- signal or algorithm layers depend on shared embedding or external model settings
- authz or rate limits should apply router-wide
- one system capability should bind to a different internal model

## Configuration

### Prompt Guard and Classifier

```yaml
global:
  model_catalog:
    modules:
      prompt_guard:
        model_ref: prompt_guard
        use_mmbert_32k: true
      classifier:
        domain:
          model_ref: domain_classifier
          threshold: 0.5
          use_mmbert_32k: true
```

### Hallucination Detector Backend

The hallucination detector supports two backends via `hallucination_mitigation.detector.backend`:

- `candle` (default): the in-process Candle token classifier. Used when `backend` is unset or `candle`.
- `endpoint`: a generative span detector served behind any OpenAI-compatible server (for example, vLLM). One structured `json_schema` call returns typed spans and an optional explanation.

```yaml
global:
  model_catalog:
    modules:
      hallucination_mitigation:
        enabled: true
        detector:
          backend: endpoint                     # default: candle
          endpoint: http://127.0.0.1:8077/v1    # required for endpoint; absolute http(s) URL
          model_id: KRLabsOrg/lettucedect-v2-qwen-2b
          include_explanation: true             # request per-span explanations
```

Notes:

- The `endpoint` backend requires an absolute `http(s)` endpoint and a `model_id`; the config is rejected at load time otherwise. An unknown `backend` value is rejected rather than silently falling back to `candle`.
- The endpoint backend does not ship a local NLI explainer, so panel-mode fusion grounding (which needs NLI) gracefully skips under its `on_error` policy. NLI readiness (`/classify/nli`) stays reported as unavailable for this backend.
- If the endpoint is unreachable or returns a malformed response, detection fails open: the response passes through and the failure is recorded on the detection-error path rather than as a clean verdict.

### Embedding and External Models

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        mmbert_model_path: models/mom-embedding-ultra
        use_cpu: true
```

### Authz and Rate Limit

```yaml
global:
  services:
    authz:
      enabled: true
    ratelimit:
      enabled: true
```

### Model Selection and Looper Defaults

```yaml
global:
  router:
    model_selection:
      enabled: true
  integrations:
    looper:
      enabled: true
```

### System Models

```yaml
global:
  model_catalog:
    system:
      prompt_guard: models/mmbert32k-jailbreak-detector-merged
      domain_classifier: models/mmbert32k-intent-classifier-merged
```
