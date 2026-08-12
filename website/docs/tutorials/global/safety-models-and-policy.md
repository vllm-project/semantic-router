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

### Prompt Guard Backend

Jailbreak detection selects a backend via two mutually exclusive fields - set at most one:

- `prompt_guard.variant` picks a local Candle-backed model:
  - `candle`: the in-process Candle model (LoRA/BERT auto-detect, falling back to ModernBERT).
  - `mmbert32k`: the in-process mmBERT-32K model (32K context, YaRN RoPE, multilingual).
- `prompt_guard.protocol` picks a remote HTTP backend, requiring an `external_models` entry with `model_role: guardrail`:
  - `http_chat`: an external model called through a generative chat-completion prompt (Qwen3Guard-style).
  - `http_classify`: an external sequence-classifier endpoint speaking the widely-used HuggingFace text-classification pipeline contract - `POST {endpoint}/classify` with `{"inputs": "<text>"}`, returning every label's score, not just the top prediction. This lets a self-hosted classifier (wrapped by an existing `transformers` pipeline, or a Text Embeddings Inference deployment) plug in without disguising it as a chat completion.

```yaml
global:
  model_catalog:
    modules:
      prompt_guard:
        protocol: http_classify
        jailbreak_mapping_path: models/custom-classifier/jailbreak_type_mapping.json
        positive_labels: [INJECTION]   # this model's positive class isn't named "jailbreak"
        threshold: 0.7
      classifier:
        domain:
          model_ref: domain_classifier
          threshold: 0.5
          use_mmbert_32k: true
    external:
      - model_role: guardrail
        llm_endpoint:
          address: 127.0.0.1
          port: 8811
        llm_model_name: my-custom-classifier
```

Notes:

- Setting both `variant` and `protocol` fails config validation - `variant` selects a local model, `protocol` selects a remote one.
- `positive_labels` (optional): the `jailbreak_mapping` label(s) that count as unsafe, for models whose positive class isn't literally `jailbreak` (e.g. `INJECTION`, `malicious`). Multiple labels are summed when computing `risk_score`. Unset defaults to the single label `jailbreak`. If configured, at least one entry must exist in the loaded `jailbreak_mapping`'s labels or the router fails to start - a misconfigured label can no longer silently mean "never detected."
- `http_classify` response labels are matched against `jailbreak_mapping` by name, not by array position (the server is free to order them however it likes, e.g. sorted by score). Its default timeout is 5s (lightweight forward pass); `http_chat`'s default stays at 30s (generative call). Both are overridable via the external model's `timeout_seconds`.
- `http_chat` is restricted to this binary guardrail use case; it isn't offered for N-way classifiers (`classifier.domain`, `complexity`) since a generative model can't reliably produce a calibrated multi-class distribution.
- Breaking change: the legacy `use_modernbert`, `use_mmbert_32k`, and `use_vllm` boolean flags are removed, and the single `backend` field from an earlier revision of this feature is split into `variant`/`protocol`. Migrate `use_mmbert_32k: true` to `variant: mmbert32k`, and `use_vllm: true` (plus its `external_models` entry) to `protocol: http_chat`.
- **When both `variant` and `protocol` are unset, the effective default is `variant: mmbert32k`, not `candle`.** A config built directly in code without going through canonical default resolution falls back to `candle` when both are empty. Canonical resolution (the path the dashboard, canonical export, and any config that doesn't set every field explicitly all go through) starts from a baseline where `variant` is already set to `mmbert32k`, matching the bundled `mmbert32k-jailbreak-detector-merged` model it also defaults `model_id` to - so simply deleting `use_mmbert_32k: false` (or `use_modernbert: false`) without adding anything in its place does **not** get you `candle`. If you were relying on the plain Candle backend, set `variant: candle` explicitly.

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
