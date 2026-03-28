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
