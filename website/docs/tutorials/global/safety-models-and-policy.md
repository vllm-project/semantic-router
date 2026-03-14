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
  prompt_guard:
    use_modernbert: false
  classifier:
    category_model:
      threshold: 0.6
```

### Embedding and External Models

```yaml
global:
  embedding_models:
    use_cpu: true
```

### Authz and Rate Limit

```yaml
global:
  authz:
    enabled: true
  ratelimit:
    enabled: true
```

### Model Selection and Looper Defaults

```yaml
global:
  model_selection:
    enabled: true
  looper:
    enabled: true
```

### System Models

```yaml
global:
  system_models:
    prompt_guard: models/mom-jailbreak-classifier
    domain_classifier: models/mom-domain-classifier
```
