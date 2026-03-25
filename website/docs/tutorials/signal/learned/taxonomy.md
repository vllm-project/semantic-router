# Taxonomy Signal

## Overview

`taxonomy` binds routing signals to the output of a named taxonomy classifier instance. It maps to `config/signal/taxonomy/` and is declared under `routing.signals.taxonomy`.

This family is learned: the classifier itself is loaded from `global.model_catalog.classifiers[]`, while the signal layer binds one classifier tier or category into a normal named routing fact.

## Key Advantages

- Keeps classifier assets reusable across multiple routes.
- Lets decisions consume stable tier or category names instead of hidden runtime strings.
- Supports taxonomy-local thresholds and contrastive metrics without turning `routing.signals` into a custom expression language.
- Scales to multiple classifier instances because signals bind by classifier name.

## What Problem Does It Solve?

Some routing policies depend on a maintained taxonomy package rather than a single keyword or embedding rule. You may want one startup-loaded classifier to map requests into privacy, security, or standard traffic tiers, then expose only the specific tier/category bindings that the routing graph should see.

`taxonomy` solves that by separating classifier loading from signal consumption:

- `global.model_catalog.classifiers[]` owns the reusable taxonomy package
- `routing.signals.taxonomy[]` binds specific tiers or categories into named routing signals
- `routing.projections` can consume classifier metrics such as `contrastive`

## When to Use

Use `taxonomy` when:

- requests must be classified against a maintained taxonomy package
- one classifier result should feed several routes
- tier-level policy is more stable than raw exemplar files
- you need explicit bindings instead of implicit signal names such as `__tier__:*`

## Configuration

Source fragment family: `config/signal/taxonomy/`

```yaml
global:
  model_catalog:
    classifiers:
      - name: privacy_classifier
        type: taxonomy
        source:
          path: classifiers/privacy/
          taxonomy_file: taxonomy.json
        threshold: 0.55
        security_threshold: 0.7

routing:
  signals:
    taxonomy:
      - name: privacy_policy
        classifier: privacy_classifier
        bind:
          kind: tier
          value: privacy_policy
      - name: proprietary_code
        classifier: privacy_classifier
        bind:
          kind: category
          value: proprietary_code
```

Keep classifier names stable because taxonomy signals bind to those names directly.

Tier bindings resolve against the classifier's best threshold-qualified tier, not every tier with at least one over-threshold category. Category bindings still match the specific category directly.
