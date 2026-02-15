---
sidebar_position: 5
---

# Latency routing migration guide

Latency routing has moved from the signal layer to the decision algorithm layer.

- Preferred configuration: `decision.algorithm.type: latency_aware`
- Preferred algorithm block: `decision.algorithm.latency_aware`
- Legacy configuration (deprecated): `signals.latency` or `signals.latency_rules` with `conditions.type: latency`

Use this guide to migrate existing configs before the legacy path is removed.

## What changed

### Old pattern (deprecated)

- Define latency thresholds under `signals.latency` (CLI config) or `signals.latency_rules` (router config)
- Match latency with `rules.conditions[].type: latency`

### New pattern (recommended)

- Keep latency logic in decision algorithm:
  - `algorithm.type: latency_aware`
  - `algorithm.latency_aware.tpot_percentile`
  - `algorithm.latency_aware.ttft_percentile`
- Keep request understanding in signals (keyword/domain/embedding/...).

## Backward compatibility behavior

During the deprecation window, legacy latency configs can be auto-migrated, but only when migration is lossless.

Auto-migration requires all of the following:

1. The decision has exactly one legacy latency condition.
2. `rules.operator` is `AND`.
3. The latency condition references a valid legacy latency rule.
4. `decision.algorithm` is absent or `type: static`.
5. At least one non-latency condition remains after removing the legacy latency condition.

If these checks pass, the router/CLI:

- rewrites the decision to `algorithm.type: latency_aware`
- copies percentile values into `algorithm.latency_aware`
- removes legacy latency signal rules
- emits deprecation warnings

## Mixed old and new config is rejected

Do not mix legacy latency config with `algorithm.type: latency_aware` in the same file.

Example rejected mix:

- legacy `signals.latency` + `conditions.type: latency`
- and any decision already using `algorithm.type: latency_aware`

## Step-by-step migration

### 1. Find legacy latency usage

Look for:

- `signals.latency` (CLI-style config)
- `signals.latency_rules` (router-style config)
- `conditions.type: latency`

### 2. Convert each legacy latency decision

Move latency thresholds to the decision algorithm.

Before (legacy):

```yaml
signals:
  latency:
    - name: "low_latency"
      tpot_percentile: 10
      ttft_percentile: 10

decisions:
  - name: "fast_route"
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "other"
        - type: "latency"
          name: "low_latency"
    modelRefs:
      - model: "openai/gpt-oss-120b"
      - model: "gpt-5.2"
```

After (recommended):

```yaml
decisions:
  - name: "fast_route"
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "other"
    modelRefs:
      - model: "openai/gpt-oss-120b"
      - model: "gpt-5.2"
    algorithm:
      type: "latency_aware"
      latency_aware:
        tpot_percentile: 10
        ttft_percentile: 10
```

### 3. Remove legacy latency signals

After all related decisions are converted, remove:

- `signals.latency`
- `signals.latency_rules`
- any remaining `conditions.type: latency`

### 4. Validate by starting with your config

Start the router/CLI with your updated config and verify there are no legacy-latency warnings or migration errors.

## Common migration errors

- `legacy latency config ... cannot be used with decision.algorithm.type=latency_aware`
  - Cause: mixed old and new latency config in one file.
- `only static can be auto-migrated to latency_aware`
  - Cause: legacy latency condition appears in a decision using non-static algorithm.
- `multiple legacy latency conditions are not supported for auto-migration`
  - Cause: one decision has more than one `type: latency` condition.
- `... rules.operator=... cannot be auto-migrated; only AND is supported`
  - Cause: legacy latency condition is used with `OR`.
- `... no non-latency conditions remain`
  - Cause: decision only had latency condition and nothing else.

## Recommended next step

After migration, keep latency routing only in `algorithm.type: latency_aware` and treat request signals and model selection as separate layers.
