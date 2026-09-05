# TD058: Observability Metrics Module Boundaries

## Status

Open

## Owner Plan

PL0032 Architecture Debt Consolidation

## Release Relevance

Not release-blocking. The current exception preserves the structure gate while
the metrics surface is split into narrower modules.

## Scope

`src/semantic-router/pkg/observability/metrics/metrics.go` and the metric
registration and recording helpers it contains.

## Summary

The central metrics implementation contains request, latency, token, cache,
batch, and other observability metric definitions and recording helpers. Adding
provider prompt-cache metrics pushed the file beyond the shared structure limit,
so the structure gate now has a file-only exception for this path.

## Evidence

- `metrics.go` is 844 lines after provider prompt-cache metrics were added.
- The shared structure rule warns at 400 lines and errors at 800 lines.
- The new exception is limited to `file_checks`; function and nesting checks
  remain active.

## Why It Matters

Keeping unrelated metric families in one file increases review and ownership
costs and makes future observability changes more likely to require structure
exceptions.

## Desired End State

Metric families and their recording helpers are grouped into focused files,
allowing `metrics.go` to pass the shared file-size limit without an exception.

## Exit Criteria

- Provider cache and other metric families have focused registration and
  recording modules.
- `src/semantic-router/pkg/observability/metrics/metrics.go` is at or below
  the shared file-size limit.
- The `metrics.go` exception is removed from
  `tools/agent/structure-rules.yaml`.
