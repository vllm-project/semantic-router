# Event context signal

## Overview

`event_context` is a heuristic routing signal family for **structured event metadata** extracted from request text: event type, severity level, temporal urgency, and domain-specific action codes.

It maps to `config/signal/event-context/` and is declared under `routing.signals.event_context_rules`.

## Key Advantages

- Zero ML inference — all matching is regex-based, running in sub-millisecond time.
- Routes enterprise event-driven payloads (error alerts, audit logs, incident reports) to specialized model pools without requiring a domain classifier.
- Confidence is proportional to the number of matched criteria, giving the decision engine a graded signal.
- Temporal urgency detection (`urgent`, `immediate`, `asap`) routes time-sensitive events independently of event type.

## What Problem Does It Solve?

Keyword and embedding signals are tuned for natural-language queries. Structured event payloads — JSON fragments, alert messages, transaction error codes — contain well-defined fields that keyword matching cannot model cleanly. `event_context` provides a named, composable signal for each class of event without forcing operators to write fragile regex keyword rules.

## When to Use

Use `event_context` when:

- requests contain machine-generated event payloads (error alerts, audit logs, transaction failures)
- you want to route by severity tier independently of event type
- domain-specific action codes (e.g. `TXN_DECLINE`, `AUTH_FAIL`) should deterministically select a model pool
- time-sensitive events need to bypass standard latency-tolerant queues

## Configuration

```yaml
routing:
  signals:
    event_context_rules:
      - name: critical_payment_event
        event_types:
          - payment_failed
          - transaction_declined
        severities:
          - critical
          - high
        action_codes:
          - TXN_DECLINE
        temporal: true
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Rule name referenced in `routing.decisions[].rules` |
| `event_types` | list of strings | Event type patterns to match (case-insensitive word boundary) |
| `severities` | list of strings | Severity keywords: `critical`, `high`, `medium`, `low` |
| `action_codes` | list of strings | Domain-specific action codes (case-insensitive word boundary) |
| `temporal` | bool | When `true`, matches urgency markers: `urgent`, `immediate`, `asap`, `deadline`, `time-sensitive` |

A rule matches when at least one configured criterion is satisfied. **Confidence** is `0.5 + 0.5 × (matched_criteria / total_criteria)`, ranging from 0.75 (one of two criteria) to 1.0 (all criteria).

## Example Decision

```yaml
routing:
  decisions:
    - name: route_critical_event
      rules:
        type: event_context
        name: critical_payment_event
      modelRefs:
        - name: fast-response-model
```
