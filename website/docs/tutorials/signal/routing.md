# Routing Signals

## Overview

Routing signals identify what the request is about.

This page covers the request-understanding families that typically drive the first route match:

- `config/signal/keyword/`
- `config/signal/domain/`
- `config/signal/embedding/`
- `config/signal/fact-check/`

## Key Advantages

- Supports both deterministic and semantic routing.
- Lets one route combine topic, intent, and verification signals.
- Gives teams a clean path from simple keyword rules to stronger semantic matching.
- Keeps request classification reusable across multiple decisions.

## What Problem Does It Solve?

Many routing graphs start with a single weak classifier or hard-coded keywords. That breaks down when prompts are paraphrased, domain boundaries matter, or factual traffic needs extra handling.

Routing signals solve that by giving you multiple ways to identify request intent without tying the detection logic to a single decision.

## When to Use

Use routing signals when:

- the route depends on topic, intent, or factuality
- lexical rules are either too weak or too expensive on their own
- one detector should feed several different decisions
- you want to grow from simple rules into semantic routing without changing the top-level contract

## Configuration

Use the families below under `routing.signals`.

### Keyword

Source fragment family: `config/signal/keyword/`

```yaml
routing:
  signals:
    keywords:
      - name: urgent_keywords
        operator: OR
        keywords: ["urgent", "immediate", "asap"]
```

Use this when lexical matches are sufficient and you want deterministic behavior.

### Domain

Source fragment family: `config/signal/domain/`

```yaml
routing:
  signals:
    domains:
      - name: business
        description: Business and management traffic
        mmlu_categories: [business]
```

Use this when one route should activate for a stable topic family.

### Embedding

Source fragment family: `config/signal/embedding/`

```yaml
routing:
  signals:
    embeddings:
      - name: support_intent
        threshold: 0.7
        examples:
          - reset my password
          - update my billing details
```

Use this when keyword matching is too brittle and semantic similarity matters.

### Fact Check

Source fragment family: `config/signal/fact-check/`

```yaml
routing:
  signals:
    fact_check:
      - name: needs_verification
        threshold: 0.6
```

Use this when a route should treat factual claims differently from creative or open-ended traffic.
