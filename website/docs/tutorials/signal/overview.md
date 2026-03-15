# Signal

## Overview

`signal/` is the detection layer of `routing`.

Signals define named detectors under `routing.signals`. A decision then references those names from `routing.decisions`, so detection stays reusable and route logic stays readable.

This tutorial group maps directly to the fragment tree under `config/signal/`.

## Key Advantages

- Reuses one detector across multiple decisions.
- Keeps detection logic separate from route outcomes.
- Lets one route combine lexical, semantic, safety, and operational inputs.
- Makes config reviews easier because signal names become stable policy building blocks.

## What Problem Does It Solve?

Without a signal layer, every decision has to inline its own detection logic. That creates duplication, makes route policies harder to audit, and mixes "what did we detect?" with "what should we do?".

Signals solve that by turning request understanding into a named catalog that the rest of the routing graph can compose.

## When to Use

Use `signal/` when:

- more than one route needs the same detector
- you want to mix different detection methods in one decision tree
- you need a clean boundary between detection, decision logic, algorithms, and plugins
- you want config fragments that map cleanly to `config/signal/`

## Configuration

In canonical v0.3 YAML, signals live under `routing.signals`:

```yaml
routing:
  signals:
    keywords:
      - name: urgent_keywords
        operator: OR
        keywords: ["urgent", "asap"]
```

The latest signal docs follow the same taxonomy as `config/signal/`:

| Signal family | Fragment examples | Purpose | Doc |
|---------------|-------------------|---------|-----|
| routing signals | `keyword`, `domain`, `embedding`, `fact-check` | classify what the request is about | [Routing Signals](./routing) |
| safety signals | `jailbreak`, `pii`, `authz` | gate unsafe or unauthorized traffic | [Safety Signals](./safety) |
| operational signals | `complexity`, `context`, `language`, `modality`, `preference`, `user-feedback` | adapt route behavior to request shape and user context | [Operational Signals](./operational) |

Keep these rules in mind:

- keep signals named and reusable
- keep signals detection-only; routing outcomes belong in `decision/`
- keep model choice separate; that belongs in `algorithm/`
- keep route-side behavior separate; that belongs in `plugin/`
