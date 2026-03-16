# Signal

## Overview

`signal/` is the detection layer of `routing`.

Signals define named detectors under `routing.signals`. A decision then references those names from `routing.decisions`, so detection stays reusable and route logic stays readable.

This tutorial group maps directly to the fragment tree under `config/signal/`, but the docs are organized by extraction style:

- `heuristic/` for request-shape, lexical, identity, and lightweight detector signals
- `learned/` for embedding- or classifier-driven signals that rely on router-owned model assets

## Key Advantages

- Reuses one detector across multiple decisions.
- Keeps detection logic separate from route outcomes.
- Lets one route combine lexical, policy, semantic, and safety inputs.
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

The latest signal docs still cover every family under `config/signal/`, but they are grouped into two second-level categories so the runtime cost and dependency model stay clear.

### Heuristic Signals

These signals route from explicit rules, request form, or lightweight detectors without depending on router-owned classifier models.

| Signal family | Fragment directory | Purpose | Doc |
|---------------|--------------------|---------|-----|
| `authz` | `config/signal/authz/` | route from identity, role, or tenant policy | [Authz](./heuristic/authz) |
| `context` | `config/signal/context/` | route by effective token-window needs | [Context](./heuristic/context) |
| `keyword` | `config/signal/keyword/` | route from lexical or BM25-style matches | [Keyword](./heuristic/keyword) |
| `language` | `config/signal/language/` | route by detected request language | [Language](./heuristic/language) |
| `modality` | `config/signal/modality/` | route by text, image-generation, or mixed flow shape | [Modality](./heuristic/modality) |

### Learned Signals

These signals use embeddings or classifier models and typically rely on `global.model_catalog` assets or module config.

| Signal family | Fragment directory | Purpose | Doc |
|---------------|--------------------|---------|-----|
| `complexity` | `config/signal/complexity/` | detect hard vs easy reasoning traffic | [Complexity](./learned/complexity) |
| `domain` | `config/signal/domain/` | classify the request topic family | [Domain](./learned/domain) |
| `embedding` | `config/signal/embedding/` | match by semantic similarity | [Embedding](./learned/embedding) |
| `fact-check` | `config/signal/fact-check/` | detect prompts that need evidence verification | [Fact Check](./learned/fact-check) |
| `jailbreak` | `config/signal/jailbreak/` | detect prompt-injection or jailbreak attempts | [Jailbreak](./learned/jailbreak) |
| `pii` | `config/signal/pii/` | detect sensitive personal data | [PII](./learned/pii) |
| `preference` | `config/signal/preference/` | infer response-style preferences | [Preference](./learned/preference) |
| `user-feedback` | `config/signal/user-feedback/` | detect correction or escalation feedback | [User Feedback](./learned/user-feedback) |

Keep these rules in mind:

- keep signals named and reusable
- keep signals detection-only; routing outcomes belong in `decision/`
- keep model choice separate; that belongs in `algorithm/`
- keep route-side behavior separate; that belongs in `plugin/`
