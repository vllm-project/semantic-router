# Signal

## Overview

`signal/` is the detection layer of `routing`.

Signals define named detectors under `routing.signals`. A decision then references those names from `routing.decisions`, so detection stays reusable and route logic stays readable.
Cross-signal coordination and derived routing bands now live under `routing.projections`. `routing.projections.partitions` is the runtime home for exclusive domain or embedding partitions, while decisions can reference `routing.projections.mappings` outputs with `type: projection`. In DSL authoring, the same concepts show up as `PROJECTION partition ...` plus `PROJECTION score ...` / `PROJECTION mapping ...` blocks.
For the full projection workflow, canonical YAML contract, dashboard path, and DSL examples, see [Projections](../projection/overview).

This tutorial group maps directly to the fragment tree under `config/signal/`, but the docs are organized by extraction style:

- `heuristic/` for request-shape, lexical, identity, and lightweight detector signals
- `learned/` for embedding- or classifier-driven signals that rely on router-owned model assets or maintained detector modules

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
    embeddings:
      - name: technical_support
        threshold: 0.75
        candidates: ["installation guide", "troubleshooting steps"]
      - name: account_management
        threshold: 0.72
        candidates: ["billing information", "subscription management"]
  projections:
    partitions:
      - name: support_intents
        semantics: exclusive
        temperature: 0.3
        members: [technical_support, account_management]
        default: technical_support
    scores:
      - name: request_difficulty
        method: weighted_sum
        inputs:
          - type: embedding
            name: technical_support
            weight: 0.18
            value_source: confidence
    mappings:
      - name: request_band
        source: request_difficulty
        method: threshold_bands
        outputs:
          - name: support_escalated
            gte: 0.25
```

The latest signal docs still cover every family under `config/signal/`, but they are grouped into two second-level categories so the runtime cost and dependency model stay clear.

### Heuristic Signals

These signals route from explicit rules, request form, or lightweight detectors without depending on router-owned classifier models.

| Signal family | Fragment directory         | Purpose                                                                      | Doc                                |
| ------------- | -------------------------- | ---------------------------------------------------------------------------- | ---------------------------------- |
| `authz`       | `config/signal/authz/`     | route from identity, role, or tenant policy                                  | [Authz](./heuristic/authz)         |
| `context`     | `config/signal/context/`   | route by effective token-window needs                                        | [Context](./heuristic/context)     |
| `keyword`     | `config/signal/keyword/`   | route from lexical or BM25-style matches                                     | [Keyword](./heuristic/keyword)     |
| `language`    | `config/signal/language/`  | route by detected request language                                           | [Language](./heuristic/language)   |
| `structure`   | `config/signal/structure/` | route from request shape such as question counts or ordered workflow markers | [Structure](./heuristic/structure) |

### Learned Signals

These signals use embeddings or classifier models and typically rely on `global.model_catalog` assets or module config.

| Signal family | Fragment directory | Purpose | Doc |
|---------------|--------------------|---------|-----|
| `complexity` | `config/signal/complexity/` | detect hard vs easy reasoning traffic | [Complexity](./learned/complexity) |
| `domain` | `config/signal/domain/` | classify the request topic family | [Domain](./learned/domain) |
| `embedding` | `config/signal/embedding/` | match by semantic similarity | [Embedding](./learned/embedding) |
| `modality` | `config/signal/modality/` | classify text-only, image-generation, or hybrid output mode | [Modality](./learned/modality) |
| `fact-check` | `config/signal/fact-check/` | detect prompts that need evidence verification | [Fact Check](./learned/fact-check) |
| `jailbreak` | `config/signal/jailbreak/` | detect prompt-injection or jailbreak attempts | [Jailbreak](./learned/jailbreak) |
| `pii` | `config/signal/pii/` | detect sensitive personal data | [PII](./learned/pii) |
| `preference` | `config/signal/preference/` | infer response-style preferences | [Preference](./learned/preference) |
| `reask` | `config/signal/reask/` | detect repeated user questions as implicit dissatisfaction | [Reask](./learned/reask) |
| `kb` | `config/signal/kb/` | bind knowledge base labels or groups into named routing signals | [Knowledge Base](./learned/kb) |
| `user-feedback` | `config/signal/user-feedback/` | detect correction or escalation feedback | [User Feedback](./learned/user-feedback) |

Keep these rules in mind:

- keep signals named and reusable
- keep signals detection-only; routing outcomes belong in `decision/`
- keep partitions and derived routing bands in `routing.projections`, not back inside `routing.signals`
- keep model choice separate; that belongs in `algorithm/`
- keep route-side behavior separate; that belongs in `plugin/`

## Next Steps

- Read [Projections](../projection/overview) when you need `PROJECTION partition`, weighted score aggregation, or named routing bands.
- Start from [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml) for the exhaustive public contract.
- Use the maintained `balance` assets when you want a realistic repo-native routing strategy:
  - [`deploy/recipes/balance.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/recipes/balance.yaml)
  - [`deploy/recipes/balance.dsl`](https://github.com/vllm-project/semantic-router/blob/main/deploy/recipes/balance.dsl)
