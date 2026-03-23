# Structure Signal

## Overview

`structure` detects request-shape facts such as many explicit questions, ordered workflow markers, or dense constraint phrasing. It maps to `config/signal/structure/` and is declared under `routing.signals.structure`.

This family is heuristic: it stays rule-based, but unlike `keyword` it can count, normalize, and compare typed structural features before emitting a named signal.

## Key Advantages

- Keeps request-shape routing explicit instead of hiding it inside ad hoc keyword lists.
- Lets one detector use counts, densities, or ordered marker sequences without changing the decision DSL.
- Produces named reusable signals that projections and decisions can consume like any other family.
- Preserves the repo-native layering: detector thresholds stay in signals, route policy stays in decisions.

## What Problem Does It Solve?

Some routing choices depend on how a prompt is written, not just what topic it mentions. A prompt with five questions, or one that says "first ... then ...", often needs different routing from a single short ask even when the domain is the same.

`structure` solves that by turning request-shape features into stable named signals.

## When to Use

Use `structure` when:

- routing depends on question count, list shape, or ordered workflow markers
- you need normalized counts or thresholds, not just raw keyword presence
- the detector is still rule-based and does not need a learned model
- you want projections to consume structural facts with `type: structure`

## Configuration

Source fragment family: `config/signal/structure/`

```yaml
routing:
  signals:
    structure:
      - name: many_questions
        description: Prompts with many explicit questions.
        feature:
          type: count
          source:
            type: regex
            pattern: '[?？]'
        predicate:
          gte: 4
      - name: numbered_steps
        description: Prompts that contain numbered list items such as "1. ..."
        feature:
          type: exists
          source:
            type: regex
            pattern: '(?m)^\s*\d+\.\s+'
      - name: first_then_flow
        description: Prompts that express an ordered workflow.
        feature:
          type: sequence
          source:
            type: sequence
            case_sensitive: false
            sequences:
              - ["first", "then"]
              - ["first", "next", "finally"]
              - ["首先", "然后"]
              - ["先", "再"]
      - name: constraint_dense_chars
        description: Constraint language is dense relative to character length.
        feature:
          type: density
          normalize_by: char_count
          source:
            type: keyword_set
            case_sensitive: false
            keywords:
              - under
              - at most
              - at least
              - within
              - no more than
              - 不超过
              - 至少
              - 最多
        predicate:
          gt: 0.02
      - name: format_directive_dense_words
        description: Output-format directives are dense relative to word count.
        feature:
          type: density
          normalize_by: word_count
          source:
            type: keyword_set
            keywords:
              - table
              - bullet
              - json
              - markdown
              - 表格
              - 列表
              - JSON
        predicate:
          gt: 0.08
      - name: low_question_density
        description: Prompts with very low question density.
        feature:
          type: density
          normalize_by: char_count
          source:
            type: regex
            pattern: '[?？]'
        predicate:
          lt: 0.02
      - name: tight_first_then_gap
        description: Start and end workflow markers are close together.
        feature:
          type: span_distance
          normalize_by: token_count
          source:
            type: marker_pair
            case_sensitive: false
            start: ["first", "首先", "先"]
            end: ["then", "然后", "再"]
        predicate:
          lte: 6
```

`feature.type` defines how the value is computed. `feature.source` defines what is being scanned. `predicate` turns that numeric or boolean result into a named matched signal.

Current supported contract:

- `feature.type`: `exists`, `count`, `density`, `sequence`, `span_distance`
- `feature.source.type`: `regex`, `keyword_set`, `sequence`, `marker_pair`
- `feature.normalize_by`: `char_count`, `word_count`, `token_count`
- `predicate`: `gt`, `gte`, `lt`, `lte`

Notes:

- `exists` does not accept a predicate; it emits a match when the source is present.
- `density` requires `feature.normalize_by`.
- `sequence` requires `feature.source.type=sequence`.
- `span_distance` requires `feature.source.type=marker_pair`.
- `regex` is a real regular-expression source in this family.

Example signal meanings:

- `many_questions`: count how many `?` or `？` characters appear, and match when there are at least four.
- `numbered_steps`: match when the prompt already contains numbered list items such as `1. ...`.
- `first_then_flow`: match when ordered workflow markers appear in sequence, such as `first ... then ...` or `先 ... 再 ...`.
- `constraint_dense_chars`: count constraint markers and divide by character length to capture prompts whose requirements are unusually dense.
- `tight_first_then_gap`: measure the distance between start and end workflow markers and match when the transition stays compact.

Use `structure` when routing depends on request form, but you still want the router contract to stay typed and declarative.
