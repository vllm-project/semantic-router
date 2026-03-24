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
      - name: at_most_one_question
        description: Prompts with one or fewer explicit questions.
        feature:
          type: count
          source:
            type: regex
            pattern: '[?？]'
        predicate:
          lte: 1
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
      - name: constraint_dense
        description: Constraint language is dense relative to multilingual text units.
        feature:
          type: density
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
          gt: 0.08
      - name: format_directive_dense
        description: Output-format directives are dense relative to multilingual text units.
        feature:
          type: density
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
        description: Prompts with very low question density relative to multilingual text units.
        feature:
          type: density
          source:
            type: regex
            pattern: '[?？]'
        predicate:
          lt: 0.05
```

`feature.type` defines how the value is computed. `feature.source` defines what is being scanned. `predicate` turns that numeric or boolean result into a named matched signal.

Current supported contract:

- `feature.type`: `exists`, `count`, `density`, `sequence`
- `feature.source.type`: `regex`, `keyword_set`, `sequence`
- `predicate`: `gt`, `gte`, `lt`, `lte`

Notes:

- `exists` does not accept a predicate; it emits a match when the source is present.
- `density` automatically normalizes by multilingual text units. CJK characters count individually, contiguous runs of non-CJK letters/digits count as one unit, and punctuation is ignored.
- `sequence` requires `feature.source.type=sequence`.
- `keyword_set` uses script-aware matching so continuous CJK text and mixed-script prompts still register expected hits.
- `regex` is a real regular-expression source in this family.

Example signal meanings:

- `many_questions`: count how many `?` or `？` characters appear, and match when there are at least four.
- `at_most_one_question`: count how many `?` or `？` characters appear, and match when there are zero or one.
- `numbered_steps`: match when the prompt already contains numbered list items such as `1. ...`.
- `first_then_flow`: match when ordered workflow markers appear in sequence, such as `first ... then ...` or `先 ... 再 ...`.
- `constraint_dense`: count constraint markers and divide by multilingual text units to capture prompts whose requirements are unusually dense across English, Chinese, and mixed-script prompts.

Use `structure` when routing depends on request form, but you still want the router contract to stay typed and declarative.
