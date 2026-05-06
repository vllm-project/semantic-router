# Random

## Overview

`random` emits a random integer from 0 to 9 for each evaluated request. It maps to `config/signal/random/` and is declared under `routing.signals.random`.

## Key Advantages

- Adds a zero-dependency sampling input.
- Exposes the generated digit as a raw signal value.
- Works with normal decisions and projection scores.

## What Problem Does It Solve?

Some routing experiments need a cheap per-request bucket without inspecting the prompt or calling a classifier. `random` provides that bucket as a named signal.

## When to Use

Use it when you need a lightweight random value for experiments, sampling, or projection scores without depending on request text or model inference.

## Configuration

Source fragment family: `config/signal/random/`

```yaml
routing:
  signals:
    random:
      - name: random_digit
```

The emitted value is available as the raw signal value for `random:random_digit`. Decisions can reference the signal by name, and projections can read the digit with `value_source: raw`.

```yaml
routing:
  projections:
    scores:
      - name: random_bucket
        method: weighted_sum
        inputs:
          - type: random
            name: random_digit
            weight: 1.0
            value_source: raw
```
