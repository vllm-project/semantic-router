# Ratings

## Overview

`ratings` calls every candidate model and returns one OpenAI-compatible choice
per successful model. `max_concurrent` limits parallel work; it does not limit
the total number of candidates executed.

Despite its name, the current runtime does not score, vote on, or synthesize
the choices. The caller receives them for comparison or downstream rating.

## Key Advantages

- Compares the same request across all declared candidates.
- Bounds parallel work without dropping later candidates.
- Preserves one identifiable response choice per successful model.

## What Problem Does It Solve?

Evaluation and comparison clients sometimes need the same prompt answered by
several models through one Router request. Ratings provides bounded fan-out
without introducing a judge model.

## When to Use

Use Ratings for side-by-side evaluation or applications that understand
multiple `choices`. Do not use it when the caller expects a single synthesized
answer; use `fusion` or `remom` for that.

## Configuration

```yaml
algorithm:
  type: ratings
  ratings:
    max_concurrent: 3
    on_error: skip
```

The maintained fragment is
[`config/fragments/algorithm/looper/ratings.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/looper/ratings.yaml).

## Dependencies and Limitations

- Requires more than one `modelRef` and a reachable
  `global.integrations.looper.endpoint`.
- Every candidate receives the request content, so all candidate providers must
  be allowed by the route's data policy.
- Cost grows with the number of candidates. Concurrency reduces wall-clock
  time but not total model calls.
- `on_error: skip` returns the successful choices; `on_error: fail` fails the
  run if any model call fails. The run fails if all models fail.
- Tool definitions are removed from Ratings subrequests; use Router Flow for
  agent workflows that must continue tool calls.
