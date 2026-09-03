# Confidence

## Overview

`confidence` is a **looper** algorithm that escalates across candidate models until confidence is high enough. It tries smaller/cheaper models first and only escalates to larger models when the response confidence is below a configured threshold.

It aligns to `config/fragments/algorithm/looper/confidence.yaml`.

## Key Advantages

- Supports small-to-large escalation instead of a fixed winner.
- Makes stopping conditions explicit and configurable.
- Multiple confidence evaluation methods: `avg_logprob`, `margin`, `hybrid`, `self_verify`, `automix_entailment`.
- Lets one route trade extra latency for higher confidence only when needed.

## Algorithm Principle

The confidence algorithm evaluates model responses using either token-level logprobs or external verification:

1. **Generate**: Call the current model (starting with the smallest).
2. **Evaluate Confidence**:
   - `avg_logprob`: Average log probability across all output tokens. Higher (closer to 0) = more confident.
   - `margin`: Average margin between top-1 and top-2 logprobs per token. Higher = more confident.
   - `hybrid`: Weighted combination of both methods.
   - `self_verify`: Prompt the same model to grade its own answer (returns a JSON `{confidence, reason}`).
   - `automix_entailment`: Delegate verification to an external few-shot entailment server, per arXiv:2310.12963 §3.2. Confidence is `verified_samples / total_samples`.
3. **Decide**:
   - Confidence >= threshold → return response.
   - Confidence < threshold → escalate to next model.
   - On error → skip or fail (configurable).

## Execution Flow

```mermaid
flowchart TD
    A[Request arrives] --> B[Decision matched]
    B --> C[algorithm.type = confidence]
    C --> D[Sort modelRefs by size, small → large]
    D --> E[Call current model]
    E --> F{Model succeeded?}
    F -- No --> G{on_error = skip?}
    G -- Yes --> H[Move to next model]
    G -- No --> I[Return error]
    H --> E
    F -- Yes --> J[Compute confidence score]
    J --> K{confidence >= threshold?}
    K -- Yes --> L[Return response]
    K -- No --> M{More models available?}
    M -- Yes --> H
    M -- No --> N[Return last response with warning]
```

## What Problem Does It Solve?

Some routes should try cheaper candidates first and only pay for escalation when the current answer is not confident enough. `confidence` makes that sequential escalate-on-low-confidence policy explicit in router config instead of burying it in application code.

## When to Use

- A route should escalate across several candidate models.
- Confidence should decide whether to continue to the next model.
- The route should stop as soon as one response is good enough.
- You want to minimize cost by trying cheaper models first.

## Known Limitations

- Each escalation adds latency (sequential model calls).
- Confidence thresholds may need tuning per route type.
- Logprob-based confidence may not always correlate with factual correctness.
- `hybrid` method requires tuning `hybrid_weights` for optimal performance.
- `automix_entailment` requires running a separate verification server (see [`automix_verifier.py`](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_selection/rl_model_selection/automix_verifier.py)) and adds one HTTP round-trip per model call.

## Configuration

```yaml
algorithm:
  type: confidence
  confidence:
    confidence_method: hybrid        # avg_logprob, margin, hybrid, self_verify, automix_entailment
    threshold: 0.72                  # Normalized escalation threshold
    escalation_order: small_to_large # size, small_to_large, declared, cost, or automix
    cost_quality_tradeoff: 0.3       # Cost vs quality balance in (0, 1]
    token_filter: tool_call_args     # all or tool_call_args
    on_error: skip                   # skip or fail
    hybrid_weights:
      logprob_weight: 0.5            # Weight for avg_logprob in hybrid
      margin_weight: 0.5             # Weight for margin in hybrid
    # Required when confidence_method = automix_entailment
    verifier_server_url: ""          # AutoMix entailment verifier HTTP URL
    verifier_timeout_seconds: 0      # 0 = default (60s)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confidence_method` | string | `avg_logprob` | Evaluation method: `avg_logprob`, `margin`, `hybrid`, `self_verify`, or `automix_entailment` |
| `threshold` | float | method-dependent | Configured nonzero thresholds are normalized values in `(0, 1]`. `0` is indistinguishable from omission and selects the method default. |
| `escalation_order` | string | `size` | One of `size`, `small_to_large`, `declared`, `cost`, or `automix`. |
| `cost_quality_tradeoff` | float | `0.3` | Cost vs. quality balance in `(0, 1]`. `0` is the unset sentinel and therefore also selects `0.3`. |
| `token_filter` | string | `all` | `all` uses every generated token; `tool_call_args` excludes structural tool-call JSON where possible. |
| `on_error` | string | `skip` | Behavior on model call failure: `skip` or `fail` |
| `hybrid_weights.logprob_weight` | float | `0.5` | Weight for avg_logprob in hybrid mode. Zero is the unset sentinel; the two effective weights must sum to `1`. |
| `hybrid_weights.margin_weight` | float | `0.5` | Weight for margin in hybrid mode. Zero is the unset sentinel; the two effective weights must sum to `1`. |
| `verifier_server_url` | string | — | Required only when `confidence_method = automix_entailment`. Must be an absolute HTTP(S) URL without credentials, query, or fragment (see [`automix_verifier.py`](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_selection/rl_model_selection/automix_verifier.py)). |
| `verifier_timeout_seconds` | int | `60` | Positive HTTP timeout for `automix_entailment`; `0` is the unset sentinel and selects 60 seconds. |

The method defaults used when `threshold` is omitted (or explicitly `0`) are
`-1` for `avg_logprob` (the permissive evidence-present default), `0.5` for
`margin` and `hybrid`, and `0.7` for `self_verify` and
`automix_entailment`. An explicitly configured threshold is always normalized
to `(0, 1]`; negative configured thresholds are rejected.

### `self_verify` vs `automix_entailment`

Both implement the AutoMix paper's cascade idea but differ in how the verification signal is produced:

| Aspect | `self_verify` | `automix_entailment` |
|---|---|---|
| Verifier | The same generation model | A separate few-shot entailment model on its own HTTP server |
| Per-request cost | 1 extra prompt to the generation model | 1 HTTP round-trip; `k` sampled completions in the verifier |
| Faithfulness to arXiv:2310.12963 | Loose (prompt-graded JSON) | Strict (paper §3.2 entailment) |
| Extra infra | None | Requires running [`automix_verifier.py`](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_selection/rl_model_selection/automix_verifier.py) |
| When to pick | Single-deployment setups; no extra server | Production routes where verifier model can be smaller/specialized |
