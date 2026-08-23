# AutoMix

## Overview

`automix` is an experimental selector that ranks candidate models by configured
quality and cost plus internal verification and escalation estimates. It
returns one model; it is not the sequential
[`confidence`](../looper/confidence) Looper.

This selector is inspired by
[Automatically Mixing Language Models](https://arxiv.org/abs/2310.12963), but
the public configuration is intentionally smaller than the research system.

## Key Advantages

- Balances configured quality metadata with configured cost.
- Keeps the candidate set bounded by the matched decision.
- Keeps the experimental value calculation separate from route eligibility.

## What Problem Does It Solve?

Always choosing the strongest model wastes budget, while always choosing the
cheapest model can hurt quality. AutoMix computes a cost-quality value from
configured metadata and internal estimates for a bounded candidate set.

## When to Use

Use AutoMix for experiments where candidate pricing and quality metadata are
available. Prefer `static`, `router_dc`, or `multi_factor` when you need a
supported, stateless policy with easier operational reasoning.

## Configuration

```yaml
algorithm:
  type: automix
  automix:
    verification_threshold: 0.78
    max_escalations: 2
    cost_aware_routing: true
    cost_quality_tradeoff: 0.3
    discount_factor: 0.95
    use_logprob_verification: true
```

Only the fields above are part of the current decision-level AutoMix contract.
`max_escalations` and `use_logprob_verification` are accepted for compatibility
but do not affect AutoMix selection.
See a complete example:
[`config/fragments/algorithm/selection/automix.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/automix.yaml).

## Dependencies and Limitations

- Candidate prices come from `models[].pricing`; missing metadata
  reduces the usefulness of cost-aware scoring.
- Capability estimates start from configured model metadata and defaults.
  AutoMix does not learn from the public outcome endpoint; retune estimates
  explicitly when traffic or models change.
- `verification_threshold`, configured costs, `cost_quality_tradeoff`, and
  `discount_factor` affect the one-model score. `max_escalations` and
  `use_logprob_verification` currently do not.
- This decision algorithm does not itself make multiple backend calls. Use
  `confidence` for request-time generation and escalation.
- Request content is embedded through the configured semantic embedding path.
- AutoMix is experimental. Validate it on your own traffic before relying on it
  for an SLO.
