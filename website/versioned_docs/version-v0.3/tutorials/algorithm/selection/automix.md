# AutoMix

## Overview

`automix` is an experimental selection algorithm that returns **one model** for the matched decision, choosing among the decision's candidates with a cost-quality value estimate.

It aligns to `config/fragments/algorithm/selection/automix.yaml`.

**Paper**: [Automatically Mixing Language Models](https://arxiv.org/abs/2310.12963) (NeurIPS 2024)

The selector is inspired by the paper, but its public configuration is intentionally smaller than the research system.

## Key Advantages

- Weighs candidate quality against configured cost instead of always picking the strongest model.
- Keeps the candidate set bounded by the matched decision.
- Keeps the cost-quality policy in router config instead of application code.

## What Problem Does It Solve?

Always choosing the strongest model wastes budget, while always choosing the cheapest model can hurt quality. `automix` scores a bounded candidate set on quality and cost so a route can prefer cheaper models when they are expected to be good enough.

## When to Use

- One route has several candidate models with different cost and quality profiles.
- The route should stay cost-aware instead of always choosing the strongest model.
- You are experimenting and can validate the result on your own traffic.

## Verification and Escalation

`automix` selects one model before the request is sent. It does not generate, verify, and then escalate to a larger model.

For request-time generation, confidence verification (including AutoMix-style entailment verification), and escalation across models, use the [`confidence`](../looper/confidence) algorithm.

## Known Limitations

- This decision algorithm makes a single backend call. Use `confidence` for multi-call escalation and verification.
- In v0.3, the AutoMix selector reads its settings from `global.router.model_selection.automix`, not from the decision. The six fields under `algorithm.automix` are accepted; of these, only a positive `cost_quality_tradeoff` is used, as the decision's cost weight, and only when the global block sets `cost_aware_routing: true`.
- Cost-aware scoring is only as useful as the configured model pricing and quality metadata.
- AutoMix is experimental. Validate it on your own traffic before relying on it for an SLO.

## Configuration

```yaml
algorithm:
  type: automix
  automix:
    verification_threshold: 0.78
    max_escalations: 2
    cost_aware_routing: true
    cost_quality_tradeoff: 0.35
    discount_factor: 0.9
    use_logprob_verification: true
```

The values above are an example, not defaults.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `verification_threshold` | float | Verification confidence threshold (0–1) in the AutoMix value estimate; read from the global block |
| `max_escalations` | int | Accepted for compatibility; this algorithm makes a single backend call |
| `cost_aware_routing` | bool | Switches the AutoMix value estimate to cost-aware scoring; read from the global block |
| `cost_quality_tradeoff` | float | Cost weight (0–1); at the decision level only a positive value is used (see Known Limitations) |
| `discount_factor` | float | Discount factor γ (0–1) in the AutoMix value estimate; read from the global block |
| `use_logprob_verification` | bool | Accepted for compatibility; does not enable or change verification |

Only the six fields above belong to the v0.3 decision-level AutoMix public configuration contract. Other AutoMix research settings are not supported under `algorithm.automix`.
