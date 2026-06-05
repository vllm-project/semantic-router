# Choosing a Selection Algorithm

This guide helps you select the right model selection algorithm for your use case.

## Quick Decision Tree

```
Do you need deterministic routing?
├── Yes → Static Selection
└── No → Continue...

Do you have user feedback available?
├── Yes, abundant → Elo Rating
├── Some feedback → Hybrid (with Elo component)
└── No feedback → Continue...

Do you have good model descriptions?
├── Yes → RouterDC
└── No → Continue...

Is cost optimization important?
├── Yes → AutoMix
└── No → Static or Hybrid
```

## Algorithm Comparison

### Core Algorithms

| Algorithm | Feedback Needed | Setup Complexity | Adaptability | Cost Optimization |
|-----------|-----------------|------------------|--------------|-------------------|
| Static | None | Low | None | Manual |
| Elo | High | Medium | High | Indirect |
| RouterDC | None | Medium | Medium | No |
| AutoMix | Low | Medium | Medium | Yes |
| Hybrid | Varies | High | High | Yes |

### RL-Driven Algorithms

| Algorithm | Feedback Needed | Setup Complexity | Adaptability | Personalization |
|-----------|-----------------|------------------|--------------|-----------------|
| Thompson | Medium | Low | High | Per-user option |
| GMTRouter | Medium | High | Very High | Built-in |
| Router-R1 | None | High | High | Via LLM |

## Use Case Recommendations

### Startup / MVP
**Recommended: Static**

Start simple with explicit rules. Migrate to adaptive methods as you collect data.

```yaml
algorithm:
  type: static
  static:
    default_model: gpt-3.5-turbo
```

### High-Volume Production
**Recommended: AutoMix or Hybrid**

Optimize costs while maintaining quality at scale.

```yaml
algorithm:
  type: automix
  automix:
    cost_quality_tradeoff: 0.4
```

### User-Facing Applications
**Recommended: Elo**

Let user feedback drive model selection for subjective quality.

```yaml
algorithm:
  type: elo
  elo:
    k_factor: 32
    storage_path: /data/elo-ratings.json
```

### Specialized Domains
**Recommended: RouterDC**

When models have distinct specializations, match queries to model capabilities.

```yaml
algorithm:
  type: router_dc
  router_dc:
    require_descriptions: true
```

### Enterprise / Multi-objective
**Recommended: Hybrid**

Balance multiple factors: quality, cost, user satisfaction, and specialization.

```yaml
algorithm:
  type: hybrid
  hybrid:
    elo_weight: 0.3
    router_dc_weight: 0.3
    automix_weight: 0.2
    cost_weight: 0.2
```

### Personalized Multi-User Platforms
**Recommended: Thompson Sampling or GMTRouter**

Learn individual user preferences over time.

```yaml
algorithm:
  type: thompson
  thompson:
    per_user: true
    min_samples: 10
```

### Research / Complex Routing Logic
**Recommended: Router-R1**

When routing decisions require semantic understanding that's hard to encode in rules.

```yaml
algorithm:
  type: router_r1
  router_r1:
    router_endpoint: http://localhost:8001
    use_cot: true
```

## Migration Path

A typical progression as your system matures:

1. **Start**: Static selection with simple rules
2. **Add feedback**: Migrate to Elo as you collect user feedback
3. **Add descriptions**: Add RouterDC for query-model matching
4. **Optimize cost**: Incorporate AutoMix for cost efficiency
5. **Combine**: Use Hybrid to leverage all methods

## Key Considerations

### Data Requirements

- **Static**: No data needed
- **Elo**: Needs consistent user feedback (thumbs up/down)
- **RouterDC**: Needs quality model descriptions
- **AutoMix**: Needs accurate pricing and quality scores
- **Hybrid**: Combination of above
- **Thompson**: Needs feedback; works online
- **GMTRouter**: Benefits from interaction history; can pre-train
- **Router-R1**: Needs router LLM server; model descriptions help

### Latency Impact

| Algorithm | Typical Latency |
|-----------|-----------------|
| Static | Under 1ms |
| Elo | Under 2ms |
| RouterDC | 2-5ms (embedding) |
| AutoMix | Under 3ms |
| Hybrid | 3-5ms |
| Thompson | Under 2ms |
| GMTRouter | 5-15ms (GNN) |
| Router-R1 | 100-500ms (LLM) |

### Maintenance

- **Static**: Manual rule updates
- **Elo**: Self-maintaining with feedback
- **RouterDC**: Update descriptions when models change
- **AutoMix**: Update pricing when costs change
- **Hybrid**: Periodic weight tuning recommended
