# Model Selection Metrics

This document describes the Prometheus metrics exposed by the model selection package for observability and traceability of model selection evolution over time.

## Overview

The model selection metrics enable:

- **Explainability**: Understand why certain models are selected
- **Traceability**: Track the evolution of model ratings over time
- **Monitoring**: Alert on anomalous selection patterns
- **Optimization**: Identify opportunities to improve routing

## Metrics Reference

### Elo Rating Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_model_elo_rating` | Gauge | `model`, `category` | Current Elo rating for each model. Higher values indicate better performance based on feedback. |
| `llm_model_rating_change` | Histogram | `model`, `category`, `feedback_type` | Distribution of rating changes during feedback updates. Buckets from -32 to +32 cover typical K-factor changes. |
| `llm_model_comparisons_total` | Gauge | `model`, `category` | Total number of comparisons a model has participated in. |
| `llm_model_win_rate` | Gauge | `model`, `category` | Win rate for each model (wins / total comparisons). |

### Selection Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_model_selection_total` | Counter | `method`, `model`, `decision` | Total number of model selections by algorithm and selected model. |
| `llm_model_selection_history` | Counter | `method`, `decision` | Selection count over time by algorithm type for trend analysis. |
| `llm_model_selection_duration_seconds` | Histogram | `method` | Duration of model selection operations. |
| `llm_model_selection_score` | Histogram | `method`, `model` | Score of selected models (normalized 0-1). |
| `llm_model_selection_confidence` | Histogram | `method` | Confidence score distribution of model selections. |

### Feedback Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_model_feedback_total` | Counter | `winner`, `loser`, `is_tie`, `category` | Total feedback events by model pair. |
| `llm_model_selection_component_agreement` | Histogram | (none) | Agreement ratio between hybrid selector components (1.0 = all agree). |

### AutoMix-Specific Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_model_automix_verification_prob` | Gauge | `model` | Learned verification probability per model (evolves with feedback). Based on arXiv:2310.12963. |
| `llm_model_automix_quality` | Gauge | `model` | Learned average quality score per model (evolves with feedback). |
| `llm_model_automix_success_rate` | Gauge | `model` | Query success rate per model (success_count / total_count). |

### RouterDC-Specific Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_model_routerdc_similarity` | Histogram | `model` | Distribution of query-model similarity scores. Based on arXiv:2409.19886. |
| `llm_model_routerdc_affinity` | Gauge | `model` | Learned query-model affinity from feedback (evolves over time). |

## Grafana Dashboard

A pre-built Grafana dashboard is available at:

- `deploy/docker-compose/addons/model-selection-dashboard.json`

### Dashboard Panels

1. **Elo Rating Over Time**: Line chart showing rating evolution for all models
2. **Current Elo Ratings**: Bar chart for quick comparison of current ratings
3. **Model Selection Rate by Algorithm**: Stacked bar chart of selection patterns
4. **Selection Confidence Distribution**: Percentile view of confidence scores
5. **Feedback Events Rate**: Wins/losses/ties over time
6. **Rating Change Distribution**: Shows volatility of rating updates
7. **Model Win Rates**: Gauge showing win rate per model
8. **Model Comparisons Over Time**: Comparison activity over time
9. **Selection Algorithm Usage**: Pie chart of algorithm distribution

### Dashboard Variables

- `$category`: Filter by decision category (e.g., "tech", "finance", "_global")
- `$method`: Filter by selection algorithm (e.g., "elo", "router_dc", "hybrid")

## Example Queries

### Track Elo Rating Evolution

```promql
# Current Elo ratings for all models in a category
llm_model_elo_rating{category="tech"}

# Rate of rating change over time
rate(llm_model_elo_rating{category="tech"}[5m])
```

### Analyze Selection Patterns

```promql
# Selection rate by model and algorithm
sum by(model, method) (rate(llm_model_selection_total[5m]))

# Most frequently selected model
topk(5, sum by(model) (increase(llm_model_selection_total[1h])))
```

### Monitor Feedback Quality

```promql
# Feedback rate by winner
sum by(winner) (rate(llm_model_feedback_total[5m]))

# Tie ratio (may indicate unclear preferences)
sum(rate(llm_model_feedback_total{is_tie="true"}[5m])) / sum(rate(llm_model_feedback_total[5m]))
```

### Confidence Analysis

```promql
# Median selection confidence by algorithm
histogram_quantile(0.5, sum by(le, method) (rate(llm_model_selection_confidence_bucket[5m])))

# Low confidence selections (potential escalation candidates)
histogram_quantile(0.1, sum by(le) (rate(llm_model_selection_confidence_bucket[5m])))
```

## Interpreting the Metrics

### Elo Rating Interpretation

- **1500**: Default/initial rating for new models
- **1400-1600**: Average performance range
- **>1600**: Above average, preferred by users
- **<1400**: Below average, may need investigation

### Rating Change Patterns

- **Large positive changes** (+16 to +32): Strong user preference, upset wins
- **Small changes** (-4 to +4): Expected outcomes, stable ratings
- **Large negative changes** (-16 to -32): Strong negative feedback, upset losses

### Win Rate Interpretation

- **>0.6**: Model consistently outperforms alternatives
- **0.4-0.6**: Competitive with other models
- **<0.4**: Underperforming, consider removing or improving

### Confidence Scores

- **>0.8**: High confidence, stable selection
- **0.5-0.8**: Moderate confidence, reasonable selection
- **<0.5**: Low confidence, may need more feedback data

## Alerts (Example)

```yaml
groups:
  - name: model-selection
    rules:
      - alert: LowModelConfidence
        expr: histogram_quantile(0.5, sum by(le, method) (rate(llm_model_selection_confidence_bucket[5m]))) < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: Model selection confidence is low
          description: "Median selection confidence for {{ $labels.method }} is {{ $value }}"

      - alert: EloRatingDrift
        expr: abs(deriv(llm_model_elo_rating[1h])) > 50
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: Rapid Elo rating change detected
          description: "Model {{ $labels.model }} in {{ $labels.category }} is experiencing rapid rating changes"
```

## Related Documentation

- [RouteLLM Paper (Elo)](https://arxiv.org/abs/2406.18665) - Bradley-Terry model for LLM routing
- [RouterDC](https://arxiv.org/abs/2409.19886) - Dual-contrastive query-to-model matching
- [AutoMix](https://arxiv.org/abs/2310.12963) - POMDP-based cost-quality optimization
