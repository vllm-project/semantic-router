# Troubleshooting & FAQ

Common issues and frequently asked questions about model selection.

## Frequently Asked Questions

### Which algorithm should I start with?

**Start with Static selection** if you're new to model selection. It's deterministic and easy to debug. Once you understand your traffic patterns, migrate to adaptive algorithms.

### Do I need to configure all algorithms?

No. Configure only the algorithm you're using. Each algorithm has sensible defaults, so you only need to specify fields you want to customize.

### Can I switch algorithms without downtime?

Yes. Algorithm changes take effect on configuration reload. In-flight requests complete with the previous algorithm.

## Common Issues

### Elo Selection

**Issue: Ratings not changing**

Possible causes:

1. Feedback not being submitted - verify POST requests to `/api/v1/feedback` return 200
2. K-factor too low - increase from 32 to 64 for faster adaptation
3. Not enough traffic - Elo needs consistent feedback volume

```bash
# Verify feedback endpoint is working
curl -X POST http://localhost:8080/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"request_id": "test", "model": "gpt-4", "rating": 1}'
```

**Issue: One model always selected**

This is expected if one model has significantly higher Elo rating. Options:

- Reset ratings by deleting `storage_path` file
- Increase `k_factor` to allow faster rating changes
- Use `decay_factor` to reduce weight of old comparisons

### RouterDC Selection

**Issue: Wrong model selected for queries**

1. Check model descriptions are specific enough:

```yaml
# Bad - too generic
description: "A good AI model"

# Good - specific capabilities
description: "Mathematical reasoning, theorem proving, step-by-step solutions"
```

2. Verify embeddings are being computed:

```bash
# Check metrics for embedding latency
curl http://localhost:8080/metrics | grep embedding
```

**Issue: Startup failure with "missing descriptions"**

If `require_descriptions: true`, all models must have descriptions:

```yaml
models:
  - name: gpt-4
    description: "Required when require_descriptions is true"
```

### AutoMix Selection

**Issue: Always selecting expensive models**

Your `cost_quality_tradeoff` is too low (favoring quality). Increase it:

```yaml
automix:
  cost_quality_tradeoff: 0.5  # Balance cost and quality
```

**Issue: Always selecting cheap models**

Your `cost_quality_tradeoff` is too high. Decrease it:

```yaml
automix:
  cost_quality_tradeoff: 0.2  # Favor quality
```

**Issue: Missing pricing data**

AutoMix requires pricing information:

```yaml
models:
  - name: gpt-4
    pricing:
      input_cost_per_1k: 0.03
      output_cost_per_1k: 0.06
```

### Hybrid Selection

**Issue: Weights validation error**

Weights must sum to 1.0 (±0.01 tolerance):

```yaml
hybrid:
  elo_weight: 0.3
  router_dc_weight: 0.3
  automix_weight: 0.2
  cost_weight: 0.2
  # Total: 1.0 ✓
```

**Issue: Component not contributing**

Ensure the component has required data:

- Elo: needs feedback history
- RouterDC: needs model descriptions
- AutoMix: needs pricing data

## Debugging Tips

### Enable verbose logging

```yaml
logging:
  level: debug
```

### Check selection metrics

```bash
curl http://localhost:8080/metrics | grep selection
```

Key metrics:

- `model_selection_duration_seconds` - selection latency
- `model_selection_total` - selection counts by algorithm
- `model_elo_rating` - current Elo ratings (if using Elo)

### Trace individual requests

Add request ID header and check logs:

```bash
curl -H "X-Request-ID: debug-123" http://localhost:8080/v1/chat/completions ...
```

Then search logs:

```bash
vllm-sr logs router | grep debug-123
```

## Getting Help

If you're still stuck:

1. Check [GitHub Issues](https://github.com/vllm-project/semantic-router/issues) for similar problems
2. Enable debug logging and capture relevant output
3. Open a new issue with:
   - Configuration (redact secrets)
   - Steps to reproduce
   - Expected vs actual behavior
   - Relevant log output
