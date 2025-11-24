# Ensemble Orchestration Configuration

This directory contains configuration examples for the ensemble orchestration feature, which enables parallel model inference with configurable aggregation strategies.

## Overview

The ensemble orchestration feature allows you to:
- Query multiple LLM models in parallel
- Combine their outputs using various aggregation strategies
- Improve reliability, accuracy, and cost-performance trade-offs

## Configuration

### Basic Setup

Enable ensemble mode in your `config.yaml`:

```yaml
ensemble:
  enabled: true
  default_strategy: "voting"
  default_min_responses: 2
  timeout_seconds: 30
  max_concurrent_requests: 10
  endpoint_mappings:
    model-a: "http://localhost:8001/v1/chat/completions"
    model-b: "http://localhost:8002/v1/chat/completions"
    model-c: "http://localhost:8003/v1/chat/completions"
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable/disable ensemble orchestration |
| `default_strategy` | string | `"voting"` | Default aggregation strategy |
| `default_min_responses` | integer | `2` | Minimum successful responses required |
| `timeout_seconds` | integer | `30` | Maximum time to wait for responses |
| `max_concurrent_requests` | integer | `10` | Limit on parallel model queries |
| `endpoint_mappings` | map | `{}` | Model name to OpenAI-compatible API endpoint mapping |

## Usage

### Request Headers

Control ensemble behavior using HTTP headers:

| Header | Description | Example |
|--------|-------------|---------|
| `x-ensemble-enable` | Enable ensemble mode | `true` |
| `x-ensemble-models` | Comma-separated list of models | `model-a,model-b,model-c` |
| `x-ensemble-strategy` | Aggregation strategy | `voting` |
| `x-ensemble-min-responses` | Minimum responses required | `2` |

### Example Request

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-ensemble-enable: true" \
  -H "x-ensemble-models: model-a,model-b,model-c" \
  -H "x-ensemble-strategy: voting" \
  -H "x-ensemble-min-responses: 2" \
  -d '{
    "model": "ensemble",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

### Response Headers

The response includes metadata about the ensemble process:

| Header | Description | Example |
|--------|-------------|---------|
| `x-vsr-ensemble-used` | Indicates ensemble was used | `true` |
| `x-vsr-ensemble-models-queried` | Number of models queried | `3` |
| `x-vsr-ensemble-responses-received` | Number of successful responses | `3` |

## Aggregation Strategies

### 1. Voting (Majority Consensus)
**Best for:** Classification, multiple choice, yes/no questions

Selects the most common response among all models.

```bash
-H "x-ensemble-strategy: voting"
```

### 2. Weighted Consensus
**Best for:** Combining models with different reliability profiles

Weights responses by confidence scores from each model.

```bash
-H "x-ensemble-strategy: weighted"
```

### 3. First Success
**Best for:** Latency-sensitive applications

Returns the first valid response received, optimizing for speed.

```bash
-H "x-ensemble-strategy: first_success"
```

### 4. Score Averaging
**Best for:** Numerical outputs, probability distributions

Averages numerical scores across all models.

```bash
-H "x-ensemble-strategy: score_averaging"
```

### 5. Reranking
**Best for:** Generation tasks, open-ended responses

Collects multiple candidate responses and selects the best one (requires additional ranking logic).

```bash
-H "x-ensemble-strategy: reranking"
```

## Use Cases

### Critical Applications
- Medical diagnosis assistance (consensus increases confidence)
- Legal document analysis (high accuracy verification)
- Financial advisory systems (reliability impacts business outcomes)

### Cost Optimization
- Query multiple smaller models instead of one large expensive model
- Start with fast/cheap models, escalate for uncertain cases
- Adaptive routing based on query complexity

### Reliability & Accuracy
- Voting mechanisms to reduce hallucinations
- Consensus-based outputs for higher confidence
- Graceful degradation with fallback chains

### Model Diversity
- Combine different model architectures (GPT-style + Llama-style)
- Ensemble different model sizes for balanced performance
- Cross-validate responses from models with different training data

## Examples

See `ensemble-example.yaml` for a complete configuration example.

## Security Considerations

- Ensure all endpoint URLs are from trusted sources
- Use TLS/HTTPS for production deployments
- Set appropriate timeout values to prevent resource exhaustion
- Monitor and log ensemble operations for debugging

## Performance Tips

1. **Optimize Concurrency**: Set `max_concurrent_requests` based on your infrastructure capacity
2. **Tune Timeouts**: Balance between latency and completeness with `timeout_seconds`
3. **Select Appropriate Strategy**: Choose the strategy that best matches your use case
4. **Monitor Metrics**: Track response times and success rates per model

## Troubleshooting

### No responses received
- Verify endpoint URLs are correct and reachable
- Check network connectivity to model endpoints
- Ensure models are running and accepting requests

### Insufficient responses error
- Reduce `x-ensemble-min-responses` header value
- Add more model endpoints to `endpoint_mappings`
- Check model health and availability

### Slow responses
- Reduce `timeout_seconds` for faster failures
- Increase `max_concurrent_requests` for better parallelism
- Use `first_success` strategy for latency optimization

## Related Documentation

- [Main Configuration Guide](../README.md)
- [API Documentation](../../docs/api.md)
- [Deployment Guide](../../docs/deployment.md)
