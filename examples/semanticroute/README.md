# SemanticRoute Examples

This directory contains various examples of SemanticRoute configurations demonstrating different routing scenarios and capabilities.

## Examples Overview

### 1. Simple Intent Routing (`simple-intent-routing.yaml`)

A basic example showing intent-based routing for math and computer science queries.

**Features:**

- Simple intent matching with categories
- Single model reference with fallback
- Minimal configuration

**Use Case:** Basic routing based on query categories without complex filtering.

### 2. Complex Filter Chain (`complex-filter-chain.yaml`)

Demonstrates a comprehensive filter chain with multiple security and performance filters.

**Features:**

- PII detection with custom allowed types
- Prompt guard with custom security rules
- Semantic caching for performance
- Reasoning control configuration

**Use Case:** Production environments requiring security, privacy, and performance optimizations.

### 3. Multiple Routes (`multiple-routes.yaml`)

Shows how to define multiple routing rules within a single SemanticRoute resource.

**Features:**

- Separate rules for technical vs. creative queries
- Different reasoning configurations per rule
- Rule-specific caching strategies

**Use Case:** Applications serving diverse query types with different processing requirements.

### 4. Weighted Routing (`weighted-routing.yaml`)

Demonstrates traffic distribution across multiple model endpoints using weights and priorities.

**Features:**

- Traffic splitting (80/20) between models
- Priority-based failover
- Load balancing configuration

**Use Case:** A/B testing, gradual rollouts, or load distribution across model endpoints.

### 5. Tool Selection Example (`tool-selection-example.yaml`)

Demonstrates automatic tool selection based on semantic similarity to user queries.

**Features:**

- Automatic tool selection with configurable similarity threshold
- Tool filtering by categories and tags
- Fallback behavior configuration
- Integration with semantic caching and reasoning control

**Use Case:** Applications requiring dynamic tool selection based on user intent and query content.

### 6. Comprehensive Example (`comprehensive-example.yaml`)

A production-ready configuration showcasing all SemanticRoute features.

**Features:**

- Multiple rules with different configurations
- Advanced filtering with custom rules
- External cache backend (Redis)
- High-availability model setup
- Comprehensive security policies

**Use Case:** Enterprise production deployments requiring full feature utilization.

## Deployment Instructions

### Prerequisites

1. Kubernetes cluster with SemanticRoute CRD installed:

   ```bash
   kubectl apply -f ../../deploy/kubernetes/crds/vllm.ai_semanticroutes.yaml
   ```

2. Ensure your model endpoints are accessible from the cluster.

### Deploy Examples

1. **Deploy a single example:**

   ```bash
   kubectl apply -f simple-intent-routing.yaml
   ```

2. **Deploy all examples:**

   ```bash
   kubectl apply -f .
   ```

3. **Verify deployment:**

   ```bash
   kubectl get semanticroutes
   kubectl describe semanticroute reasoning-route
   ```

## Configuration Reference

### Intent Configuration

```yaml
intents:
- category: "math"                    # Required: Intent category name
  description: "Mathematics queries"  # Optional: Human-readable description
  threshold: 0.7                     # Optional: Confidence threshold (0.0-1.0)
```

### Model Reference Configuration

```yaml
modelRefs:
- modelName: "gpt-oss"              # Required: Model identifier
  address: "127.0.0.1"              # Required: Endpoint address
  port: 8080                        # Required: Endpoint port
  weight: 80                        # Optional: Traffic weight (0-100)
  priority: 100                     # Optional: Priority for failover
```

### Filter Configuration

Each filter type has specific configuration options:

- **PIIDetection**: Controls PII detection and handling
- **PromptGuard**: Provides security and jailbreak protection
- **SemanticCache**: Enables response caching for performance
- **ReasoningControl**: Manages reasoning mode behavior
- **ToolSelection**: Enables automatic tool selection based on semantic similarity

## Best Practices

1. **Start Simple**: Begin with basic intent routing and add filters as needed.

2. **Test Thoroughly**: Validate routing behavior with representative queries.

3. **Monitor Performance**: Use appropriate cache settings and monitor hit rates.

4. **Security First**: Enable PII detection and prompt guard in production.

5. **Gradual Rollout**: Use weighted routing for safe model deployments.

## Troubleshooting

### Common Issues

1. **Route Not Matching**: Check intent categories and thresholds.
2. **Model Unreachable**: Verify endpoint addresses and network connectivity.
3. **Filter Errors**: Validate filter configurations against the schema.

### Debugging Commands

```bash
# Check SemanticRoute status
kubectl get sr -o wide

# View detailed configuration
kubectl describe semanticroute <name>

# Check logs (if controller is deployed)
kubectl logs -l app=semantic-router-controller
```
