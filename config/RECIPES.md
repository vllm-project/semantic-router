# Configuration Recipes

This directory contains versioned, curated configuration presets ("recipes") optimized for different objectives. Each recipe tunes classification thresholds, reasoning modes, caching strategies, security policies, and observability settings to achieve specific performance goals.

## Available Recipes

### 1. Accuracy-Optimized (`config.recipe-accuracy.yaml`)

**Objective:** Maximum accuracy and response quality

**Use Cases:**

- Research and academic applications
- Critical decision-making systems
- High-stakes business applications
- Medical or legal information systems
- Applications where correctness is paramount

**Key Characteristics:**

- ✅ Reasoning enabled for most complex categories
- ✅ High reasoning effort level (`high`)
- ✅ Strict classification thresholds (0.7)
- ✅ Semantic cache disabled for fresh responses
- ✅ Comprehensive tool selection (top_k: 5)
- ✅ Strict PII detection (threshold: 0.6)
- ✅ Jailbreak protection enabled
- ✅ Full tracing enabled (100% sampling)

**Trade-offs:**

- ⚠️ Higher token usage (~2-3x vs baseline)
- ⚠️ Increased latency (~1.5-2x vs baseline)
- ⚠️ Higher computational costs
- ⚠️ No caching means repeated queries aren't optimized

**Performance Metrics:**

```
Expected latency: 2-5 seconds per request
Token usage: High (reasoning overhead)
Throughput: ~10-20 requests/second
Cost: High (maximum quality)
```

---

### 2. Token Efficiency-Optimized (`config.recipe-token-efficiency.yaml`)

**Objective:** Minimize token usage and reduce operational costs

**Use Cases:**

- High-volume production deployments
- Cost-sensitive applications
- Budget-constrained projects
- Applications with tight token budgets
- Bulk processing workloads

**Key Characteristics:**

- ✅ Reasoning disabled for most categories
- ✅ Low reasoning effort when needed (`low`)
- ✅ Aggressive semantic caching (0.75 threshold, 2hr TTL)
- ✅ Lower classification thresholds (0.5)
- ✅ Minimal tool selection (top_k: 1)
- ✅ Relaxed PII policies
- ✅ Large batch sizes (100)
- ✅ Reduced observability (10% sampling)

**Trade-offs:**

- ⚠️ May sacrifice some accuracy (~5-10%)
- ⚠️ Cache hits depend on query patterns
- ⚠️ Less comprehensive tool coverage
- ⚠️ Relaxed security policies

**Performance Metrics:**

```
Expected latency: 0.5-2 seconds per request
Token usage: Low (~50-60% of baseline)
Throughput: ~50-100 requests/second
Cost: Low (optimized for budget)
Cache hit rate: 40-60% (typical)
```

**Cost Savings:**

- ~40-50% token reduction vs baseline
- ~50-70% cost reduction with effective caching

---

### 3. Latency-Optimized (`config.recipe-latency.yaml`)

**Objective:** Minimize response time and maximize throughput

**Use Cases:**

- Real-time APIs
- Interactive chatbots
- Live customer support systems
- Gaming or entertainment applications
- Applications requiring sub-second responses

**Key Characteristics:**

- ✅ Reasoning disabled for all categories
- ✅ Aggressive semantic caching (0.7 threshold, 3hr TTL)
- ✅ Very low classification thresholds (0.4)
- ✅ Tools disabled for minimal overhead
- ✅ Security checks relaxed/disabled
- ✅ Maximum concurrency (32)
- ✅ Minimal observability overhead (5% sampling)
- ✅ Tracing disabled by default

**Trade-offs:**

- ⚠️ Reduced accuracy (~10-15% vs baseline)
- ⚠️ No reasoning means simpler responses
- ⚠️ Security features minimal/disabled
- ⚠️ Less comprehensive responses

**Performance Metrics:**

```
Expected latency: 0.1-0.8 seconds per request
Token usage: Low (~50-60% of baseline)
Throughput: ~100-200 requests/second
Cost: Low (fast and efficient)
Cache hit rate: 50-70% (typical)
```

**Speed Improvements:**

- ~3-5x faster than accuracy-optimized
- ~2-3x faster than baseline

---

## Quick Start

### Using a Recipe

**Option 1: Direct Usage**

```bash
# Use a recipe directly
cp config/config.recipe-accuracy.yaml config/config.yaml
make run-router
```

**Option 2: Kubernetes/Helm**

```yaml
# In your Helm values.yaml
configMap:
  data:
    config.yaml: |-
      {{- .Files.Get "config.recipe-latency.yaml" | nindent 6 }}
```

**Option 3: Docker Compose**

```yaml
services:
  semantic-router:
    image: vllm/semantic-router:latest
    volumes:
      - ./config/config.recipe-token-efficiency.yaml:/app/config/config.yaml:ro
```

**Option 4: ArgoCD**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: router-config
data:
  config.yaml: |
    # Content from config.recipe-accuracy.yaml
```

### Customizing a Recipe

1. Copy the recipe that best matches your needs:

   ```bash
   cp config/config.recipe-accuracy.yaml config/config.custom.yaml
   ```

2. Modify specific settings in `config.custom.yaml`:

   ```yaml
   # Example: Enable caching in accuracy recipe
   semantic_cache:
     enabled: true  # Was: false
     similarity_threshold: 0.90  # High threshold
   ```

3. Test your custom configuration:

   ```bash
   # Validate YAML syntax
   python -c "import yaml; yaml.safe_load(open('config/config.custom.yaml'))"

   # Test with your custom config
   export CONFIG_FILE=config/config.custom.yaml
   make run-router
   ```

---

## Configuration Comparison

| Feature | Accuracy | Token Efficiency | Latency |
|---------|----------|-----------------|---------|
| **Reasoning (complex tasks)** | ✅ Enabled (high) | ⚠️ Minimal | ❌ Disabled |
| **Semantic Cache** | ❌ Disabled | ✅ Aggressive | ✅ Very Aggressive |
| **Classification Threshold** | 0.7 (strict) | 0.5 (moderate) | 0.4 (relaxed) |
| **Tool Selection** | 5 tools | 1 tool | Disabled |
| **PII Detection** | 0.6 (strict) | 0.8 (relaxed) | 0.9 (minimal) |
| **Jailbreak Protection** | ✅ Enabled | ✅ Enabled | ❌ Disabled |
| **Batch Size** | 50 | 100 | 200 |
| **Max Concurrency** | 4 | 16 | 32 |
| **Tracing Sampling** | 100% | 10% | 5% (disabled) |
| **Expected Latency** | 2-5s | 0.5-2s | 0.1-0.8s |
| **Token Usage** | High | Low (50-60%) | Low (50-60%) |
| **Relative Cost** | High | Low | Low |

---

## Choosing the Right Recipe

### Decision Tree

```
Start Here
│
├─ Need maximum accuracy?
│  └─ → Use: config.recipe-accuracy.yaml
│
├─ Need to minimize costs?
│  └─ → Use: config.recipe-token-efficiency.yaml
│
├─ Need fast responses?
│  └─ → Use: config.recipe-latency.yaml
│
└─ Balanced requirements?
   └─ → Start with: config.yaml (baseline)
      Then customize based on metrics
```

### Use Case Mapping

| Use Case | Recommended Recipe | Reason |
|----------|-------------------|--------|
| Medical diagnosis support | Accuracy | Correctness is critical |
| Legal research assistant | Accuracy | High-stakes decisions |
| Customer chatbot | Latency | Real-time interaction |
| Bulk document processing | Token Efficiency | High volume, cost-sensitive |
| Educational tutor | Accuracy | Quality explanations needed |
| API rate limiting concerns | Token Efficiency | Budget constraints |
| Gaming NPC dialogue | Latency | Sub-second responses |
| Research paper analysis | Accuracy | Comprehensive analysis |

---

## Tuning and Optimization

### Monitoring Your Recipe

After deploying a recipe, monitor these key metrics:

**1. Accuracy Recipe Metrics:**

```bash
# Check reasoning usage
curl localhost:9190/metrics | grep reasoning

# Monitor response quality (manual review)
# Check for comprehensive, detailed answers
```

**2. Token Efficiency Recipe Metrics:**

```bash
# Check cache hit rate
curl localhost:9190/metrics | grep cache_hit

# Monitor token usage
curl localhost:9190/metrics | grep token_count

# Expected cache hit rate: 40-60%
# Expected token reduction: 40-50%
```

**3. Latency Recipe Metrics:**

```bash
# Check p50, p95, p99 latencies
curl localhost:9190/metrics | grep duration_seconds

# Expected p95: < 1 second
# Expected p99: < 2 seconds
```

### Fine-Tuning Parameters

#### To Improve Cache Hit Rate:

```yaml
semantic_cache:
  similarity_threshold: 0.70  # Lower = more hits (was 0.75)
  ttl_seconds: 14400          # Longer TTL (was 7200)
  max_entries: 20000          # Larger cache (was 10000)
```

#### To Reduce Latency Further:

```yaml
classifier:
  category_model:
    threshold: 0.3  # Even lower threshold (was 0.4)

api:
  batch_classification:
    max_concurrency: 64  # More parallel processing (was 32)
```

#### To Balance Accuracy and Cost:

```yaml
# Enable reasoning for select categories only
categories:
  - name: math
    model_scores:
      - model: openai/gpt-oss-20b
        use_reasoning: true  # Enable for critical tasks
  - name: other
    model_scores:
      - model: openai/gpt-oss-20b
        use_reasoning: false  # Disable for simple tasks
```

---

## Best Practices

### 1. Start with a Recipe

Don't start from scratch. Choose the recipe closest to your needs and customize from there.

### 2. A/B Testing

Run two configurations side-by-side and compare metrics:

```bash
# Terminal 1: Accuracy recipe on port 8801
export CONFIG_FILE=config/config.recipe-accuracy.yaml
make run-router

# Terminal 2: Latency recipe on port 8802
export CONFIG_FILE=config/config.recipe-latency.yaml
export PORT=8802
make run-router

# Compare metrics
watch -n 5 'curl -s localhost:9190/metrics | grep duration_seconds_sum'
```

### 3. Monitor and Iterate

- Track metrics for at least 24-48 hours before making changes
- Adjust one parameter at a time
- Document changes and their impact

### 4. Environment-Specific Configs

Use different recipes for different environments:

```bash
# Development: Use latency recipe for fast iteration
config/config.recipe-latency.yaml → config/config.dev.yaml

# Staging: Use accuracy recipe for testing
config/config.recipe-accuracy.yaml → config/config.staging.yaml

# Production: Use token efficiency for cost control
config/config.recipe-token-efficiency.yaml → config/config.prod.yaml
```

### 5. Version Control Your Configs

```bash
# Track your custom configurations
git add config/config.custom-*.yaml
git commit -m "feat: add custom config for production deployment"
```

---

## Advanced: Hybrid Configurations

You can mix and match settings from different recipes:

### Example: High-Accuracy, Low-Cost Hybrid

```yaml
# Base: Token efficiency recipe
# + Enable reasoning for critical categories
# + Strict PII detection
# = Balanced approach

# Start with token efficiency
cp config/config.recipe-token-efficiency.yaml config/config.hybrid.yaml

# Then customize:
categories:
  - name: math
    model_scores:
      - model: openai/gpt-oss-20b
        use_reasoning: true  # From accuracy recipe
  - name: law
    model_scores:
      - model: openai/gpt-oss-20b
        use_reasoning: true  # From accuracy recipe
  # ... other categories keep reasoning: false

classifier:
  pii_model:
    threshold: 0.6  # Stricter (from accuracy recipe)
```

### Example: Fast + Accurate Critical Path

```yaml
# Base: Latency recipe for speed
# + Enable reasoning for specific high-value queries
# = Fast for most, accurate for critical

# Use category-specific reasoning
categories:
  - name: medical
    model_scores:
      - model: openai/gpt-oss-20b
        use_reasoning: true  # Accuracy for critical domain
  - name: other
    model_scores:
      - model: openai/gpt-oss-20b
        use_reasoning: false  # Speed for general queries
```

---

## Troubleshooting

### Recipe Not Performing as Expected

**Problem: Cache hit rate is low (<20%)**

```yaml
# Solution: Lower similarity threshold
semantic_cache:
  similarity_threshold: 0.65  # Lower = more hits
```

**Problem: Too many classification errors**

```yaml
# Solution: Increase classification threshold
classifier:
  category_model:
    threshold: 0.6  # Higher = more confident classifications
```

**Problem: High latency despite using latency recipe**

```yaml
# Solution: Profile and optimize
# 1. Check if reasoning is accidentally enabled
# 2. Verify cache is working (check metrics)
# 3. Increase concurrency
api:
  batch_classification:
    max_concurrency: 64  # Increase parallelism
```

**Problem: Token usage still high with efficiency recipe**

```yaml
# Solution: Verify reasoning is disabled
# Check all categories have use_reasoning: false
# Increase cache hit rate
semantic_cache:
  similarity_threshold: 0.65  # More aggressive caching
  max_entries: 30000  # Larger cache
```

---

## Related Documentation

- [Configuration Guide](../website/docs/installation/configuration.md)
- [Performance Tuning](../website/docs/tutorials/performance-tuning.md)
- [Observability](../website/docs/tutorials/observability/distributed-tracing.md)
- [Cost Optimization](../website/docs/tutorials/cost-optimization.md)
