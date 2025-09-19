# Global Configuration

This guide covers the configuration options for the Semantic Router. The system uses a single YAML configuration file that controls all aspects of routing, classification, and security.

## Configuration File

The configuration file is located at `config/config.yaml`. Here's the structure based on the actual implementation:

```yaml
# config/config.yaml - Actual configuration structure

# BERT model for semantic similarity
bert_model:
  model_id: sentence-transformers/all-MiniLM-L12-v2
  threshold: 0.6
  use_cpu: true

# Semantic caching
semantic_cache:
  backend_type: "memory"  # Options: "memory" or "milvus"
  enabled: false
  similarity_threshold: 0.8
  max_entries: 1000
  ttl_seconds: 3600
  eviction_policy: "fifo"  # Options: "fifo", "lru", "lfu"

# Tool auto-selection
tools:
  enabled: false
  top_k: 3
  similarity_threshold: 0.2
  tools_db_path: "config/tools_db.json"
  fallback_to_empty: true

# Jailbreak protection
prompt_guard:
  enabled: false
  use_modernbert: true
  model_id: "models/jailbreak_classifier_modernbert-base_model"
  threshold: 0.7
  use_cpu: true

# vLLM endpoints - your backend models
vllm_endpoints:
  - name: "endpoint1"
    address: "your-server.com"  # Replace with your server
    port: 11434
    models:
      - "your-model"           # Replace with your model
    weight: 1

# Model configuration
model_config:
  "your-model":
    pii_policy:
      allow_by_default: true
      pii_types_allowed: ["EMAIL_ADDRESS", "PERSON"]
    preferred_endpoints: ["endpoint1"]

# Classification models
classifier:
  category_model:
    model_id: "models/category_classifier_modernbert-base_model"
    use_modernbert: true
    threshold: 0.6
    use_cpu: true
  pii_model:
    model_id: "models/pii_classifier_modernbert-base_presidio_token_model"
    use_modernbert: true
    threshold: 0.7
    use_cpu: true

# Categories and routing rules
categories:
- name: math
  use_reasoning: true  # Enable reasoning for math
  model_scores:
  - model: your-model
    score: 1.0
- name: computer science
  use_reasoning: true  # Enable reasoning for code
  model_scores:
  - model: your-model
    score: 1.0
- name: other
  use_reasoning: false # No reasoning for general queries
  model_scores:
  - model: your-model
    score: 0.8

default_model: your-model

# Reasoning family configurations - define how different model families handle reasoning syntax
reasoning_families:
  deepseek:
    type: "chat_template_kwargs"
    parameter: "thinking"
  
  qwen3:
    type: "chat_template_kwargs"
    parameter: "enable_thinking"
  
  gpt-oss:
    type: "reasoning_effort"
    parameter: "reasoning_effort"
  
  gpt:
    type: "reasoning_effort"
    parameter: "reasoning_effort"

# Global default reasoning effort level
default_reasoning_effort: "medium"

# Model configurations - assign reasoning families to specific models
model_config:
  # Example: DeepSeek model with custom name
  "ds-v31-custom":
    reasoning_family: "deepseek"  # This model uses DeepSeek reasoning syntax
    preferred_endpoints: ["endpoint1"]
  
  # Example: Qwen3 model with custom name
  "my-qwen3-model":
    reasoning_family: "qwen3"     # This model uses Qwen3 reasoning syntax  
    preferred_endpoints: ["endpoint2"]
  
  # Example: Model without reasoning support
  "phi4":
    # No reasoning_family field - this model doesn't support reasoning mode
    preferred_endpoints: ["endpoint1"]
```

## Key Configuration Sections

### Backend Endpoints

Configure your LLM servers:

```yaml
vllm_endpoints:
  - name: "my_endpoint"
    address: "127.0.0.1"  # Your server IP
    port: 8000                # Your server port
    models:
      - "llama2-7b"          # Model name
    weight: 1                 # Load balancing weight
```

### Model Settings

Configure model-specific settings:

```yaml
model_config:
  "llama2-7b":
    pii_policy:
      allow_by_default: true    # Allow PII by default
      pii_types_allowed: ["EMAIL_ADDRESS", "PERSON"]
    preferred_endpoints: ["my_endpoint"]
```

### Pricing (Optional)

If you want the router to compute request cost and expose Prometheus cost metrics, add per-1M token pricing and currency under each model in `model_config`.

```yaml
model_config:
  phi4:
    pricing:
      currency: USD
      prompt_per_1m: 0.07
      completion_per_1m: 0.35
  "mistral-small3.1":
    pricing:
      currency: USD
      prompt_per_1m: 0.1
      completion_per_1m: 0.3
  gemma3:27b:
    pricing:
      currency: USD
      prompt_per_1m: 0.067
      completion_per_1m: 0.267
```

- Cost formula: `(prompt_tokens * prompt_per_1m + completion_tokens * completion_per_1m) / 1_000_000` (in the given currency).
- When not configured, the router still reports token and latency metrics; cost is treated as 0.

### Classification Models

Configure the BERT classification models:

```yaml
classifier:
  category_model:
    model_id: "models/category_classifier_modernbert-base_model"
    use_modernbert: true
    threshold: 0.6            # Classification confidence threshold
    use_cpu: true             # Use CPU (no GPU required)
  pii_model:
    model_id: "models/pii_classifier_modernbert-base_presidio_token_model"
    threshold: 0.7            # PII detection threshold
    use_cpu: true
```

### Categories and Routing

Define how different query types are handled:

```yaml
categories:
- name: math
  use_reasoning: true              # Enable reasoning for math problems
  reasoning_description: "Mathematical problems require step-by-step reasoning"
  model_scores:
  - model: your-model
    score: 1.0                     # Preference score for this model

- name: computer science
  use_reasoning: true              # Enable reasoning for code
  model_scores:
  - model: your-model
    score: 1.0

- name: other
  use_reasoning: false             # No reasoning for general queries
  model_scores:
  - model: your-model
    score: 0.8

default_model: your-model          # Fallback model
```

### Model Reasoning Configuration

Configure how different models handle reasoning mode syntax. This allows you to add new models without code changes:

```yaml
# Model reasoning configurations - define how different models handle reasoning syntax
model_reasoning_configs:
  - name: "deepseek"
    patterns: ["deepseek", "ds-", "ds_", "ds:", "ds "]
    reasoning_syntax:
      type: "chat_template_kwargs"
      parameter: "thinking"

  - name: "qwen3"
    patterns: ["qwen3"]
    reasoning_syntax:
      type: "chat_template_kwargs"
      parameter: "enable_thinking"

  - name: "gpt-oss"
    patterns: ["gpt-oss", "gpt_oss"]
    reasoning_syntax:
      type: "reasoning_effort"
      parameter: "reasoning_effort"

  - name: "gpt"
    patterns: ["gpt"]
    reasoning_syntax:
      type: "reasoning_effort"
      parameter: "reasoning_effort"

# Global default reasoning effort level (when not specified per category)
default_reasoning_effort: "medium"
```

#### Model Reasoning Configuration Options

**Configuration Structure:**

- `name`: A unique identifier for the model family
- `patterns`: Array of patterns to match against model names
- `reasoning_syntax.type`: How the model expects reasoning mode to be specified
  - `"chat_template_kwargs"`: Use chat template parameters (for models like DeepSeek, Qwen3)
  - `"reasoning_effort"`: Use OpenAI-compatible reasoning_effort field (for GPT models)
- `reasoning_syntax.parameter`: The specific parameter name the model uses

**Pattern Matching:**
The system supports both simple string patterns and regular expressions for flexible model matching:

- **Simple string matches**: `"deepseek"` matches any model containing "deepseek"
- **Prefix patterns**: `"ds-"` matches models starting with "ds-" or exactly "ds"
- **Regular expressions**: `"^gpt-4.*"` matches models starting with "gpt-4"
- **Wildcard**: `"*"` matches all models (use for fallback configurations)
- **Multiple patterns**: `["deepseek", "ds-", "^phi.*"]` matches any of these patterns

**Regex Pattern Examples:**

```yaml
patterns:
  - "^gpt-4.*"        # Models starting with "gpt-4"
  - ".*-instruct$"    # Models ending with "-instruct"
  - "phi[0-9]+"       # Models like "phi3", "phi4", etc.
  - "^(llama|mistral)" # Models starting with "llama" or "mistral"
```

**Adding New Models:**
To support a new model family (e.g., Claude), simply add a new configuration:

```yaml
model_reasoning_configs:
  - name: "claude"
    patterns: ["claude"]
    reasoning_syntax:
      type: "chat_template_kwargs"
      parameter: "enable_reasoning"
```

**Unknown Models:**
Models that don't match any configured pattern will have no reasoning fields applied when reasoning mode is enabled. This prevents issues with models that don't support reasoning syntax.

**Default Reasoning Effort:**
Set the global default reasoning effort level used when categories don't specify their own effort level:

```yaml
default_reasoning_effort: "high"  # Options: "low", "medium", "high"
```

**Category-Specific Reasoning Effort:**
Override the default effort level per category:

```yaml
categories:
- name: math
  use_reasoning: true
  reasoning_effort: "high"        # Use high effort for complex math
  model_scores:
  - model: your-model
    score: 1.0

- name: general
  use_reasoning: true
  reasoning_effort: "low"         # Use low effort for general queries
  model_scores:
  - model: your-model
    score: 1.0
```

### Security Features

Configure PII detection and jailbreak protection:

```yaml
# PII Detection
classifier:
  pii_model:
    threshold: 0.7                 # Higher = more strict PII detection

# Jailbreak Protection
prompt_guard:
  enabled: true                    # Enable jailbreak detection
  threshold: 0.7                   # Detection sensitivity
  use_cpu: true                    # Runs on CPU

# Model-level PII policies
model_config:
  "your-model":
    pii_policy:
      allow_by_default: true       # Allow most content
      pii_types_allowed: ["EMAIL_ADDRESS", "PERSON"]  # Specific allowed types
```

### Optional Features

Configure additional features:

```yaml
# Semantic Caching
semantic_cache:
  enabled: true                   # Enable semantic caching
  backend_type: "memory"          # Options: "memory" or "milvus"
  similarity_threshold: 0.8       # Cache hit threshold
  max_entries: 1000               # Maximum cache entries
  ttl_seconds: 3600               # Cache expiration time
  eviction_policy: "fifo"         # Options: "fifo", "lru", "lfu"

# Tool Auto-Selection
tools:
  enabled: true                    # Enable automatic tool selection
  top_k: 3                        # Number of tools to select
  similarity_threshold: 0.2        # Tool relevance threshold
  tools_db_path: "config/tools_db.json"
  fallback_to_empty: true         # Return empty on failure

# BERT Model for Similarity
bert_model:
  model_id: sentence-transformers/all-MiniLM-L12-v2
  threshold: 0.6                  # Similarity threshold
  use_cpu: true                   # CPU-only inference

# Batch Classification API Configuration
api:
  batch_classification:
    max_batch_size: 100            # Maximum texts per batch request
    concurrency_threshold: 5       # Switch to concurrent processing at this size
    max_concurrency: 8             # Maximum concurrent goroutines
    
    # Metrics configuration for monitoring
    metrics:
      enabled: true                # Enable Prometheus metrics collection
      detailed_goroutine_tracking: true  # Track individual goroutine lifecycle
      high_resolution_timing: false      # Use nanosecond precision timing
      sample_rate: 1.0                   # Collect metrics for all requests (1.0 = 100%)
      
      # Batch size range labels for metrics (OPTIONAL - uses sensible defaults)
      # Default ranges: "1", "2-5", "6-10", "11-20", "21-50", "50+"
      # Only specify if you need custom ranges:
      # batch_size_ranges:
      #   - {min: 1, max: 1, label: "1"}
      #   - {min: 2, max: 5, label: "2-5"}
      #   - {min: 6, max: 10, label: "6-10"}
      #   - {min: 11, max: 20, label: "11-20"}
      #   - {min: 21, max: 50, label: "21-50"}
      #   - {min: 51, max: -1, label: "50+"}  # -1 means no upper limit
      
      # Histogram buckets - choose from presets below or customize
      duration_buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30]
      size_buckets: [1, 2, 5, 10, 20, 50, 100, 200]
      
      # Preset examples for quick configuration (copy values above)
      preset_examples:
        fast:
          duration: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
          size: [1, 2, 3, 5, 8, 10]
        standard:
          duration: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
          size: [1, 2, 5, 10, 20, 50, 100]
        slow:
          duration: [0.1, 0.5, 1, 5, 10, 30, 60, 120]
          size: [10, 50, 100, 500, 1000, 5000]
```

### How to Use Preset Examples

The configuration includes preset examples for quick setup. Here's how to use them:

**Step 1: Choose your scenario**

- `fast` - For real-time APIs (microsecond to millisecond response times)
- `standard` - For typical web APIs (millisecond to second response times)  
- `slow` - For batch processing or heavy computation (seconds to minutes)

**Step 2: Copy the preset values**

```yaml
# Example: Switch to fast API configuration
# Copy from preset_examples.fast and paste to the actual config:
duration_buckets: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
size_buckets: [1, 2, 3, 5, 8, 10]
```

**Step 3: Restart the service**

```bash
pkill -f "router"
make run-router
```

### Default Batch Size Ranges

The system provides sensible default batch size ranges that work well for most use cases:

- **"1"** - Single text requests
- **"2-5"** - Small batch requests  
- **"6-10"** - Medium batch requests
- **"11-20"** - Large batch requests
- **"21-50"** - Very large batch requests
- **"50+"** - Maximum batch requests

**You don't need to configure `batch_size_ranges` unless you have specific requirements.** The defaults are automatically used when the configuration is omitted.

### Configuration Examples by Use Case

**Real-time Chat API (fast preset)**

```yaml
# Copy these values to your config for sub-millisecond monitoring
duration_buckets: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
size_buckets: [1, 2, 3, 5, 8, 10]
# batch_size_ranges: uses defaults (no configuration needed)
```

**E-commerce API (standard preset)**

```yaml
# Copy these values for typical web API response times
duration_buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
size_buckets: [1, 2, 5, 10, 20, 50, 100]
# batch_size_ranges: uses defaults (no configuration needed)
```

**Data Processing Pipeline (slow preset)**

```yaml
# Copy these values for heavy computation workloads
duration_buckets: [0.1, 0.5, 1, 5, 10, 30, 60, 120]
size_buckets: [10, 50, 100, 500, 1000, 5000]
# Custom batch size ranges for large-scale processing (overrides defaults)
batch_size_ranges:
  - {min: 1, max: 50, label: "1-50"}
  - {min: 51, max: 200, label: "51-200"}
  - {min: 201, max: 1000, label: "201-1000"}
  - {min: 1001, max: -1, label: "1000+"}
```

**Available Metrics:**

- `batch_classification_requests_total` - Total number of batch requests
- `batch_classification_duration_seconds` - Processing duration histogram
- `batch_classification_texts_total` - Total number of texts processed
- `batch_classification_errors_total` - Error count by type
- `batch_classification_concurrent_goroutines` - Active goroutine count
- `batch_classification_size_distribution` - Batch size distribution

Access metrics at: `http://localhost:9190/metrics`

## Common Configuration Examples

### Enable All Security Features

```yaml
# Enable PII detection
classifier:
  pii_model:
    threshold: 0.8              # Strict PII detection

# Enable jailbreak protection
prompt_guard:
  enabled: true
  threshold: 0.7

# Configure model PII policies
model_config:
  "your-model":
    pii_policy:
      allow_by_default: false   # Block all PII by default
      pii_types_allowed: []     # No PII allowed
```

### Performance Optimization

```yaml
# Enable caching
semantic_cache:
  enabled: true
  backend_type: "memory"
  similarity_threshold: 0.85    # Higher = more cache hits
  max_entries: 5000
  ttl_seconds: 7200             # 2 hour cache
  eviction_policy: "fifo"       # Options: "fifo", "lru", "lfu"

# Enable tool selection
tools:
  enabled: true
  top_k: 5                     # Select more tools
  similarity_threshold: 0.1    # Lower = more tools selected
```

### Development Setup

```yaml
# Disable security for testing
prompt_guard:
  enabled: false

# Disable caching for consistent results
semantic_cache:
  enabled: false

# Lower classification thresholds
classifier:
  category_model:
    threshold: 0.3             # Lower = more specialized routing
```

## Configuration Validation

### Test Your Configuration

Validate your configuration before starting:

```bash
# Test configuration syntax
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"

# Test the router with your config
make build
make run-router
```

### Common Configuration Patterns

**Multiple Models:**

```yaml
vllm_endpoints:
  - name: "math_endpoint"
    address: "math-server.com"
    port: 8000
    models: ["math-model"]
    weight: 1
  - name: "general_endpoint"
    address: "general-server.com"
    port: 8000
    models: ["general-model"]
    weight: 1

categories:
- name: math
  model_scores:
  - model: math-model
    score: 1.0
- name: other
  model_scores:
  - model: general-model
    score: 1.0
```

**Load Balancing:**

```yaml
vllm_endpoints:
  - name: "endpoint1"
    address: "server1.com"
    port: 8000
    models: ["my-model"]
    weight: 2              # Higher weight = more traffic
  - name: "endpoint2"
    address: "server2.com"
    port: 8000
    models: ["my-model"]
    weight: 1
```

## Best Practices

### Security Configuration

For production environments:

```yaml
# Enable all security features
classifier:
  pii_model:
    threshold: 0.8              # Strict PII detection

prompt_guard:
  enabled: true                 # Enable jailbreak protection
  threshold: 0.7

model_config:
  "your-model":
    pii_policy:
      allow_by_default: false   # Block PII by default
```

### Performance Tuning

For high-traffic scenarios:

```yaml
# Enable caching
semantic_cache:
  enabled: true
  backend_type: "memory"
  similarity_threshold: 0.85    # Higher = more cache hits
  max_entries: 10000
  ttl_seconds: 3600
  eviction_policy: "lru"        

# Optimize classification
classifier:
  category_model:
    threshold: 0.7              # Balance accuracy vs speed
```

### Development vs Production

**Development:**

```yaml
# Relaxed settings for testing
classifier:
  category_model:
    threshold: 0.3              # Lower threshold for testing
prompt_guard:
  enabled: false                # Disable for development
semantic_cache:
  enabled: false                # Disable for consistent results
```

**Production:**

```yaml
# Strict settings for production
classifier:
  category_model:
    threshold: 0.7              # Higher threshold for accuracy
prompt_guard:
  enabled: true                 # Enable security
semantic_cache:
  enabled: true                 # Enable for performance
```

## Troubleshooting

### Common Issues

**Invalid YAML syntax:**

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"
```

**Missing model files:**

```bash
# Check if models are downloaded
ls -la models/
# If missing, run: make download-models
```

**Endpoint connectivity:**

```bash
# Test your backend server
curl -f http://your-server:8000/health
```

**Configuration not taking effect:**

```bash
# Restart the router after config changes
make run-router
```

### Testing Configuration

```bash
# Test with different queries
make test-auto-prompt-reasoning      # Math query
make test-auto-prompt-no-reasoning   # General query
make test-pii                        # PII detection
make test-prompt-guard               # Jailbreak protection
```

### Model Reasoning Configuration Issues

**Model not getting reasoning fields:**

- Check that the model name matches a pattern in `model_reasoning_configs`
- Verify the pattern syntax (exact matches vs prefixes)
- Unknown models will have no reasoning fields applied (this is by design)

**Wrong reasoning syntax applied:**

- Ensure the `reasoning_syntax.type` matches your model's expected format
- Check the `reasoning_syntax.parameter` name is correct
- DeepSeek models typically use `chat_template_kwargs` with `"thinking"`
- GPT models typically use `reasoning_effort`

**Adding support for new models:**

```yaml
# Add a new model configuration
model_reasoning_configs:
  - name: "my-new-model"
    patterns: ["my-model"]
    reasoning_syntax:
      type: "chat_template_kwargs"  # or "reasoning_effort"
      parameter: "custom_parameter"
```

**Testing model reasoning configuration:**

```bash
# Test reasoning with your specific model
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

## Configuration Generation

The Semantic Router supports automated configuration generation based on model performance benchmarks. This workflow uses MMLU-Pro evaluation results to determine optimal model routing for different categories.

### Benchmarking Workflow

1. **Run MMLU-Pro Evaluation:**

   ```bash
   # Evaluate models using MMLU-Pro benchmark
   python src/training/model_eval/mmlu_pro_vllm_eval.py \
     --endpoint http://localhost:8000/v1 \
     --models phi4,gemma3:27b,mistral-small3.1 \
     --samples-per-category 5 \
     --use-cot \
     --concurrent-requests 4 \
     --output-dir results
   ```

2. **Generate Configuration:**

   ```bash
   # Generate config.yaml from benchmark results
   python src/training/model_eval/result_to_config.py \
     --results-dir results \
     --output-file config/config.yaml \
     --similarity-threshold 0.80
   ```

### Generated Configuration Features

The generated configuration includes:

- **Model Performance Rankings:** Models are ranked by performance for each category
- **Reasoning Settings:** Automatically configures reasoning requirements per category:
  - `use_reasoning`: Whether to use step-by-step reasoning
  - `reasoning_description`: Description of reasoning approach
  - `reasoning_effort`: Required effort level (low/medium/high)
- **Default Model Selection:** Best overall performing model is set as default
- **Security and Performance Settings:** Pre-configured optimal values for:
  - PII detection thresholds
  - Semantic cache settings
  - Tool selection parameters

### Customizing Generated Config

The generated config.yaml can be customized by:

1. Editing category-specific settings in `result_to_config.py`
2. Adjusting thresholds and parameters via command line arguments
3. Manually modifying the generated config.yaml

### Example Workflow

Here's a complete example workflow for generating and testing a configuration:

```bash
# Run MMLU-Pro evaluation
# Option 1: Specify models manually
python src/training/model_eval/mmlu_pro_vllm_eval.py \
  --endpoint http://localhost:8000/v1 \
  --models phi4,gemma3:27b,mistral-small3.1 \
  --samples-per-category 5 \
  --use-cot \
  --concurrent-requests 4 \
  --output-dir results \
  --max-tokens 2048 \
  --temperature 0.0 \
  --seed 42

# Option 2: Auto-discover models from endpoint
python src/training/model_eval/mmlu_pro_vllm_eval.py \
  --endpoint http://localhost:8000/v1 \
  --samples-per-category 5 \
  --use-cot \
  --concurrent-requests 4 \
  --output-dir results \
  --max-tokens 2048 \
  --temperature 0.0 \
  --seed 42

# Generate initial config
python src/training/model_eval/result_to_config.py \
  --results-dir results \
  --output-file config/config.yaml \
  --similarity-threshold 0.80

# Test the generated config
make test
```

This workflow ensures your configuration is:

- Based on actual model performance
- Properly tested before deployment
- Version controlled for tracking changes
- Optimized for your specific use case

## Next Steps

- **[Installation Guide](installation.md)** - Setup instructions
- **[Quick Start Guide](installation.md)** - Basic usage examples
- **[API Documentation](../api/router.md)** - Complete API reference

The configuration system is designed to be simple yet powerful. Start with the basic configuration and gradually enable advanced features as needed.
