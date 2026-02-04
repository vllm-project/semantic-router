---
sidebar_position: 4
---

# Configuration

This guide covers the configuration options for the Semantic Router. The system uses a single YAML configuration file that controls **signal-driven routing**, **plugin chain processing**, and **model selection**.

## Architecture Overview

The configuration defines three main layers:

1. **Signal Extraction Layer**: Define 9 types of signals (keyword, embedding, domain, fact_check, user_feedback, preference, language, latency, context)
2. **Decision Engine**: Combine signals using AND/OR operators to make routing decisions
3. **Plugin Chain**: Configure plugins for caching, security, and optimization

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
  similarity_threshold: 0.8  # Global default threshold
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
  enabled: false  # Global default - can be overridden per category
  use_modernbert: true
  model_id: "models/jailbreak_classifier_modernbert-base_model"
  threshold: 0.7
  use_cpu: true

# vLLM endpoints - your backend models
vllm_endpoints:
  - name: "endpoint1"
    address: "192.168.1.100"  # Replace with your server IP address
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
  # Example: DeepSeek model with custom name
  "ds-v31-custom":
    reasoning_family: "deepseek"  # Uses DeepSeek reasoning syntax
    preferred_endpoints: ["endpoint1"]
  # Example: Qwen3 model with custom name
  "my-qwen3-model":
    reasoning_family: "qwen3"     # Uses Qwen3 reasoning syntax
    preferred_endpoints: ["endpoint2"]
  # Example: Model without reasoning support
  "phi4":
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

# Signals - Signal extraction configuration
signals:
  # Keyword-based signals (fast pattern matching)
  keywords:
    - name: "math_keywords"
      operator: "OR"
      keywords:
        - "calculate"
        - "equation"
        - "solve"
        - "derivative"
        - "integral"
      case_sensitive: false

    - name: "code_keywords"
      operator: "OR"
      keywords:
        - "function"
        - "class"
        - "debug"
        - "compile"
      case_sensitive: false

  # Embedding-based signals (semantic similarity)
  embeddings:
    - name: "code_debug"
      threshold: 0.70
      candidates:
        - "how to debug the code"
        - "troubleshooting steps for my code"
      aggregation_method: "max"

    - name: "math_intent"
      threshold: 0.75
      candidates:
        - "solve mathematical problem"
        - "calculate the result"
      aggregation_method: "max"

  # Domain signals (MMLU classification)
  domains:
    - name: "mathematics"
      description: "Mathematical and computational problems"
      mmlu_categories:
        - "abstract_algebra"
        - "college_mathematics"
        - "elementary_mathematics"

    - name: "computer_science"
      description: "Programming and computer science"
      mmlu_categories:
        - "computer_security"
        - "machine_learning"

  # Fact check signals (verification need detection)
  fact_check:
    - name: "needs_verification"
      description: "Queries requiring fact verification"

  # User feedback signals (satisfaction analysis)
  user_feedbacks:
    - name: "correction_needed"
      description: "User indicates previous answer was wrong"

  # Preference signals (LLM-based matching)
  preferences:
    - name: "complex_reasoning"
      description: "Requires deep reasoning and analysis"
      llm_endpoint: "http://localhost:11434"

# Categories - Define domain categories
categories:
- name: math
- name: computer science
- name: other

# Decisions - Combine signals to make routing decisions
decisions:
- name: math
  description: "Route mathematical queries"
  priority: 10
  rules:
    operator: "OR"  # Match ANY of these conditions
    conditions:
      - type: "keyword"
        name: "math_keywords"
      - type: "embedding"
        name: "math_intent"
      - type: "domain"
        name: "mathematics"
  modelRefs:
    - model: your-model
      use_reasoning: true  # Enable reasoning for math problems
  # Optional: Decision-level plugins
  plugins:
    - type: "semantic-cache"
      configuration:
        enabled: true
        similarity_threshold: 0.9  # Higher threshold for math
    - type: "jailbreak"
      configuration:
        enabled: true
    - type: "pii"
      configuration:
        enabled: true
        threshold: 0.8
    - type: "system_prompt"
      configuration:
        enabled: true
        prompt: "You are a mathematics expert. Solve problems step by step."

- name: computer_science
  description: "Route computer science queries"
  priority: 10
  rules:
    operator: "OR"
    conditions:
      - type: "keyword"
        name: "code_keywords"
      - type: "embedding"
        name: "code_debug"
      - type: "domain"
        name: "computer_science"
  modelRefs:
    - model: your-model
      use_reasoning: true  # Enable reasoning for code
  plugins:
    - type: "semantic-cache"
      configuration:
        enabled: true
        similarity_threshold: 0.85
    - type: "system_prompt"
      configuration:
        enabled: true
        prompt: "You are a programming expert. Provide clear code examples."

- name: other
  description: "Route general queries"
  priority: 5
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "other"
  modelRefs:
    - model: your-model
      use_reasoning: false # No reasoning for general queries
  plugins:
    - type: "semantic-cache"
      configuration:
        enabled: true
        similarity_threshold: 0.75  # Lower threshold for general queries

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

```

Assign reasoning families inside the same `model_config` block above—use `reasoning_family` per model (see `ds-v31-custom` and `my-qwen3-model` in the example). Models without reasoning syntax simply omit the field (e.g., `phi4`).

## Configuration Recipes (presets)

We provide curated, versioned presets you can use directly or as a starting point:

- Accuracy optimized: https://github.com/vllm-project/semantic-router/blob/main/config/config.recipe-accuracy.yaml
- Token efficiency optimized: https://github.com/vllm-project/semantic-router/blob/main/config/config.recipe-token-efficiency.yaml
- Latency optimized: https://github.com/vllm-project/semantic-router/blob/main/config/config.recipe-latency.yaml
- Guide and usage: https://github.com/vllm-project/semantic-router/blob/main/config/RECIPES.md

Quick usage:

- Local: copy a recipe over config.yaml, then run
  - cp config/config.recipe-accuracy.yaml config/config.yaml
  - make run-router
- Helm/Argo: reference the recipe file contents in your config map (examples are in the guide above).

## Signals Configuration

Signals are the foundation of intelligent routing. The system supports 10 types of signals that can be combined to make routing decisions.

### 1. Keyword Signals - Fast Pattern Matching

```yaml
signals:
  keywords:
    - name: "math_keywords"
      operator: "OR"  # OR: match any keyword, AND: match all keywords
      keywords:
        - "calculate"
        - "equation"
        - "solve"
      case_sensitive: false
```

**Use Cases:**

- Deterministic routing for specific terms
- Compliance and security (PII keywords, banned terms)
- High-throughput scenarios requiring &lt;1ms latency

### 2. Embedding Signals - Semantic Understanding

```yaml
signals:
  embeddings:
    - name: "code_debug"
      threshold: 0.70  # Similarity threshold (0-1)
      candidates:
        - "how to debug the code"
        - "troubleshooting steps"
      aggregation_method: "max"  # max, avg, or min
```

**Use Cases:**

- Intent detection robust to paraphrasing
- Semantic similarity matching
- Handling diverse user phrasings

### 3. Domain Signals - MMLU Classification

```yaml
signals:
  domains:
    - name: "mathematics"
      description: "Mathematical problems"
      mmlu_categories:
        - "abstract_algebra"
        - "college_mathematics"
```

**Use Cases:**

- Academic and professional domain routing
- Subject-matter expert model selection
- 14 MMLU categories supported

### 4. Fact Check Signals - Verification Need Detection

```yaml
signals:
  fact_check:
    - name: "needs_verification"
      description: "Queries requiring fact verification"
```

**Use Cases:**

- Identify factual queries vs creative/code tasks
- Route to models with hallucination detection
- Trigger fact-checking plugins

### 5. User Feedback Signals - Satisfaction Analysis

```yaml
signals:
  user_feedbacks:
    - name: "correction_needed"
      description: "User indicates previous answer was wrong"
```

**Use Cases:**

- Handle follow-up corrections ("that's wrong", "try again")
- Detect satisfaction levels
- Route to more capable models for retries

### 6. Preference Signals - LLM-based Matching

```yaml
signals:
  preferences:
    - name: "complex_reasoning"
      description: "Requires deep reasoning"
      llm_endpoint: "http://localhost:11434"
```

**Use Cases:**

- Complex intent analysis via external LLM
- Nuanced routing decisions
- When other signals are insufficient

### 7. Language Signals - Multi-language Detection

```yaml
signals:
  language:
    - name: "en"
      description: "English language queries"
    - name: "es"
      description: "Spanish language queries"
    - name: "zh"
      description: "Chinese language queries"
    - name: "ru"
      description: "Russian language queries"
    - name: "fr"
      description: "French language queries"
```

**Use Cases:**

- Route queries to language-specific models
- Apply language-specific policies
- Support multilingual applications
- Supports 100+ languages via whatlanggo library

### 8. Latency Signals - Percentile-based Routing

```yaml
signals:
  latency:
    # RECOMMENDED: Use both TPOT and TTFT percentiles for comprehensive evaluation
    - name: "low_latency_comprehensive"
      tpot_percentile: 10  # 10th percentile for TPOT (top 10% fastest token generation)
      ttft_percentile: 10  # 10th percentile for TTFT (top 10% fastest first token)
      description: "For real-time applications - fast start and fast generation"
    
    # Different percentiles for different priorities
    - name: "balanced_latency"
      tpot_percentile: 50  # Median TPOT (top 50%)
      ttft_percentile: 10  # Top 10% TTFT (prioritize fast start)
      description: "Prioritize fast start, accept moderate generation speed"
    
    # TPOT percentile only (use case: batch processing)
    - name: "batch_processing_optimized"
      tpot_percentile: 10  # 10th percentile for TPOT
      description: "For batch processing where throughput (TPOT) is critical"
    
    # TTFT percentile only (use case: real-time chat)
    - name: "chat_fast_start"
      ttft_percentile: 10  # 10th percentile for TTFT
      description: "For chat applications where fast first token (TTFT) is critical"
```

**Use Cases:**

- Route latency-sensitive queries to faster models based on adaptive percentile thresholds
- Optimize for real-time applications (chat, streaming)
- Balance latency vs. capability based on query requirements
- TPOT (Time Per Output Token) and TTFT (Time To First Token) are automatically tracked from responses

**How it works**:

- Percentile-based thresholds adapt to each model's actual performance distribution
- Works with any number of observations: uses average for 1-2 observations, percentile calculation for 3+
- When both TPOT and TTFT percentiles are set, model must meet BOTH thresholds (AND logic)
- **Performance**: Typically 2-5ms for 10 models (runs asynchronously in goroutine, doesn't block requests) - percentile calculation with O(n log n) complexity where n = observations per model (typically 10-100, max 1000)
- **Recommendation**: Use both TPOT and TTFT percentiles for comprehensive latency evaluation

### 9. Context Signals - Token Count Routing

```yaml
signals:
  context_rules:
    - name: "low_token_count"
      min_tokens: "0"
      max_tokens: "1K"
      description: "Short requests"
    - name: "high_token_count"
      min_tokens: "1K"
      max_tokens: "128K"
      description: "Long context requests"
```

**Use Cases:**

- Route long documents to models with larger context windows
- Send short queries to faster, smaller models
- Optimize cost by routing based on request size
- Supports "K" (thousand) and "M" (million) suffixes

### 10. Complexity Signals - Query Difficulty Classification

**IMPORTANT**: It is **strongly recommended** to configure a `composer` for each complexity rule to filter based on other signals (e.g., domain). This prevents misclassification where a math question might match `code_complexity` or vice versa.

```yaml
signals:
  complexity:
    - name: "code_complexity"
      composer:
        operator: "AND"
        conditions:
          - type: "domain"
            name: "computer_science"
      threshold: 0.1
      description: "Detects code complexity level based on task difficulty"
      hard:
        candidates:
          - "design distributed system"
          - "implement consensus algorithm"
          - "optimize for scale"
          - "architect microservices"
      easy:
        candidates:
          - "print hello world"
          - "loop through array"
          - "read file"
          - "sort list"

    - name: "math_complexity"
      composer:
        operator: "AND"
        conditions:
          - type: "domain"
            name: "math"
      threshold: 0.1
      description: "Detects mathematical problem complexity"
      hard:
        candidates:
          - "prove mathematically"
          - "derive the equation"
          - "formal proof"
          - "solve differential equation"
      easy:
        candidates:
          - "add two numbers"
          - "calculate percentage"
          - "simple arithmetic"
          - "basic algebra"
```

**Use Cases:**

- Route complex queries to powerful, specialized models
- Route simple queries to fast, efficient models
- Optimize cost by using cheaper models for easy tasks
- Improve response quality by matching query difficulty to model capability
- Combine with domain signals to avoid cross-domain misclassification

**How it works:**

1. **Parallel Signal Evaluation**: All complexity rules are evaluated independently in parallel with other signals
2. **Difficulty Classification**: For each rule:
   - Query is compared to hard and easy candidates using embedding similarity
   - Difficulty signal = max_hard_similarity - max_easy_similarity
   - If signal > threshold: "hard", if signal < -threshold: "easy", else: "medium"
3. **Composer Filtering** (Phase 2): After all signals are computed:
   - If a rule has a `composer`, its conditions are evaluated against other signal results
   - Only rules whose composer conditions are satisfied are kept
   - This prevents cross-domain misclassification (e.g., math queries matching code_complexity)
4. **Result Format**: Returns "rule_name:difficulty" for each matched rule (e.g., "code_complexity:hard")

**Configuration Parameters:**

- `name`: Unique identifier for the rule
- `threshold`: Similarity difference threshold (default: 0.1)
- `composer` (optional but **strongly recommended**): Filter based on other signals
  - `operator`: "AND" or "OR" for combining conditions
  - `conditions`: Array of signal conditions (type and name)
- `description`: Human-readable description (optional, for documentation only)
- `hard.candidates`: List of phrases representing complex queries
- `easy.candidates`: List of phrases representing simple queries

**Example with Composer:**

```yaml
decisions:
  - name: "hard_code_problems"
    description: "Route complex coding problems to specialized model"
    priority: 15
    rules:
      operator: "AND"
      conditions:
        - type: "complexity"
          name: "code_complexity:hard"
    modelRefs:
      - model: "deepseek-coder-v3"
        use_reasoning: true
        reasoning_effort: "high"
```

In this example, the complexity signal will only match if:

1. The query is classified as "hard" based on hard/easy candidates
2. The domain signal has matched "computer_science" (due to composer)

## Decision Rules - Signal Fusion

Combine signals using AND/OR operators:

```yaml
decisions:
  - name: math
    description: "Route mathematical queries"
    priority: 10
    rules:
      operator: "OR"  # Match ANY condition
      conditions:
        - type: "keyword"
          name: "math_keywords"
        - type: "embedding"
          name: "math_intent"
        - type: "domain"
          name: "mathematics"
    modelRefs:
      - model: math-specialist
        weight: 1.0
```

**Example with Complexity Signal:**

```yaml
decisions:
  - name: complex_code
    description: "Route complex coding tasks to specialized model"
    priority: 15
    rules:
      operator: "AND"
      conditions:
        - type: "complexity"
          name: "code_complexity:hard"
        - type: "domain"
          name: "computer_science"
    modelRefs:
      - model: deepseek-coder-v2
        weight: 1.0
```

**Strategies:**

- **Priority-based**: Higher priority decisions evaluated first
- **Confidence-based**: Select decision with highest confidence score
- **Hybrid**: Combine priority and confidence

## Plugin Chain Configuration

Plugins process requests/responses in a chain. Each decision can override global plugin settings.

### Global Plugin Configuration

```yaml
# Global defaults
semantic_cache:
  enabled: true
  similarity_threshold: 0.8

prompt_guard:
  enabled: true
  threshold: 0.7

classifier:
  pii_model:
    enabled: true
    threshold: 0.8
```

### Decision-Level Plugin Override

```yaml
decisions:
  - name: math
    description: "Route mathematical queries"
    priority: 10
    plugins:
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.9  # Higher for math
      - type: "jailbreak"
        configuration:
          enabled: true
      - type: "pii"
        configuration:
          enabled: true
          threshold: 0.8
      - type: "system_prompt"
        configuration:
          enabled: true
          prompt: "You are a mathematics expert."
      - type: "header_mutation"
        configuration:
          enabled: true
          headers:
            X-Math-Mode: "enabled"
      - type: "hallucination"
        configuration:
          enabled: false  # Optional real-time detection
```

### Plugin Types

| Plugin | Description | Configuration |
|--------|-------------|---------------|
| **semantic-cache** | Semantic similarity-based caching | `similarity_threshold`, `ttl_seconds` |
| **jailbreak** | Adversarial prompt detection | `threshold`, `model_id` |
| **pii** | PII detection and masking | `threshold`, `pii_types_allowed` |
| **system_prompt** | Dynamic prompt injection | `prompt` |
| **header_mutation** | HTTP header manipulation | `headers` |
| **hallucination** | Token-level hallucination detection | `enabled` |

## Key Configuration Sections

### Backend Endpoints

Configure your LLM servers:

```yaml
vllm_endpoints:
  - name: "my_endpoint"
    address: "127.0.0.1"  # Your server IP - MUST be IP address format
    port: 8000            # Your server port
    weight: 1             # Load balancing weight

# Model configuration - maps models to endpoints
model_config:
  "llama2-7b":            # Model name - must match vLLM --served-model-name
    preferred_endpoints: ["my_endpoint"]
  "qwen3":               # Another model served by the same endpoint
    preferred_endpoints: ["my_endpoint"]
```

### Example: Llama / Qwen Backend Configuration

```yaml
vllm_endpoints:
  - name: "local-vllm"
    address: "127.0.0.1"
    port: 8000

model_config:
  "llama2-7b":
    preferred_endpoints: ["local-vllm"]
  "qwen3":
    preferred_endpoints: ["local-vllm"]
```

#### Address Format Requirements

**IMPORTANT**: The `address` field must contain a valid IP address (IPv4 or IPv6). Domain names and other formats are not supported.

**✅ Supported formats:**

```yaml
# IPv4 addresses
address: "127.0.0.1"

# IPv6 addresses
address: "2001:db8::1"
```

**❌ NOT supported:**

```yaml
# Domain names
address: "localhost"        # ❌ Use 127.0.0.1 instead
address: "api.openai.com"   # ❌ Use IP address instead

# Protocol prefixes
address: "http://127.0.0.1"   # ❌ Remove protocol prefix

# Paths
address: "127.0.0.1/api"      # ❌ Remove path, use IP only

# Ports in address
address: "127.0.0.1:8080"     # ❌ Use separate 'port' field
```

#### Model Name Consistency

Model names in `model_config` must **exactly match** the `--served-model-name` parameter used when starting your vLLM server:

```bash
# vLLM server command (examples):
vllm serve meta-llama/Llama-2-7b-hf --served-model-name llama2-7b --port 8000
vllm serve Qwen/Qwen3-1.8B --served-model-name qwen3 --port 8000

# config.yaml must reference the model in model_config:
model_config:
  "llama2-7b":  # ✅ Matches --served-model-name
    preferred_endpoints: ["your-endpoint"]
  "qwen3":      # ✅ Matches --served-model-name
    preferred_endpoints: ["your-endpoint"]
```

### Model Settings

Configure model-specific settings:

```yaml
model_config:
  "llama2-7b":
    pii_policy:
      allow_by_default: true    # Allow PII by default
      pii_types_allowed: ["EMAIL_ADDRESS", "PERSON"]
    preferred_endpoints: ["my_endpoint"]  # Optional: specify which endpoints can serve this model

  "gpt-4":
    pii_policy:
      allow_by_default: false
    # preferred_endpoints omitted - router will not set endpoint header
    # Useful when external load balancer handles endpoint selection
```

**Note on `preferred_endpoints`:**

- **Optional field**: If omitted, the router will not set the `x-vsr-destination-endpoint` header
- **When specified**: Router selects the best endpoint based on weights and sets the header
- **When omitted**: Upstream load balancer or service mesh handles endpoint selection
- **Validation**: Models used in categories or as `default_model` must have `preferred_endpoints` configured

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

Define how different query types are handled using the Decision-based routing system:

```yaml
# Categories define domains for classification
categories:
- name: math
- name: computer science
- name: other

# Decisions define routing logic with rules and model selection
decisions:
- name: math
  description: "Route mathematical queries"
  priority: 10
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "math"
  modelRefs:
    - model: your-model
      use_reasoning: true            # Enable reasoning for this model on math problems

- name: computer science
  description: "Route computer science queries"
  priority: 10
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "computer science"
  modelRefs:
    - model: your-model
      use_reasoning: true            # Enable reasoning for code

- name: other
  description: "Route general queries"
  priority: 5
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "other"
  modelRefs:
    - model: your-model
      use_reasoning: false           # No reasoning for general queries

default_model: your-model          # Fallback model
```

### Model-Specific Reasoning

The `use_reasoning` field is configured per model within each decision's modelRefs, allowing fine-grained control:

```yaml
decisions:
- name: math
  description: "Route mathematical queries"
  priority: 10
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "math"
  modelRefs:
    - model: gpt-oss-120b
      use_reasoning: true            # GPT-OSS-120b supports reasoning for math
    - model: phi4
      use_reasoning: false           # phi4 doesn't support reasoning mode
    - model: deepseek-v31
      use_reasoning: true            # DeepSeek supports reasoning for math
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

**Decision-Specific Reasoning Effort:**
Override the default effort level per decision:

```yaml
decisions:
- name: math
  description: "Route mathematical queries"
  priority: 10
  reasoning_effort: "high"        # Use high effort for complex math
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "math"
  modelRefs:
    - model: your-model
      use_reasoning: true           # Enable reasoning for this model

- name: general
  description: "Route general queries"
  priority: 5
  reasoning_effort: "low"         # Use low effort for general queries
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "general"
  modelRefs:
    - model: your-model
      use_reasoning: true           # Enable reasoning for this model
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
  enabled: true                   # Enable semantic caching globally
  backend_type: "memory"          # Options: "memory" or "milvus"
  similarity_threshold: 0.8       # Global default cache hit threshold
  max_entries: 1000               # Maximum cache entries
  ttl_seconds: 3600               # Cache expiration time
  eviction_policy: "fifo"         # Options: "fifo", "lru", "lfu"

# Decision-Level Cache Configuration (New)
# Override global cache settings for specific decisions
categories:
  - name: health
  - name: general_chat
  - name: troubleshooting

decisions:
  - name: health
    description: "Route health queries"
    priority: 10
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "health"
    modelRefs:
      - model: your-model
        use_reasoning: false
    plugins:
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.95  # Very strict - medical accuracy critical

  - name: general_chat
    description: "Route general chat queries"
    priority: 5
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "general_chat"
    modelRefs:
      - model: your-model
        use_reasoning: false
    plugins:
      - type: "semantic-cache"
        configuration:
          similarity_threshold: 0.75  # Relaxed for better cache hits

  - name: troubleshooting
    description: "Route troubleshooting queries"
    priority: 5
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "troubleshooting"
    modelRefs:
      - model: your-model
        use_reasoning: false
    # No cache plugin - uses global default (0.8)

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

## Category-Level Cache Configuration

**NEW**: Configure semantic cache settings at the category level for fine-grained control over caching behavior.

### Why Use Category-Level Cache Settings?

Different categories have different tolerance for semantic variations:

- **Sensitive categories** (health, psychology, law): Small word changes can have significant meaning differences. Require high similarity thresholds (0.92-0.95).
- **General categories** (chat, troubleshooting): Less sensitive to minor wording changes. Can use lower thresholds (0.75-0.82) for better cache hit rates.
- **Privacy categories**: May need caching disabled entirely for compliance or security reasons.

### Configuration Examples

#### Example 1: Mixed Thresholds for Different Decisions

```yaml
semantic_cache:
  enabled: true
  backend_type: "memory"
  similarity_threshold: 0.8  # Global default

categories:
  - name: health
  - name: psychology
  - name: general_chat
  - name: troubleshooting

decisions:
  - name: health
    description: "Route health queries"
    priority: 10
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "health"
    modelRefs:
      - model: your-model
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          enabled: true
          system_prompt: "You are a health expert..."
          mode: "replace"
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.95  # Very strict - "headache" vs "severe headache" = different

  - name: psychology
    description: "Route psychology queries"
    priority: 10
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "psychology"
    modelRefs:
      - model: your-model
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          enabled: true
          system_prompt: "You are a psychology expert..."
          mode: "replace"
      - type: "semantic-cache"
        configuration:
          similarity_threshold: 0.92  # Strict - clinical nuances matter

  - name: general_chat
    description: "Route general chat queries"
    priority: 5
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "general_chat"
    modelRefs:
      - model: your-model
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          enabled: true
          system_prompt: "You are a helpful assistant..."
          mode: "replace"
      - type: "semantic-cache"
        configuration:
          similarity_threshold: 0.75  # Relaxed - "how's the weather" = "what's the weather"

  - name: troubleshooting
    description: "Route troubleshooting queries"
    priority: 5
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "troubleshooting"
    modelRefs:
      - model: your-model
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          enabled: true
          system_prompt: "You are a tech support expert..."
          mode: "replace"
    # No cache plugin - uses global threshold of 0.8
```

#### Example 2: Disable Cache for Sensitive Data

```yaml
categories:
  - name: personal_data

decisions:
  - name: personal_data
    description: "Route personal data queries"
    priority: 10
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "personal_data"
    modelRefs:
      - model: your-model
        use_reasoning: false
    plugins:
      - type: "system_prompt"
        configuration:
          enabled: true
          system_prompt: "Handle personal information..."
          mode: "replace"
      - type: "semantic-cache"
        configuration:
          enabled: false  # Disable cache entirely for privacy
```

### Configuration Options

**Decision-Level Plugin Fields:**

- `plugins[].type: "semantic-cache"` - Semantic cache plugin configuration
  - `configuration.enabled` (optional, boolean): Enable/disable caching for this decision. If not specified, inherits from global `semantic_cache.enabled`.
  - `configuration.similarity_threshold` (optional, float 0.0-1.0): Minimum similarity score for cache hits in this decision. If not specified, inherits from global `semantic_cache.similarity_threshold`.

**Fallback Hierarchy:**

1. Decision-specific plugin `similarity_threshold` (if set)
2. Global `semantic_cache.similarity_threshold` (if set)
3. `bert_model.threshold` (final fallback)

### Best Practices

**Threshold Selection:**

- **High precision (0.92-0.95)**: health, psychology, law, finance
- **Medium precision (0.85-0.90)**: technical documentation, education
- **Lower precision (0.75-0.82)**: general chat, FAQs, troubleshooting

**Privacy and Compliance:**

- Disable caching (set plugin `enabled: false`) for decisions handling:
  - Personal identifiable information (PII)
  - Financial data
  - Health records
  - Sensitive business information

**Performance Tuning:**

- Start with conservative (higher) thresholds
- Monitor cache hit rates per decision
- Lower thresholds for decisions with low hit rates
- Raise thresholds for decisions with incorrect cache hits

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
    address: "192.168.1.10"  # Math server IP
    port: 8000
    weight: 1
  - name: "general_endpoint"
    address: "192.168.1.20"  # General server IP
    port: 8000
    weight: 1

categories:
- name: math
- name: other

decisions:
- name: math
  description: "Route mathematical queries"
  priority: 10
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "math"
  modelRefs:
    - model: math-model
      use_reasoning: true           # Enable reasoning for math

- name: other
  description: "Route general queries"
  priority: 5
  rules:
    operator: "OR"
    conditions:
      - type: "domain"
        name: "other"
  modelRefs:
    - model: general-model
      use_reasoning: false          # No reasoning for general queries
```

**Load Balancing:**

```yaml
vllm_endpoints:
  - name: "endpoint1"
    address: "192.168.1.30"  # Primary server IP
    port: 8000
    weight: 2              # Higher weight = more traffic
  - name: "endpoint2"
    address: "192.168.1.31"  # Secondary server IP
    port: 8000
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
