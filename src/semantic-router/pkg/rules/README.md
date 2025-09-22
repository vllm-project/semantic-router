# Hybrid Routing Rules System

This package implements a configurable and interpretable routing rules system that extends the semantic router to support both model-based and rule-based routing approaches.

## Features

### Core Capabilities
- **Hybrid routing approach**: Support both model-based classification AND user-defined rules
- **Transparent decision-making**: Every routing decision provides a clear explanation of which rules fired and why
- **User-defined rules**: Ability to create custom routing logic with multiple condition types
- **Configurable thresholds**: Full control over classification sensitivity and decision boundaries
- **Rule precedence**: Ability to define when rules take precedence over model classification
- **Real-time evaluation**: Rules are evaluated in real-time without service restart

### Supported Rule Conditions

#### 1. Category Classification
```yaml
- type: "category_classification"
  category: "math"
  threshold: 0.8
  operator: "gte"
```

#### 2. Content Complexity
```yaml
- type: "content_complexity"
  metric: "token_count"  # or "character_count", "line_count"
  threshold: 50
  operator: "gt"
```

#### 3. Request Headers
```yaml
- type: "request_header"
  header_name: "x-user-tier"
  value: "premium"
  operator: "equals"
```

#### 4. Pattern Matching
```yaml
- type: "pattern_match"
  pattern_match: "math"
  operator: "contains"
```

#### 5. Time-based Conditions
```yaml
- type: "time_based"
  time_range: "business_hours"
```

### Supported Rule Actions

#### 1. Route to Model
```yaml
- type: "route_to_model"
  model: "math-specialized-model"
```

#### 2. Enable Reasoning
```yaml
- type: "enable_reasoning"
  enable_reasoning: true
  reasoning_effort: "high"
```

#### 3. Set Headers
```yaml
- type: "set_headers"
  headers:
    x-routing-decision: "rule-based"
    x-model-tier: "premium"
```

#### 4. Block Request
```yaml
- type: "block_request"
  block_with_message: "Content violates usage policy"
```

## Configuration Example

```yaml
# Hybrid Routing Configuration
routing_strategy:
  type: "hybrid"  # Options: "model", "rules", "hybrid"
  
  model_routing:
    enabled: true
    fallback_to_rules: false
    confidence_threshold: 0.7
  
  rule_routing:
    enabled: true
    fallback_to_model: true
    evaluation_timeout_ms: 100

# Custom Routing Rules
routing_rules:
  - name: "enterprise-math-routing"
    description: "Route complex math problems to specialized model"
    enabled: true
    priority: 100
    
    conditions:
      - type: "category_classification"
        category: "math"
        threshold: 0.8
        operator: "gte"
      - type: "content_complexity"
        metric: "token_count"
        threshold: 50
        operator: "gt"
    
    actions:
      - type: "route_to_model"
        model: "math-specialized-model"
      - type: "enable_reasoning"
        enable_reasoning: true
        reasoning_effort: "high"
    
    evaluation:
      timeout_ms: 100
      fallback_action: "use_model_classification"
```

## API Endpoints

The rule management API provides the following endpoints:

### Rule Management
- `GET /api/v1/rules` - List all rules
- `POST /api/v1/rules` - Create new rule
- `GET /api/v1/rules/{name}` - Get specific rule
- `PUT /api/v1/rules/{name}` - Update rule
- `DELETE /api/v1/rules/{name}` - Delete rule

### Rule Evaluation and Debugging
- `POST /api/v1/rules/evaluate` - Evaluate rules for request
- `GET /api/v1/rules/explain/{id}` - Get decision explanation
- `POST /api/v1/rules/test` - Test rule with sample data

## Usage

### Basic Usage

```go
// Create hybrid router
hybridRouter := rules.NewHybridRouter(config, classifier)

// Route a request
decision, err := hybridRouter.RouteRequest(
    ctx,
    userContent,
    nonUserContent,
    headers,
    originalModel,
)

// Check decision
if decision.RuleMatched {
    fmt.Printf("Rule matched: %s\n", decision.MatchedRule.Name)
    fmt.Printf("Selected model: %s\n", decision.SelectedModel)
    fmt.Printf("Use reasoning: %v\n", decision.UseReasoning)
}
```

### Decision Explanation

Every routing decision includes detailed explanations:

```go
type DecisionExplanation struct {
    DecisionType      string                     // "rule_based", "model_based", "fallback"
    RuleName          string                     // Name of matched rule
    MatchedConditions []ConditionResult          // Details of condition evaluation
    ExecutedActions   []ActionResult             // Details of action execution
    Reasoning         string                     // Human-readable explanation
    Confidence        float64                    // Confidence score
}
```

## Testing

Run the comprehensive test suite:

```bash
cd src/semantic-router
LD_LIBRARY_PATH=../../candle-binding/target/release go test ./pkg/rules -v
```

Run the interactive demonstration:

```bash
cd src/semantic-router
LD_LIBRARY_PATH=../../candle-binding/target/release go run ../../examples/hybrid-routing-demo.go
```

## Architecture

The hybrid routing system consists of three main components:

1. **RuleEngine**: Evaluates routing rules and conditions
2. **HybridRouter**: Orchestrates rule-based and model-based routing
3. **RuleManagementAPI**: Provides HTTP endpoints for rule management

The system maintains backward compatibility with existing model-based routing while adding powerful rule-based capabilities for interpretable and configurable routing decisions.