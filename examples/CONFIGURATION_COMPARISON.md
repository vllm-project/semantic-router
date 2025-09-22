# Hybrid Routing Configuration Comparison

## Before: Model-Only Routing (Black Box)

```yaml
# Original semantic router - limited interpretability
categories:
  - name: math
    model_scores:
      - model: openai/gpt-oss-20b
        score: 0.9
        use_reasoning: true

default_model: openai/gpt-oss-20b

# Problems:
# - No visibility into routing decisions
# - Cannot customize routing logic beyond categories
# - No threshold control per use case
# - No request blocking capabilities
# - No explanation of why a model was selected
```

## After: Hybrid Routing (Interpretable & Configurable)

```yaml
# New hybrid approach - full control and transparency
routing_strategy:
  type: "hybrid"
  model_routing:
    enabled: true
    confidence_threshold: 0.7
  rule_routing:
    enabled: true
    fallback_to_model: true

routing_rules:
  - name: "enterprise-math-routing"
    description: "Route complex math to specialized model"
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

  - name: "premium-user-routing"
    description: "Premium users get best models"
    enabled: true
    priority: 90
    
    conditions:
      - type: "request_header"
        header_name: "x-user-tier"
        value: "premium"
        operator: "equals"
    
    actions:
      - type: "route_to_model"
        model: "premium-model"

  - name: "content-filter"
    description: "Block inappropriate content"
    enabled: true
    priority: 150
    
    conditions:
      - type: "pattern_match"
        pattern_match: "inappropriate"
        operator: "contains"
    
    actions:
      - type: "block_request"
        block_with_message: "Content violates policy"

# Benefits:
# ✅ Full transparency: Know exactly why each decision was made
# ✅ Custom logic: Business rules beyond ML categories  
# ✅ Configurable thresholds: Fine-tune sensitivity per use case
# ✅ Request blocking: Security and policy enforcement
# ✅ Rule precedence: Control decision priority
# ✅ Real-time updates: Modify rules without restart
# ✅ Audit trail: Detailed decision explanations
```

## Decision Explanation Example

```json
{
  "rule_matched": true,
  "selected_model": "math-specialized-model",
  "use_reasoning": true,
  "reasoning_effort": "high",
  "explanation": {
    "decision_type": "rule_based",
    "rule_name": "enterprise-math-routing",
    "matched_conditions": [
      {
        "condition_type": "pattern_match",
        "matched": true,
        "details": "Pattern 'math' found in content"
      },
      {
        "condition_type": "content_complexity", 
        "matched": true,
        "actual_value": 15,
        "threshold": 50,
        "details": "token_count: 15 > 50"
      }
    ],
    "executed_actions": [
      {
        "action_type": "route_to_model",
        "executed": true,
        "details": "Routed to model: math-specialized-model"
      }
    ],
    "reasoning": "Rule 'enterprise-math-routing' matched based on content analysis",
    "confidence": 0.95
  },
  "evaluation_time_ms": 2
}
```