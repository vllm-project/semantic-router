package rules

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRuleEngine_BasicEvaluation(t *testing.T) {
	// Create test rules
	rules := []config.RoutingRule{
		{
			Name:        "math-rule",
			Description: "Route math problems to math model",
			Enabled:     true,
			Priority:    100,
			Conditions: []config.RuleCondition{
				{
					Type:      "pattern_match",
					PatternMatch: "math",
					Operator:  "contains",
				},
			},
			Actions: []config.RuleAction{
				{
					Type:  "route_to_model",
					Model: "math-model",
				},
			},
		},
	}

	// Create test config
	cfg := &config.RouterConfig{
		DefaultModel: "default-model",
	}

	// Create rule engine
	engine := NewRuleEngine(rules, nil, cfg)

	// Create evaluation context
	evalCtx := &EvaluationContext{
		AllContent: "solve this math problem",
		Headers:    make(map[string]string),
		Timestamp:  time.Now(),
	}

	// Evaluate rules
	decision, err := engine.EvaluateRules(context.Background(), evalCtx)
	if err != nil {
		t.Fatalf("Rule evaluation failed: %v", err)
	}

	// Check results
	if !decision.RuleMatched {
		t.Error("Expected rule to match")
	}

	if decision.SelectedModel != "math-model" {
		t.Errorf("Expected model 'math-model', got '%s'", decision.SelectedModel)
	}

	if decision.MatchedRule.Name != "math-rule" {
		t.Errorf("Expected rule 'math-rule', got '%s'", decision.MatchedRule.Name)
	}
}

func TestRuleEngine_ContentComplexity(t *testing.T) {
	// Create rule with content complexity condition
	rules := []config.RoutingRule{
		{
			Name:     "long-content-rule",
			Enabled:  true,
			Priority: 100,
			Conditions: []config.RuleCondition{
				{
					Type:      "content_complexity",
					Metric:    "token_count",
					Threshold: 5,
					Operator:  "gt",
				},
			},
			Actions: []config.RuleAction{
				{
					Type:  "route_to_model",
					Model: "complex-model",
				},
			},
		},
	}

	cfg := &config.RouterConfig{DefaultModel: "default-model"}
	engine := NewRuleEngine(rules, nil, cfg)

	// Test with long content (should match)
	evalCtx := &EvaluationContext{
		AllContent: "this is a very long piece of content that should trigger the rule",
		Headers:    make(map[string]string),
		Timestamp:  time.Now(),
	}

	decision, err := engine.EvaluateRules(context.Background(), evalCtx)
	if err != nil {
		t.Fatalf("Rule evaluation failed: %v", err)
	}

	if !decision.RuleMatched {
		t.Error("Expected rule to match for long content")
	}

	// Test with short content (should not match)
	evalCtx.AllContent = "short content"
	decision, err = engine.EvaluateRules(context.Background(), evalCtx)
	if err != nil {
		t.Fatalf("Rule evaluation failed: %v", err)
	}

	if decision.RuleMatched {
		t.Error("Expected rule not to match for short content")
	}
}

func TestRuleEngine_HeaderConditions(t *testing.T) {
	rules := []config.RoutingRule{
		{
			Name:     "api-key-rule",
			Enabled:  true,
			Priority: 100,
			Conditions: []config.RuleCondition{
				{
					Type:       "request_header",
					HeaderName: "x-api-key",
					Value:      "premium",
					Operator:   "equals",
				},
			},
			Actions: []config.RuleAction{
				{
					Type:  "route_to_model",
					Model: "premium-model",
				},
			},
		},
	}

	cfg := &config.RouterConfig{DefaultModel: "default-model"}
	engine := NewRuleEngine(rules, nil, cfg)

	// Test with matching header
	evalCtx := &EvaluationContext{
		AllContent: "test content",
		Headers:    map[string]string{"x-api-key": "premium"},
		Timestamp:  time.Now(),
	}

	decision, err := engine.EvaluateRules(context.Background(), evalCtx)
	if err != nil {
		t.Fatalf("Rule evaluation failed: %v", err)
	}

	if !decision.RuleMatched {
		t.Error("Expected rule to match for premium API key")
	}

	// Test with wrong header value
	evalCtx.Headers["x-api-key"] = "basic"
	decision, err = engine.EvaluateRules(context.Background(), evalCtx)
	if err != nil {
		t.Fatalf("Rule evaluation failed: %v", err)
	}

	if decision.RuleMatched {
		t.Error("Expected rule not to match for basic API key")
	}
}

func TestRuleEngine_MultipleConditions(t *testing.T) {
	rules := []config.RoutingRule{
		{
			Name:     "complex-rule",
			Enabled:  true,
			Priority: 100,
			Conditions: []config.RuleCondition{
				{
					Type:         "pattern_match",
					PatternMatch: "math",
					Operator:     "contains",
				},
				{
					Type:      "content_complexity",
					Metric:    "token_count",
					Threshold: 3,
					Operator:  "gte",
				},
			},
			Actions: []config.RuleAction{
				{
					Type:  "route_to_model",
					Model: "advanced-math-model",
				},
			},
		},
	}

	cfg := &config.RouterConfig{DefaultModel: "default-model"}
	engine := NewRuleEngine(rules, nil, cfg)

	// Test content that matches both conditions
	evalCtx := &EvaluationContext{
		AllContent: "solve this complex math problem",
		Headers:    make(map[string]string),
		Timestamp:  time.Now(),
	}

	decision, err := engine.EvaluateRules(context.Background(), evalCtx)
	if err != nil {
		t.Fatalf("Rule evaluation failed: %v", err)
	}

	if !decision.RuleMatched {
		t.Error("Expected rule to match when all conditions are met")
	}

	// Test content that matches only one condition
	evalCtx.AllContent = "math"
	decision, err = engine.EvaluateRules(context.Background(), evalCtx)
	if err != nil {
		t.Fatalf("Rule evaluation failed: %v", err)
	}

	if decision.RuleMatched {
		t.Error("Expected rule not to match when only one condition is met")
	}
}

func TestRuleEngine_RulePriority(t *testing.T) {
	rules := []config.RoutingRule{
		{
			Name:     "low-priority-rule",
			Enabled:  true,
			Priority: 50,
			Conditions: []config.RuleCondition{
				{
					Type:         "pattern_match",
					PatternMatch: "test",
					Operator:     "contains",
				},
			},
			Actions: []config.RuleAction{
				{
					Type:  "route_to_model",
					Model: "low-priority-model",
				},
			},
		},
		{
			Name:     "high-priority-rule",
			Enabled:  true,
			Priority: 100,
			Conditions: []config.RuleCondition{
				{
					Type:         "pattern_match",
					PatternMatch: "test",
					Operator:     "contains",
				},
			},
			Actions: []config.RuleAction{
				{
					Type:  "route_to_model",
					Model: "high-priority-model",
				},
			},
		},
	}

	cfg := &config.RouterConfig{DefaultModel: "default-model"}
	engine := NewRuleEngine(rules, nil, cfg)

	evalCtx := &EvaluationContext{
		AllContent: "test content",
		Headers:    make(map[string]string),
		Timestamp:  time.Now(),
	}

	decision, err := engine.EvaluateRules(context.Background(), evalCtx)
	if err != nil {
		t.Fatalf("Rule evaluation failed: %v", err)
	}

	if !decision.RuleMatched {
		t.Error("Expected a rule to match")
	}

	// Should match the high priority rule first
	if decision.SelectedModel != "high-priority-model" {
		t.Errorf("Expected high priority model, got '%s'", decision.SelectedModel)
	}
}

func TestRuleEngine_BlockAction(t *testing.T) {
	rules := []config.RoutingRule{
		{
			Name:     "block-rule",
			Enabled:  true,
			Priority: 100,
			Conditions: []config.RuleCondition{
				{
					Type:         "pattern_match",
					PatternMatch: "forbidden",
					Operator:     "contains",
				},
			},
			Actions: []config.RuleAction{
				{
					Type:             "block_request",
					BlockWithMessage: "Content contains forbidden patterns",
				},
			},
		},
	}

	cfg := &config.RouterConfig{DefaultModel: "default-model"}
	engine := NewRuleEngine(rules, nil, cfg)

	evalCtx := &EvaluationContext{
		AllContent: "this contains forbidden content",
		Headers:    make(map[string]string),
		Timestamp:  time.Now(),
	}

	decision, err := engine.EvaluateRules(context.Background(), evalCtx)
	if err != nil {
		t.Fatalf("Rule evaluation failed: %v", err)
	}

	if !decision.RuleMatched {
		t.Error("Expected rule to match")
	}

	if !decision.BlockRequest {
		t.Error("Expected request to be blocked")
	}

	if decision.BlockMessage != "Content contains forbidden patterns" {
		t.Errorf("Expected block message 'Content contains forbidden patterns', got '%s'", decision.BlockMessage)
	}
}