package main

import (
	"context"
	"fmt"
	"log"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/rules"
)

func main() {
	fmt.Println("=== Semantic Router - Hybrid Routing Demonstration ===")

	// Create example configuration with routing rules
	cfg := &config.RouterConfig{
		DefaultModel:            "default-model",
		DefaultReasoningEffort:  "medium",
		RoutingStrategy: config.RoutingStrategyConfig{
			Type: "hybrid",
			ModelRouting: config.ModelRoutingConfig{
				Enabled:             true,
				FallbackToRules:     false,
				ConfidenceThreshold: 0.7,
			},
			RuleRouting: config.RuleRoutingConfig{
				Enabled:             true,
				FallbackToModel:     true,
				EvaluationTimeoutMs: 100,
			},
		},
		RoutingRules: []config.RoutingRule{
			{
				Name:        "math-specialization",
				Description: "Route math problems to specialized model",
				Enabled:     true,
				Priority:    100,
				Conditions: []config.RuleCondition{
					{
						Type:         "pattern_match",
						PatternMatch: "math",
						Operator:     "contains",
					},
					{
						Type:      "content_complexity",
						Metric:    "token_count",
						Threshold: 5,
						Operator:  "gte",
					},
				},
				Actions: []config.RuleAction{
					{
						Type:  "route_to_model",
						Model: "math-specialized-model",
					},
					{
						Type:            "enable_reasoning",
						EnableReasoning: true,
						ReasoningEffort: "high",
					},
				},
			},
			{
				Name:        "premium-user",
				Description: "Route premium users to best models",
				Enabled:     true,
				Priority:    90,
				Conditions: []config.RuleCondition{
					{
						Type:       "request_header",
						HeaderName: "x-user-tier",
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
			{
				Name:        "content-filter",
				Description: "Block inappropriate content",
				Enabled:     true,
				Priority:    150,
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
						BlockWithMessage: "Content violates usage policy",
					},
				},
			},
		},
	}

	// Create hybrid router (without classifier for demo)
	hybridRouter := rules.NewHybridRouter(cfg, nil)

	fmt.Printf("Created hybrid router with %d rules\n\n", hybridRouter.GetRuleCount())

	// Test scenarios
	testScenarios := []struct {
		name    string
		content string
		headers map[string]string
	}{
		{
			name:    "Math Problem",
			content: "solve this complex math equation: 2x + 3y = 10",
			headers: map[string]string{},
		},
		{
			name:    "Premium User Request",
			content: "write a story about cats",
			headers: map[string]string{"x-user-tier": "premium"},
		},
		{
			name:    "Blocked Content",
			content: "this contains forbidden content",
			headers: map[string]string{},
		},
		{
			name:    "Simple Query",
			content: "hi",
			headers: map[string]string{},
		},
		{
			name:    "Long Math Problem (Premium User)",
			content: "solve this very complex mathematical proof involving advanced calculus and linear algebra",
			headers: map[string]string{"x-user-tier": "premium"},
		},
	}

	for i, scenario := range testScenarios {
		fmt.Printf("--- Test %d: %s ---\n", i+1, scenario.name)
		fmt.Printf("Content: %s\n", scenario.content)
		fmt.Printf("Headers: %v\n", scenario.headers)

		// Route the request
		decision, err := hybridRouter.RouteRequest(
			context.Background(),
			scenario.content,
			nil,
			scenario.headers,
			"auto",
		)

		if err != nil {
			log.Printf("Error routing request: %v", err)
			continue
		}

		// Display results
		fmt.Printf("\n🎯 Routing Decision:\n")
		if decision.RuleMatched {
			fmt.Printf("  ✅ Rule Matched: %s\n", decision.MatchedRule.Name)
			fmt.Printf("  📝 Rule Description: %s\n", decision.MatchedRule.Description)
		} else {
			fmt.Printf("  📊 Model-based routing (no rules matched)\n")
		}

		fmt.Printf("  🚀 Selected Model: %s\n", decision.SelectedModel)
		fmt.Printf("  🧠 Use Reasoning: %v\n", decision.UseReasoning)
		if decision.UseReasoning {
			fmt.Printf("  💪 Reasoning Effort: %s\n", decision.ReasoningEffort)
		}

		if decision.BlockRequest {
			fmt.Printf("  🚫 Request Blocked: %s\n", decision.BlockMessage)
		}

		fmt.Printf("  ⏱️  Evaluation Time: %d ms\n", decision.EvaluationTimeMs)
		fmt.Printf("  🔍 Decision Type: %s\n", decision.Explanation.DecisionType)
		fmt.Printf("  💡 Reasoning: %s\n", decision.Explanation.Reasoning)

		if decision.RuleMatched && len(decision.Explanation.MatchedConditions) > 0 {
			fmt.Printf("  📋 Matched Conditions:\n")
			for _, condition := range decision.Explanation.MatchedConditions {
				status := "❌"
				if condition.Matched {
					status = "✅"
				}
				fmt.Printf("    %s %s: %s\n", status, condition.ConditionType, condition.Details)
			}
		}

		if decision.RuleMatched && len(decision.Explanation.ExecutedActions) > 0 {
			fmt.Printf("  ⚡ Executed Actions:\n")
			for _, action := range decision.Explanation.ExecutedActions {
				status := "❌"
				if action.Executed {
					status = "✅"
				}
				fmt.Printf("    %s %s: %s\n", status, action.ActionType, action.Details)
			}
		}

		fmt.Println()
	}

	fmt.Println("=== Demonstration Complete ===")
}