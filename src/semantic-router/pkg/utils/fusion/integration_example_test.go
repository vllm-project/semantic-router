package fusion_test

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/fusion"
)

// This file demonstrates how to integrate the Signal Fusion Engine with the router

// Example showing end-to-end integration with the router
func Example_integration() {
	// Step 1: Create a fusion policy from configuration
	policy := &fusion.Policy{
		Rules: []fusion.Rule{
			{
				Name:      "safety-block",
				Condition: "regex.ssn.matched || regex.credit-card.matched",
				Action:    fusion.ActionBlock,
				Priority:  200,
				Message:   "PII detected - request blocked",
			},
			{
				Name:      "k8s-routing",
				Condition: "keyword.kubernetes.matched && similarity.infrastructure.score > 0.75",
				Action:    fusion.ActionRoute,
				Priority:  150,
				Models:    []string{"k8s-expert", "devops-model"},
			},
			{
				Name:      "boost-reasoning",
				Condition: "similarity.reasoning.score > 0.75",
				Action:    fusion.ActionBoostCategory,
				Priority:  100,
				Category:  "reasoning",
				BoostWeight: 1.5,
			},
			{
				Name:      "default-fallthrough",
				Condition: "!regex.ssn.matched",
				Action:    fusion.ActionFallthrough,
				Priority:  0,
			},
		},
	}

	// Step 2: Initialize the fusion engine
	engine := fusion.NewEngine(policy)

	// Step 3: Simulate gathering signals from various providers
	// In real implementation, these would come from:
	// - Keyword matcher scanning the query
	// - Regex scanner looking for patterns
	// - Embedding similarity comparing to concepts
	// - BERT classifier categorizing the query

	context := fusion.NewSignalContext()

	// Keyword signal (from keyword matcher)
	context.AddSignal(fusion.Signal{
		Provider: "keyword",
		Name:     "kubernetes",
		Matched:  true,
	})

	// Similarity signal (from embedding similarity)
	context.AddSignal(fusion.Signal{
		Provider: "similarity",
		Name:     "infrastructure",
		Score:    0.85,
		Matched:  true,
	})

	// Regex signal (from regex scanner)
	context.AddSignal(fusion.Signal{
		Provider: "regex",
		Name:     "ssn",
		Matched:  false,
	})

	// BERT signal (from existing classifier)
	context.AddSignal(fusion.Signal{
		Provider: "bert",
		Name:     "category",
		Value:    "computer science",
		Score:    0.92,
		Matched:  true,
	})

	// Step 4: Evaluate the policy
	result, _ := engine.Evaluate(context)

	// Step 5: Handle the result
	fmt.Printf("Rule matched: %s\n", result.MatchedRule)
	fmt.Printf("Action: %s\n", result.Action)
	fmt.Printf("Route to models: %v\n", result.Models)

	// Output:
	// Rule matched: k8s-routing
	// Action: route
	// Route to models: [k8s-expert devops-model]
}

// Example showing how to handle different action types
func Example_actionHandling() {
	// Policy with various action types
	policy := &fusion.Policy{
		Rules: []fusion.Rule{
			{
				Name:      "block-rule",
				Condition: "regex.pii.matched",
				Action:    fusion.ActionBlock,
				Priority:  200,
				Message:   "Blocked due to PII",
			},
			{
				Name:      "route-rule",
				Condition: "keyword.topic.matched",
				Action:    fusion.ActionRoute,
				Priority:  150,
				Models:    []string{"specialist-model"},
			},
			{
				Name:      "boost-rule",
				Condition: "similarity.concept.score > 0.8",
				Action:    fusion.ActionBoostCategory,
				Priority:  100,
				Category:  "science",
				BoostWeight: 1.5,
			},
			{
				Name:      "fallthrough-rule",
				Condition: "!regex.pii.matched",
				Action:    fusion.ActionFallthrough,
				Priority:  0,
			},
		},
	}

	engine := fusion.NewEngine(policy)

	// Test Case 1: Block action
	fmt.Println("=== Test Case 1: Block Action ===")
	ctx1 := fusion.NewSignalContext()
	ctx1.AddSignal(fusion.Signal{
		Provider: "regex",
		Name:     "pii",
		Matched:  true,
	})

	result1, _ := engine.Evaluate(ctx1)
	if result1.Action == fusion.ActionBlock {
		fmt.Printf("Blocked: %s\n", result1.Message)
	}

	// Test Case 2: Route action
	fmt.Println("\n=== Test Case 2: Route Action ===")
	ctx2 := fusion.NewSignalContext()
	ctx2.AddSignal(fusion.Signal{
		Provider: "keyword",
		Name:     "topic",
		Matched:  true,
	})
	ctx2.AddSignal(fusion.Signal{
		Provider: "regex",
		Name:     "pii",
		Matched:  false,
	})

	result2, _ := engine.Evaluate(ctx2)
	if result2.Action == fusion.ActionRoute {
		fmt.Printf("Route to: %v\n", result2.Models)
	}

	// Test Case 3: Boost action
	fmt.Println("\n=== Test Case 3: Boost Action ===")
	ctx3 := fusion.NewSignalContext()
	ctx3.AddSignal(fusion.Signal{
		Provider: "similarity",
		Name:     "concept",
		Score:    0.85,
		Matched:  true,
	})
	ctx3.AddSignal(fusion.Signal{
		Provider: "regex",
		Name:     "pii",
		Matched:  false,
	})

	result3, _ := engine.Evaluate(ctx3)
	if result3.Action == fusion.ActionBoostCategory {
		fmt.Printf("Boost %s by %.1fx\n", result3.Category, result3.BoostWeight)
	}

	// Test Case 4: Fallthrough action
	fmt.Println("\n=== Test Case 4: Fallthrough Action ===")
	ctx4 := fusion.NewSignalContext()
	ctx4.AddSignal(fusion.Signal{
		Provider: "regex",
		Name:     "pii",
		Matched:  false,
	})

	result4, _ := engine.Evaluate(ctx4)
	if result4.Action == fusion.ActionFallthrough {
		fmt.Println("Fallthrough to BERT classification")
	}

	// Output:
	// === Test Case 1: Block Action ===
	// Blocked: Blocked due to PII
	//
	// === Test Case 2: Route Action ===
	// Route to: [specialist-model]
	//
	// === Test Case 3: Boost Action ===
	// Boost science by 1.5x
	//
	// === Test Case 4: Fallthrough Action ===
	// Fallthrough to BERT classification
}

// Example showing how to load policy from configuration struct
func Example_policyFromConfig() {
	// Simulate loading from YAML config
	// In real implementation, this would be unmarshaled from config.yaml
	type ConfigRule struct {
		Name        string
		Condition   string
		Action      string
		Priority    int
		Models      []string
		Category    string
		BoostWeight float64
		Message     string
	}

	configRules := []ConfigRule{
		{
			Name:      "safety-check",
			Condition: "regex.ssn.matched",
			Action:    "block",
			Priority:  200,
			Message:   "SSN detected",
		},
		{
			Name:      "expert-routing",
			Condition: "keyword.kubernetes.matched && similarity.infra.score > 0.8",
			Action:    "route",
			Priority:  150,
			Models:    []string{"k8s-expert"},
		},
	}

	// Convert config rules to fusion rules
	policy := &fusion.Policy{
		Rules: make([]fusion.Rule, 0, len(configRules)),
	}

	for _, cfgRule := range configRules {
		rule := fusion.Rule{
			Name:        cfgRule.Name,
			Condition:   cfgRule.Condition,
			Action:      fusion.ActionType(cfgRule.Action),
			Priority:    cfgRule.Priority,
			Models:      cfgRule.Models,
			Category:    cfgRule.Category,
			BoostWeight: cfgRule.BoostWeight,
			Message:     cfgRule.Message,
		}
		policy.Rules = append(policy.Rules, rule)
	}

	// Create engine
	engine := fusion.NewEngine(policy)

	// Test it
	ctx := fusion.NewSignalContext()
	ctx.AddSignal(fusion.Signal{
		Provider: "keyword",
		Name:     "kubernetes",
		Matched:  true,
	})
	ctx.AddSignal(fusion.Signal{
		Provider: "similarity",
		Name:     "infra",
		Score:    0.85,
		Matched:  true,
	})

	result, _ := engine.Evaluate(ctx)
	fmt.Printf("Matched: %s\n", result.MatchedRule)
	fmt.Printf("Models: %v\n", result.Models)

	// Output:
	// Matched: expert-routing
	// Models: [k8s-expert]
}

// Example showing signal provider integration
func Example_signalProviders() {
	// This shows how different providers contribute signals

	// Simulated provider functions (in real implementation, these would be actual providers)
	detectKeywords := func(query string) []fusion.Signal {
		// Keyword matcher scans query for configured keywords
		return []fusion.Signal{
			{
				Provider: "keyword",
				Name:     "kubernetes",
				Matched:  true, // Found "kubernetes" in query
			},
			{
				Provider: "keyword",
				Name:     "security",
				Matched:  true, // Found "security" in query
			},
		}
	}

	scanRegex := func(query string) []fusion.Signal {
		// Regex scanner looks for patterns
		return []fusion.Signal{
			{
				Provider: "regex",
				Name:     "ssn",
				Matched:  false, // No SSN pattern found
			},
			{
				Provider: "regex",
				Name:     "cve-id",
				Matched:  true, // Found CVE pattern
			},
		}
	}

	computeSimilarity := func(query string) []fusion.Signal {
		// Embedding similarity compares to concepts
		return []fusion.Signal{
			{
				Provider: "similarity",
				Name:     "infrastructure",
				Score:    0.87,
				Matched:  true,
			},
			{
				Provider: "similarity",
				Name:     "security",
				Score:    0.82,
				Matched:  true,
			},
		}
	}

	classifyBERT := func(query string) fusion.Signal {
		// BERT classifier (existing)
		return fusion.Signal{
			Provider: "bert",
			Name:     "category",
			Value:    "computer science",
			Score:    0.91,
			Matched:  true,
		}
	}

	// Gather all signals for a query
	query := "How to secure a Kubernetes cluster against CVE-2024-1234?"

	ctx := fusion.NewSignalContext()

	// Add signals from all providers
	for _, sig := range detectKeywords(query) {
		ctx.AddSignal(sig)
	}
	for _, sig := range scanRegex(query) {
		ctx.AddSignal(sig)
	}
	for _, sig := range computeSimilarity(query) {
		ctx.AddSignal(sig)
	}
	ctx.AddSignal(classifyBERT(query))

	// Now evaluate with a policy
	policy := &fusion.Policy{
		Rules: []fusion.Rule{
			{
				Name:      "security-k8s-expert",
				Condition: "keyword.kubernetes.matched && keyword.security.matched && regex.cve-id.matched",
				Action:    fusion.ActionRoute,
				Priority:  150,
				Models:    []string{"k8s-security-expert"},
			},
		},
	}

	engine := fusion.NewEngine(policy)
	result, _ := engine.Evaluate(ctx)

	fmt.Printf("Query: %s\n", query)
	fmt.Printf("Matched rule: %s\n", result.MatchedRule)
	fmt.Printf("Route to: %v\n", result.Models)

	// Output:
	// Query: How to secure a Kubernetes cluster against CVE-2024-1234?
	// Matched rule: security-k8s-expert
	// Route to: [k8s-security-expert]
}
