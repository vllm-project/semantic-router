package fusion

import (
	"fmt"
)

// Example demonstrates basic Signal Fusion Engine usage
func Example() {
	// Create a signal context
	context := NewSignalContext()

	// Add signals from various providers
	context.AddSignal(Signal{
		Provider: "keyword",
		Name:     "kubernetes",
		Matched:  true,
	})

	context.AddSignal(Signal{
		Provider: "similarity",
		Name:     "infrastructure",
		Score:    0.85,
		Matched:  true,
	})

	// Define a simple policy
	policy := &Policy{
		Rules: []Rule{
			{
				Name:      "k8s-routing",
				Condition: "keyword.kubernetes.matched && similarity.infrastructure.score > 0.75",
				Action:    ActionRoute,
				Priority:  150,
				Models:    []string{"k8s-expert", "devops-model"},
			},
		},
	}

	// Create engine and evaluate
	engine := NewEngine(policy)
	result, _ := engine.Evaluate(context)

	if result.Matched {
		fmt.Printf("Matched rule: %s\n", result.MatchedRule)
		fmt.Printf("Action: %s\n", result.Action)
		fmt.Printf("Models: %v\n", result.Models)
	}

	// Output:
	// Matched rule: k8s-routing
	// Action: route
	// Models: [k8s-expert devops-model]
}

// Example_priorityEvaluation demonstrates priority-based rule evaluation
func Example_priorityEvaluation() {
	context := NewSignalContext()

	// Add signals
	context.AddSignal(Signal{
		Provider: "regex",
		Name:     "ssn",
		Matched:  true,
	})

	context.AddSignal(Signal{
		Provider: "keyword",
		Name:     "kubernetes",
		Matched:  true,
	})

	// Define policy with multiple priority levels
	policy := &Policy{
		Rules: []Rule{
			{
				Name:      "safety-block",
				Condition: "regex.ssn.matched",
				Action:    ActionBlock,
				Priority:  200, // Highest priority
				Message:   "PII detected - request blocked",
			},
			{
				Name:      "k8s-routing",
				Condition: "keyword.kubernetes.matched",
				Action:    ActionRoute,
				Priority:  150,
				Models:    []string{"k8s-expert"},
			},
		},
	}

	// Safety block wins due to higher priority
	engine := NewEngine(policy)
	result, _ := engine.Evaluate(context)

	fmt.Printf("Matched rule: %s (priority: 200)\n", result.MatchedRule)
	fmt.Printf("Action: %s\n", result.Action)

	// Output:
	// Matched rule: safety-block (priority: 200)
	// Action: block
}

// Example_complexExpressions demonstrates complex boolean expressions
func Example_complexExpressions() {
	context := NewSignalContext()

	// Add multiple signals
	context.AddSignal(Signal{
		Provider: "keyword",
		Name:     "kubernetes",
		Matched:  true,
	})

	context.AddSignal(Signal{
		Provider: "keyword",
		Name:     "security",
		Matched:  true,
	})

	context.AddSignal(Signal{
		Provider: "similarity",
		Name:     "infrastructure",
		Score:    0.88,
		Matched:  true,
	})

	context.AddSignal(Signal{
		Provider: "bert",
		Name:     "category",
		Value:    "computer science",
		Matched:  true,
	})

	// Complex policy requiring consensus from multiple signals
	policy := &Policy{
		Rules: []Rule{
			{
				Name: "multi-signal-consensus",
				Condition: `keyword.kubernetes.matched && 
					        keyword.security.matched && 
					        similarity.infrastructure.score > 0.8 && 
					        bert.category.value == 'computer science'`,
				Action:   ActionRoute,
				Priority: 50,
				Models:   []string{"k8s-security-expert"},
			},
		},
	}

	engine := NewEngine(policy)
	result, _ := engine.Evaluate(context)

	if result.Matched {
		fmt.Printf("All signals agree - routing to: %v\n", result.Models)
	}

	// Output:
	// All signals agree - routing to: [k8s-security-expert]
}

// Example_boostCategory demonstrates category boosting
func Example_boostCategory() {
	context := NewSignalContext()

	context.AddSignal(Signal{
		Provider: "similarity",
		Name:     "reasoning",
		Score:    0.82,
		Matched:  true,
	})

	policy := &Policy{
		Rules: []Rule{
			{
				Name:        "boost-reasoning",
				Condition:   "similarity.reasoning.score > 0.75",
				Action:      ActionBoostCategory,
				Priority:    100,
				Category:    "reasoning",
				BoostWeight: 1.5,
			},
		},
	}

	engine := NewEngine(policy)
	result, _ := engine.Evaluate(context)

	if result.Matched {
		fmt.Printf("Boost %s category by %.1fx\n", result.Category, result.BoostWeight)
	}

	// Output:
	// Boost reasoning category by 1.5x
}

// Example_shortCircuit demonstrates short-circuit evaluation
func Example_shortCircuit() {
	context := NewSignalContext()

	context.AddSignal(Signal{
		Provider: "keyword",
		Name:     "test",
		Matched:  true,
	})

	// Multiple rules could match, but first one wins
	policy := &Policy{
		Rules: []Rule{
			{
				Name:      "first-rule",
				Condition: "keyword.test.matched",
				Action:    ActionRoute,
				Priority:  100,
				Models:    []string{"model-a"},
			},
			{
				Name:      "second-rule",
				Condition: "keyword.test.matched",
				Action:    ActionRoute,
				Priority:  50,
				Models:    []string{"model-b"},
			},
		},
	}

	engine := NewEngine(policy)
	result, _ := engine.Evaluate(context)

	// Only first rule is evaluated and returned
	fmt.Printf("Matched: %s\n", result.MatchedRule)
	fmt.Printf("Models: %v\n", result.Models)

	// Output:
	// Matched: first-rule
	// Models: [model-a]
}

// Example_fallthrough demonstrates fallthrough behavior
func Example_fallthrough() {
	context := NewSignalContext()

	// No signals match
	context.AddSignal(Signal{
		Provider: "keyword",
		Name:     "docker",
		Matched:  false,
	})

	policy := &Policy{
		Rules: []Rule{
			{
				Name:      "docker-routing",
				Condition: "keyword.docker.matched",
				Action:    ActionRoute,
				Priority:  100,
				Models:    []string{"docker-expert"},
			},
		},
	}

	engine := NewEngine(policy)
	result, _ := engine.Evaluate(context)

	if !result.Matched {
		fmt.Printf("No rules matched - action: %s\n", result.Action)
	}

	// Output:
	// No rules matched - action: fallthrough
}
