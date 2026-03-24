package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type evaluateDecisionsTestCase struct {
	name                  string
	decisions             []config.Decision
	strategy              string
	matchedKeywordRules   []string
	matchedEmbeddingRules []string
	matchedDomainRules    []string
	expectedDecision      string
	expectError           bool
}

type evaluateWithSignalsTestCase struct {
	name             string
	decisions        []config.Decision
	signals          *SignalMatches
	expectedDecision string
	expectError      bool
	assertResult     func(t *testing.T, result *DecisionResult)
}

func TestDecisionEngine_EvaluateDecisions(t *testing.T) {
	for _, tt := range evaluateDecisionsTestCases() {
		t.Run(tt.name, func(t *testing.T) {
			evaluateDecisionsCase(t, tt)
		})
	}
}

func TestDecisionEngine_EvaluateDecisionsWithFactCheck(t *testing.T) {
	for _, tt := range factCheckDecisionTestCases() {
		t.Run(tt.name, func(t *testing.T) {
			evaluateDecisionSignalsCase(t, tt, "priority")
		})
	}
}

func TestDecisionEngine_AnnotatesWinnerMarginAndRunnerUp(t *testing.T) {
	engine := NewDecisionEngine(
		[]config.KeywordRule{},
		[]config.EmbeddingRule{},
		[]config.Category{},
		[]config.Decision{
			{
				Name:     "primary",
				Priority: 20,
				Rules: config.RuleCombination{
					Operator: "OR",
					Conditions: []config.RuleCondition{
						{Type: "keyword", Name: "alpha"},
					},
				},
			},
			{
				Name:     "secondary",
				Priority: 10,
				Rules: config.RuleCombination{
					Operator: "OR",
					Conditions: []config.RuleCondition{
						{Type: "keyword", Name: "beta"},
					},
				},
			},
		},
		"confidence",
	)

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		KeywordRules: []string{"alpha", "beta"},
		SignalConfidences: map[string]float64{
			"keyword:alpha": 0.91,
			"keyword:beta":  0.73,
		},
	})
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals failed: %v", err)
	}
	if result == nil {
		t.Fatal("expected decision result, got nil")
	}
	if result.Decision.Name != "primary" {
		t.Fatalf("winner = %q, want primary", result.Decision.Name)
	}
	if result.CandidateCount != 2 {
		t.Fatalf("candidate_count = %d, want 2", result.CandidateCount)
	}
	if result.DecisionMargin < 0.179 || result.DecisionMargin > 0.181 {
		t.Fatalf("decision_margin = %.3f, want about 0.18", result.DecisionMargin)
	}
	if result.DecisionWinnerBasis != "confidence" {
		t.Fatalf("winner_basis = %q, want confidence", result.DecisionWinnerBasis)
	}
	if result.RunnerUp == nil || result.RunnerUp.Name != "secondary" {
		t.Fatalf("runner_up = %+v, want secondary", result.RunnerUp)
	}
}

func TestDecisionEngine_PrioritySelectionCanExposeNegativeConfidenceMargin(t *testing.T) {
	engine := NewDecisionEngine(
		[]config.KeywordRule{},
		[]config.EmbeddingRule{},
		[]config.Category{},
		[]config.Decision{
			{
				Name:     "priority-winner",
				Priority: 20,
				Rules: config.RuleCombination{
					Operator: "OR",
					Conditions: []config.RuleCondition{
						{Type: "keyword", Name: "alpha"},
					},
				},
			},
			{
				Name:     "confidence-runner-up",
				Priority: 10,
				Rules: config.RuleCombination{
					Operator: "OR",
					Conditions: []config.RuleCondition{
						{Type: "keyword", Name: "beta"},
					},
				},
			},
		},
		"priority",
	)

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		KeywordRules: []string{"alpha", "beta"},
		SignalConfidences: map[string]float64{
			"keyword:alpha": 0.51,
			"keyword:beta":  0.92,
		},
	})
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals failed: %v", err)
	}
	if result == nil {
		t.Fatal("expected decision result, got nil")
	}
	if result.Decision.Name != "priority-winner" {
		t.Fatalf("winner = %q, want priority-winner", result.Decision.Name)
	}
	if result.DecisionWinnerBasis != "priority" {
		t.Fatalf("winner_basis = %q, want priority", result.DecisionWinnerBasis)
	}
	if result.DecisionMargin >= 0 {
		t.Fatalf("decision_margin = %.2f, want negative value", result.DecisionMargin)
	}
	if result.RunnerUp == nil || result.RunnerUp.Name != "confidence-runner-up" {
		t.Fatalf("runner_up = %+v, want confidence-runner-up", result.RunnerUp)
	}
}

func TestDecisionEngine_EvaluateDecisionsWithNOTOperator(t *testing.T) {
	for _, tt := range notOperatorDecisionTestCases() {
		t.Run(tt.name, func(t *testing.T) {
			evaluateDecisionSignalsCase(t, tt, "priority")
		})
	}
}

func TestDecisionEngine_LatencyConditionIsIgnored(t *testing.T) {
	engine := NewDecisionEngine(
		[]config.KeywordRule{},
		[]config.EmbeddingRule{},
		[]config.Category{},
		[]config.Decision{
			{
				Name:     "legacy-latency",
				Priority: 10,
				Rules: config.RuleCombination{
					Operator: "AND",
					Conditions: []config.RuleCondition{
						{Type: "latency", Name: "low_latency"},
					},
				},
			},
		},
		"priority",
	)

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if result != nil {
		t.Fatalf("Expected no decision match for deprecated latency condition, got %q", result.Decision.Name)
	}
}

func evaluateDecisionsTestCases() []evaluateDecisionsTestCase {
	return []evaluateDecisionsTestCase{
		{
			name: "Single decision with AND operator - all rules match",
			decisions: []config.Decision{
				decisionWithConditions(
					"coding-task",
					10,
					"AND",
					[]config.RuleCondition{
						{Type: "keyword", Name: "programming"},
						{Type: "domain", Name: "coding"},
					},
					"codellama",
				),
			},
			strategy:              "priority",
			matchedKeywordRules:   []string{"programming"},
			matchedEmbeddingRules: []string{},
			matchedDomainRules:    []string{"coding"},
			expectedDecision:      "coding-task",
		},
		{
			name: "Single decision with AND operator - partial match",
			decisions: []config.Decision{
				decisionWithConditions(
					"coding-task",
					10,
					"AND",
					[]config.RuleCondition{
						{Type: "keyword", Name: "programming"},
						{Type: "domain", Name: "coding"},
					},
				),
			},
			strategy:              "priority",
			matchedKeywordRules:   []string{"programming"},
			matchedEmbeddingRules: []string{},
			expectedDecision:      "",
		},
		{
			name: "Single decision with OR operator - partial match",
			decisions: []config.Decision{
				decisionWithConditions(
					"coding-task",
					10,
					"OR",
					[]config.RuleCondition{
						{Type: "keyword", Name: "programming"},
						{Type: "domain", Name: "coding"},
					},
				),
			},
			strategy:              "priority",
			matchedKeywordRules:   []string{"programming"},
			matchedEmbeddingRules: []string{},
			expectedDecision:      "coding-task",
		},
		{
			name: "Multiple decisions - priority strategy",
			decisions: []config.Decision{
				decisionWithConditions("high-priority-task", 20, "OR", []config.RuleCondition{{Type: "keyword", Name: "urgent"}}),
				decisionWithConditions("low-priority-task", 10, "OR", []config.RuleCondition{{Type: "keyword", Name: "urgent"}}),
			},
			strategy:              "priority",
			matchedKeywordRules:   []string{"urgent"},
			matchedEmbeddingRules: []string{},
			expectedDecision:      "high-priority-task",
		},
	}
}

func factCheckDecisionTestCases() []evaluateWithSignalsTestCase {
	return []evaluateWithSignalsTestCase{
		{
			name: "Decision with fact_check condition - needs_fact_check matches",
			decisions: []config.Decision{
				decisionWithConditions("factual-query", 10, "AND", []config.RuleCondition{{Type: "fact_check", Name: "needs_fact_check"}}),
			},
			signals:          &SignalMatches{FactCheckRules: []string{"needs_fact_check"}},
			expectedDecision: "factual-query",
		},
		{
			name: "Decision with fact_check condition - no_fact_check_needed matches",
			decisions: []config.Decision{
				decisionWithConditions("creative-query", 10, "AND", []config.RuleCondition{{Type: "fact_check", Name: "no_fact_check_needed"}}),
			},
			signals:          &SignalMatches{FactCheckRules: []string{"no_fact_check_needed"}},
			expectedDecision: "creative-query",
		},
		{
			name: "Decision with mixed conditions - fact_check AND domain",
			decisions: []config.Decision{
				decisionWithConditions(
					"factual-science",
					10,
					"AND",
					[]config.RuleCondition{
						{Type: "fact_check", Name: "needs_fact_check"},
						{Type: "domain", Name: "science"},
					},
				),
			},
			signals: &SignalMatches{
				DomainRules:    []string{"science"},
				FactCheckRules: []string{"needs_fact_check"},
			},
			expectedDecision: "factual-science",
		},
		{
			name: "Decision with fact_check condition - no match",
			decisions: []config.Decision{
				decisionWithConditions("factual-query", 10, "AND", []config.RuleCondition{{Type: "fact_check", Name: "needs_fact_check"}}),
			},
			signals:          &SignalMatches{FactCheckRules: []string{"no_fact_check_needed"}},
			expectedDecision: "",
		},
	}
}

func notOperatorDecisionTestCases() []evaluateWithSignalsTestCase {
	return []evaluateWithSignalsTestCase{
		{
			name: "NOT operator - no conditions match (should match)",
			decisions: []config.Decision{
				notDecision("exclude-coding", 10, []config.RuleCondition{orCondition(
					[]config.RuleCondition{
						{Type: "keyword", Name: "programming"},
						{Type: "domain", Name: "coding"},
					},
				)}, "general-model"),
			},
			signals:          &SignalMatches{},
			expectedDecision: "exclude-coding",
			assertResult:     assertNOTOperatorConfidence,
		},
		{
			name: "NOT operator - one condition matches (should NOT match)",
			decisions: []config.Decision{
				notDecision("exclude-coding", 10, []config.RuleCondition{orCondition(
					[]config.RuleCondition{
						{Type: "keyword", Name: "programming"},
						{Type: "domain", Name: "coding"},
					},
				)}),
			},
			signals:          &SignalMatches{KeywordRules: []string{"programming"}},
			expectedDecision: "",
		},
		{
			name: "NOT operator - all conditions match (should NOT match)",
			decisions: []config.Decision{
				notDecision("exclude-coding", 10, []config.RuleCondition{orCondition(
					[]config.RuleCondition{
						{Type: "keyword", Name: "programming"},
						{Type: "domain", Name: "coding"},
					},
				)}),
			},
			signals: &SignalMatches{
				KeywordRules: []string{"programming"},
				DomainRules:  []string{"coding"},
			},
			expectedDecision: "",
		},
		{
			name: "NOT operator - confidence is 1.0 when matched",
			decisions: []config.Decision{
				notDecision("non-medical", 10, []config.RuleCondition{{Type: "domain", Name: "medical"}}, "general-model"),
			},
			signals:          &SignalMatches{DomainRules: []string{}},
			expectedDecision: "non-medical",
			assertResult:     assertNOTOperatorConfidence,
		},
		{
			name: "NOT operator priority over lower priority decision",
			decisions: []config.Decision{
				notDecision("not-medical-high", 20, []config.RuleCondition{{Type: "domain", Name: "medical"}}, "general-model"),
				notDecision("not-medical-low", 5, []config.RuleCondition{{Type: "domain", Name: "medical"}}, "backup-model"),
			},
			signals:          &SignalMatches{},
			expectedDecision: "not-medical-high",
			assertResult:     assertNOTOperatorConfidence,
		},
	}
}

func evaluateDecisionsCase(t *testing.T, tt evaluateDecisionsTestCase) {
	t.Helper()

	engine := NewDecisionEngine(
		[]config.KeywordRule{},
		[]config.EmbeddingRule{},
		[]config.Category{},
		tt.decisions,
		tt.strategy,
	)

	result, err := engine.EvaluateDecisions(
		tt.matchedKeywordRules,
		tt.matchedEmbeddingRules,
		tt.matchedDomainRules,
	)

	assertDecisionResult(t, result, err, tt.expectedDecision, tt.expectError)
}

func evaluateDecisionSignalsCase(t *testing.T, tt evaluateWithSignalsTestCase, strategy string) {
	t.Helper()

	engine := NewDecisionEngine(
		[]config.KeywordRule{},
		[]config.EmbeddingRule{},
		[]config.Category{},
		tt.decisions,
		strategy,
	)

	result, err := engine.EvaluateDecisionsWithSignals(tt.signals)
	assertDecisionResult(t, result, err, tt.expectedDecision, tt.expectError)
	if result != nil && tt.assertResult != nil {
		tt.assertResult(t, result)
	}
}

func assertDecisionResult(t *testing.T, result *DecisionResult, err error, expectedDecision string, expectError bool) {
	t.Helper()

	if expectError {
		if err == nil {
			t.Errorf("Expected error but got none")
		}
		return
	}

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
		return
	}

	if expectedDecision == "" {
		if result != nil {
			t.Errorf("Expected nil result but got decision: %s", result.Decision.Name)
		}
		return
	}

	if result == nil {
		t.Errorf("Expected result but got nil")
		return
	}

	if result.Decision.Name != expectedDecision {
		t.Errorf("Expected decision %s, got %s", expectedDecision, result.Decision.Name)
	}
}

func assertNOTOperatorConfidence(t *testing.T, result *DecisionResult) {
	t.Helper()

	if result.Confidence != 1.0 {
		t.Errorf("Expected confidence 1.0 for NOT operator match, got %f", result.Confidence)
	}
}

func decisionWithConditions(name string, priority int, operator string, conditions []config.RuleCondition, modelNames ...string) config.Decision {
	decision := config.Decision{
		Name:     name,
		Priority: priority,
		Rules: config.RuleCombination{
			Operator:   operator,
			Conditions: conditions,
		},
	}
	if len(modelNames) > 0 {
		decision.ModelRefs = make([]config.ModelRef, 0, len(modelNames))
		for _, modelName := range modelNames {
			decision.ModelRefs = append(decision.ModelRefs, config.ModelRef{Model: modelName})
		}
	}
	return decision
}

func notDecision(name string, priority int, conditions []config.RuleCondition, modelNames ...string) config.Decision {
	return decisionWithConditions(name, priority, "NOT", conditions, modelNames...)
}

func orCondition(conditions []config.RuleCondition) config.RuleCondition {
	return config.RuleCondition{
		Operator:   "OR",
		Conditions: conditions,
	}
}
