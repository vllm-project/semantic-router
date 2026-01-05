package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDecisionEngine_EvaluateDecisions(t *testing.T) {
	tests := []struct {
		name                  string
		decisions             []config.Decision
		strategy              string
		matchedKeywordRules   []string
		matchedEmbeddingRules []string
		matchedDomainRules    []string
		expectedDecision      string
		expectError           bool
	}{
		{
			name: "Single decision with AND operator - all rules match",
			decisions: []config.Decision{
				{
					Name:     "coding-task",
					Priority: 10,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: "keyword", Name: "programming"},
							{Type: "domain", Name: "coding"},
						},
					},
					ModelRefs: []config.ModelRef{
						{Model: "codellama"},
					},
				},
			},
			strategy:              "priority",
			matchedKeywordRules:   []string{"programming"},
			matchedEmbeddingRules: []string{},
			matchedDomainRules:    []string{"coding"},
			expectedDecision:      "coding-task",
			expectError:           false,
		},
		{
			name: "Single decision with AND operator - partial match",
			decisions: []config.Decision{
				{
					Name:     "coding-task",
					Priority: 10,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: "keyword", Name: "programming"},
							{Type: "domain", Name: "coding"},
						},
					},
				},
			},
			strategy:              "priority",
			matchedKeywordRules:   []string{"programming"},
			matchedEmbeddingRules: []string{},
			matchedDomainRules:    []string{}, // Missing domain rule
			expectedDecision:      "",
			expectError:           false, // Changed: no match should return nil result, not error
		},
		{
			name: "Single decision with OR operator - partial match",
			decisions: []config.Decision{
				{
					Name:     "coding-task",
					Priority: 10,
					Rules: config.RuleCombination{
						Operator: "OR",
						Conditions: []config.RuleCondition{
							{Type: "keyword", Name: "programming"},
							{Type: "domain", Name: "coding"},
						},
					},
				},
			},
			strategy:              "priority",
			matchedKeywordRules:   []string{"programming"},
			matchedEmbeddingRules: []string{},
			matchedDomainRules:    []string{}, // Missing domain rule, but OR should still match
			expectedDecision:      "coding-task",
			expectError:           false,
		},
		{
			name: "Multiple decisions - priority strategy",
			decisions: []config.Decision{
				{
					Name:     "high-priority-task",
					Priority: 20,
					Rules: config.RuleCombination{
						Operator: "OR",
						Conditions: []config.RuleCondition{
							{Type: "keyword", Name: "urgent"},
						},
					},
				},
				{
					Name:     "low-priority-task",
					Priority: 10,
					Rules: config.RuleCombination{
						Operator: "OR",
						Conditions: []config.RuleCondition{
							{Type: "keyword", Name: "urgent"},
						},
					},
				},
			},
			strategy:              "priority",
			matchedKeywordRules:   []string{"urgent"},
			matchedEmbeddingRules: []string{},
			matchedDomainRules:    []string{},
			expectedDecision:      "high-priority-task", // Higher priority wins
			expectError:           false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
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

			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			// If expectedDecision is empty, we expect nil result (no match)
			if tt.expectedDecision == "" {
				if result != nil {
					t.Errorf("Expected nil result but got decision: %s", result.Decision.Name)
				}
				return
			}

			if result == nil {
				t.Errorf("Expected result but got nil")
				return
			}

			if result.Decision.Name != tt.expectedDecision {
				t.Errorf("Expected decision %s, got %s", tt.expectedDecision, result.Decision.Name)
			}
		})
	}
}

func TestDecisionEngine_EvaluateDecisionsWithFactCheck(t *testing.T) {
	tests := []struct {
		name             string
		decisions        []config.Decision
		signals          *SignalMatches
		expectedDecision string
		expectError      bool
	}{
		{
			name: "Decision with fact_check condition - needs_fact_check matches",
			decisions: []config.Decision{
				{
					Name:     "factual-query",
					Priority: 10,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: "fact_check", Name: "needs_fact_check"},
						},
					},
				},
			},
			signals: &SignalMatches{
				FactCheckRules: []string{"needs_fact_check"},
			},
			expectedDecision: "factual-query",
			expectError:      false,
		},
		{
			name: "Decision with fact_check condition - no_fact_check_needed matches",
			decisions: []config.Decision{
				{
					Name:     "creative-query",
					Priority: 10,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: "fact_check", Name: "no_fact_check_needed"},
						},
					},
				},
			},
			signals: &SignalMatches{
				FactCheckRules: []string{"no_fact_check_needed"},
			},
			expectedDecision: "creative-query",
			expectError:      false,
		},
		{
			name: "Decision with mixed conditions - fact_check AND domain",
			decisions: []config.Decision{
				{
					Name:     "factual-science",
					Priority: 10,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: "fact_check", Name: "needs_fact_check"},
							{Type: "domain", Name: "science"},
						},
					},
				},
			},
			signals: &SignalMatches{
				DomainRules:    []string{"science"},
				FactCheckRules: []string{"needs_fact_check"},
			},
			expectedDecision: "factual-science",
			expectError:      false,
		},
		{
			name: "Decision with fact_check condition - no match",
			decisions: []config.Decision{
				{
					Name:     "factual-query",
					Priority: 10,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: "fact_check", Name: "needs_fact_check"},
						},
					},
				},
			},
			signals: &SignalMatches{
				FactCheckRules: []string{"no_fact_check_needed"},
			},
			expectedDecision: "",
			expectError:      false, // Changed: no match should return nil result, not error
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine := NewDecisionEngine(
				[]config.KeywordRule{},
				[]config.EmbeddingRule{},
				[]config.Category{},
				tt.decisions,
				"priority",
			)

			result, err := engine.EvaluateDecisionsWithSignals(tt.signals)

			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			// If expectedDecision is empty, we expect nil result (no match)
			if tt.expectedDecision == "" {
				if result != nil {
					t.Errorf("Expected nil result but got decision: %s", result.Decision.Name)
				}
				return
			}

			if result == nil {
				t.Errorf("Expected result but got nil")
				return
			}

			if result.Decision.Name != tt.expectedDecision {
				t.Errorf("Expected decision %s, got %s", tt.expectedDecision, result.Decision.Name)
			}
		})
	}
}
