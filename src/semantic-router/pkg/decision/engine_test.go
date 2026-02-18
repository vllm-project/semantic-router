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

func TestDecisionEngine_EvaluateDecisionsWithNOTOperator(t *testing.T) {
	tests := []struct {
		name             string
		decisions        []config.Decision
		signals          *SignalMatches
		expectedDecision string
		expectError      bool
	}{
		{
			name: "NOT operator - no conditions match (should match)",
			decisions: []config.Decision{
				{
					Name:     "exclude-coding",
					Priority: 10,
					Rules: config.RuleCombination{
						Operator: "NOT",
						Conditions: []config.RuleCondition{
							{Type: "keyword", Name: "programming"},
							{Type: "domain", Name: "coding"},
						},
					},
					ModelRefs: []config.ModelRef{
						{Model: "general-model"},
					},
				},
			},
			signals:          &SignalMatches{},
			expectedDecision: "exclude-coding",
			expectError:      false,
		},
		{
			name: "NOT operator - one condition matches (should NOT match)",
			decisions: []config.Decision{
				{
					Name:     "exclude-coding",
					Priority: 10,
					Rules: config.RuleCombination{
						Operator: "NOT",
						Conditions: []config.RuleCondition{
							{Type: "keyword", Name: "programming"},
							{Type: "domain", Name: "coding"},
						},
					},
				},
			},
			signals: &SignalMatches{
				KeywordRules: []string{"programming"},
			},
			expectedDecision: "",
			expectError:      false,
		},
		{
			name: "NOT operator - all conditions match (should NOT match)",
			decisions: []config.Decision{
				{
					Name:     "exclude-coding",
					Priority: 10,
					Rules: config.RuleCombination{
						Operator: "NOT",
						Conditions: []config.RuleCondition{
							{Type: "keyword", Name: "programming"},
							{Type: "domain", Name: "coding"},
						},
					},
				},
			},
			signals: &SignalMatches{
				KeywordRules: []string{"programming"},
				DomainRules:  []string{"coding"},
			},
			expectedDecision: "",
			expectError:      false,
		},
		{
			name: "NOT operator - confidence is 1.0 when matched",
			decisions: []config.Decision{
				{
					Name:     "non-medical",
					Priority: 10,
					Rules: config.RuleCombination{
						Operator: "NOT",
						Conditions: []config.RuleCondition{
							{Type: "domain", Name: "medical"},
						},
					},
					ModelRefs: []config.ModelRef{
						{Model: "general-model"},
					},
				},
			},
			signals:          &SignalMatches{DomainRules: []string{}},
			expectedDecision: "non-medical",
			expectError:      false,
		},
		{
			name: "NOT operator priority over lower priority decision",
			decisions: []config.Decision{
				{
					Name:     "not-medical-high",
					Priority: 20,
					Rules: config.RuleCombination{
						Operator: "NOT",
						Conditions: []config.RuleCondition{
							{Type: "domain", Name: "medical"},
						},
					},
					ModelRefs: []config.ModelRef{{Model: "general-model"}},
				},
				{
					Name:     "not-medical-low",
					Priority: 5,
					Rules: config.RuleCombination{
						Operator: "NOT",
						Conditions: []config.RuleCondition{
							{Type: "domain", Name: "medical"},
						},
					},
					ModelRefs: []config.ModelRef{{Model: "backup-model"}},
				},
			},
			signals:          &SignalMatches{},
			expectedDecision: "not-medical-high",
			expectError:      false,
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

			// Verify confidence is 1.0 for NOT operator matches
			if result.Confidence != 1.0 {
				t.Errorf("Expected confidence 1.0 for NOT operator match, got %f", result.Confidence)
			}
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
