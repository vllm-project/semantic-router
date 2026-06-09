package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDecisionEngine_EvaluateDecisionsWithContext(t *testing.T) {
	tests := []struct {
		name             string
		decisions        []config.Decision
		signals          *SignalMatches
		expectedDecision string
		expectError      bool
	}{
		{
			name: "Decision with context condition - matches",
			decisions: []config.Decision{
				{
					Name:     "context-decision",
					Priority: 10,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: "context", Name: "high_tokens"},
						},
					},
				},
			},
			signals: &SignalMatches{
				ContextRules: []string{"high_tokens"},
			},
			expectedDecision: "context-decision",
			expectError:      false,
		},
		{
			name: "Decision with context condition - no match",
			decisions: []config.Decision{
				{
					Name:     "context-decision",
					Priority: 10,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: "context", Name: "high_tokens"},
						},
					},
				},
			},
			signals: &SignalMatches{
				ContextRules: []string{"low_tokens"},
			},
			expectedDecision: "",
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
		})
	}
}

func TestDecisionEngine_ContextGuardPreventsLowTokenRouteShadowing(t *testing.T) {
	tests := []struct {
		name             string
		decisions        []config.Decision
		signals          *SignalMatches
		expectedDecision string
	}{
		{
			name: "domain-only code route shadows high-token route by priority",
			decisions: []config.Decision{
				{
					Name:     "low-token-hard-code-route",
					Priority: 180,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: "domain", Name: "code_generation"},
						},
					},
				},
				{
					Name:     "high-token-route",
					Priority: 170,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: "context", Name: "high_token_count"},
						},
					},
				},
			},
			signals: &SignalMatches{
				DomainRules:  []string{"code_generation"},
				ContextRules: []string{"high_token_count"},
			},
			expectedDecision: "low-token-hard-code-route",
		},
		{
			name: "explicit low-token guard lets long code route to high-token decision",
			decisions: []config.Decision{
				{
					Name:     "low-token-hard-code-route",
					Priority: 180,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: "domain", Name: "code_generation"},
							{Type: "context", Name: "low_token_count"},
						},
					},
				},
				{
					Name:     "high-token-route",
					Priority: 170,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: "context", Name: "high_token_count"},
						},
					},
				},
			},
			signals: &SignalMatches{
				DomainRules:  []string{"code_generation"},
				ContextRules: []string{"high_token_count"},
			},
			expectedDecision: "high-token-route",
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
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			if result == nil {
				t.Fatalf("Expected decision %s, got nil", tt.expectedDecision)
			}
			if result.Decision.Name != tt.expectedDecision {
				t.Fatalf("Expected decision %s, got %s", tt.expectedDecision, result.Decision.Name)
			}
		})
	}
}
