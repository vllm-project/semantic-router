package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDecisionEngine_CategoryKBSignalMatching(t *testing.T) {
	tests := []struct {
		name             string
		decisions        []config.Decision
		categoryKBRules  []string
		expectedDecision string
	}{
		{
			name: "category_kb match triggers route",
			decisions: []config.Decision{
				{
					Name:     "privacy-route",
					Priority: 200,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: "category_kb", Name: "proprietary_code"},
						},
					},
					ModelRefs: []config.ModelRef{{Model: "local-model"}},
				},
			},
			categoryKBRules:  []string{"proprietary_code"},
			expectedDecision: "privacy-route",
		},
		{
			name: "category_kb no match",
			decisions: []config.Decision{
				{
					Name:     "privacy-route",
					Priority: 200,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: "category_kb", Name: "proprietary_code"},
						},
					},
					ModelRefs: []config.ModelRef{{Model: "local-model"}},
				},
			},
			categoryKBRules:  []string{"generic_coding"},
			expectedDecision: "",
		},
		{
			name: "category_kb with OR keyword",
			decisions: []config.Decision{
				{
					Name:     "security-route",
					Priority: 300,
					Rules: config.RuleCombination{
						Operator: "OR",
						Conditions: []config.RuleCondition{
							{Type: "keyword", Name: "threat_kw"},
							{Type: "category_kb", Name: "prompt_injection"},
						},
					},
					ModelRefs: []config.ModelRef{{Model: "guard-model"}},
				},
			},
			categoryKBRules:  []string{"prompt_injection"},
			expectedDecision: "security-route",
		},
		{
			name: "category_kb AND keyword both required - keyword missing",
			decisions: []config.Decision{
				{
					Name:     "strict-route",
					Priority: 200,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: "category_kb", Name: "pii"},
							{Type: "keyword", Name: "sensitive"},
						},
					},
					ModelRefs: []config.ModelRef{{Model: "local-model"}},
				},
			},
			categoryKBRules:  []string{"pii"},
			expectedDecision: "",
		},
		{
			name: "category_kb priority ordering",
			decisions: []config.Decision{
				{
					Name:     "low-priority",
					Priority: 100,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: "category_kb", Name: "generic_coding"},
						},
					},
					ModelRefs: []config.ModelRef{{Model: "default-model"}},
				},
				{
					Name:     "high-priority",
					Priority: 300,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: "category_kb", Name: "prompt_injection"},
						},
					},
					ModelRefs: []config.ModelRef{{Model: "guard-model"}},
				},
			},
			categoryKBRules:  []string{"prompt_injection", "generic_coding"},
			expectedDecision: "high-priority",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine := NewDecisionEngine(nil, nil, nil, tt.decisions, "priority")

			signals := &SignalMatches{
				CategoryKBRules: tt.categoryKBRules,
			}

			result, err := engine.EvaluateDecisionsWithSignals(signals)
			if err != nil {
				t.Fatalf("EvaluateDecisionsWithSignals: %v", err)
			}

			if tt.expectedDecision == "" {
				if result != nil {
					t.Errorf("expected no match, got %q", result.Decision.Name)
				}
				return
			}

			if result == nil {
				t.Fatalf("expected match %q, got nil", tt.expectedDecision)
			}
			if result.Decision.Name != tt.expectedDecision {
				t.Errorf("expected %q, got %q", tt.expectedDecision, result.Decision.Name)
			}
		})
	}
}

func TestDecisionEngine_CategoryKBWithProjectionRules(t *testing.T) {
	decisions := []config.Decision{
		{
			Name:     "override-route",
			Priority: 250,
			Rules: config.RuleCombination{
				Operator: "OR",
				Conditions: []config.RuleCondition{
					{Type: "category_kb", Name: "pii"},
					{Type: "projection", Name: "privacy_override_active"},
				},
			},
			ModelRefs: []config.ModelRef{{Model: "local-model"}},
		},
	}

	engine := NewDecisionEngine(nil, nil, nil, decisions, "priority")

	t.Run("projection match alone", func(t *testing.T) {
		signals := &SignalMatches{
			ProjectionRules: []string{"privacy_override_active"},
		}
		result, err := engine.EvaluateDecisionsWithSignals(signals)
		if err != nil {
			t.Fatalf("EvaluateDecisionsWithSignals: %v", err)
		}
		if result == nil || result.Decision.Name != "override-route" {
			name := "<nil>"
			if result != nil {
				name = result.Decision.Name
			}
			t.Errorf("expected override-route via projection, got %s", name)
		}
	})

	t.Run("category_kb match alone", func(t *testing.T) {
		signals := &SignalMatches{
			CategoryKBRules: []string{"pii"},
		}
		result, err := engine.EvaluateDecisionsWithSignals(signals)
		if err != nil {
			t.Fatalf("EvaluateDecisionsWithSignals: %v", err)
		}
		if result == nil || result.Decision.Name != "override-route" {
			name := "<nil>"
			if result != nil {
				name = result.Decision.Name
			}
			t.Errorf("expected override-route via category_kb, got %s", name)
		}
	})

	t.Run("neither matches", func(t *testing.T) {
		signals := &SignalMatches{
			CategoryKBRules: []string{"generic_coding"},
			ProjectionRules: []string{"some_other"},
		}
		result, err := engine.EvaluateDecisionsWithSignals(signals)
		if err != nil {
			t.Fatalf("EvaluateDecisionsWithSignals: %v", err)
		}
		if result != nil {
			t.Errorf("expected no match, got %q", result.Decision.Name)
		}
	})
}
