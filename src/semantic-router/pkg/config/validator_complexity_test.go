package config

import (
	"strings"
	"testing"
)

// complexityDecisionConfig builds a RouterConfig that declares the given
// complexity rules and a single decision whose (possibly nested) rule tree
// contains one complexity condition with the given name.
func complexityDecisionConfig(declaredRules []string, conditionName string, nested bool) *RouterConfig {
	rules := make([]ComplexityRule, 0, len(declaredRules))
	for _, name := range declaredRules {
		rules = append(rules, ComplexityRule{Name: name})
	}

	cond := RuleNode{Type: SignalTypeComplexity, Name: conditionName}
	root := RuleNode{Operator: "AND", Conditions: []RuleNode{cond}}
	if nested {
		// Bury the complexity condition under OR -> NOT to exercise recursion.
		root = RuleNode{
			Operator: "AND",
			Conditions: []RuleNode{
				{Operator: "OR", Conditions: []RuleNode{
					{Type: SignalTypeKeyword, Name: "kw"},
					{Operator: "NOT", Conditions: []RuleNode{cond}},
				}},
			},
		}
	}

	return &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals:   Signals{ComplexityRules: rules},
			Decisions: []Decision{{Name: "route", Rules: root}},
		},
	}
}

func TestValidateComplexityContracts(t *testing.T) {
	cases := []struct {
		name       string
		declared   []string
		condition  string
		nested     bool
		wantErr    bool
		errSubstrs []string
	}{
		{
			name:      "valid hard",
			declared:  []string{"needs_reasoning"},
			condition: "needs_reasoning:hard",
		},
		{
			name:      "valid easy",
			declared:  []string{"needs_reasoning"},
			condition: "needs_reasoning:easy",
		},
		{
			name:      "valid medium",
			declared:  []string{"needs_reasoning"},
			condition: "needs_reasoning:medium",
		},
		{
			name:      "valid nested under OR/NOT",
			declared:  []string{"needs_reasoning"},
			condition: "needs_reasoning:hard",
			nested:    true,
		},
		{
			name:       "bare rule name rejected",
			declared:   []string{"needs_reasoning"},
			condition:  "needs_reasoning",
			wantErr:    true,
			errSubstrs: []string{"needs_reasoning", "difficulty", "easy|hard|medium"},
		},
		{
			name:       "bare rule name rejected even when nested",
			declared:   []string{"needs_reasoning"},
			condition:  "needs_reasoning",
			nested:     true,
			wantErr:    true,
			errSubstrs: []string{"difficulty"},
		},
		{
			name:       "unknown difficulty rejected",
			declared:   []string{"needs_reasoning"},
			condition:  "needs_reasoning:hardd",
			wantErr:    true,
			errSubstrs: []string{"unsupported difficulty", "hardd"},
		},
		{
			name:       "undeclared rule rejected",
			declared:   []string{"needs_reasoning"},
			condition:  "bogus_rule:hard",
			wantErr:    true,
			errSubstrs: []string{"bogus_rule", "no routing.signals.complexity"},
		},
		{
			name:       "trailing colon rejected",
			declared:   []string{"needs_reasoning"},
			condition:  "needs_reasoning:",
			wantErr:    true,
			errSubstrs: []string{"difficulty"},
		},
		{
			name:       "leading colon rejected",
			declared:   []string{"needs_reasoning"},
			condition:  ":hard",
			wantErr:    true,
			errSubstrs: []string{"difficulty"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := complexityDecisionConfig(tc.declared, tc.condition, tc.nested)
			err := validateComplexityContracts(cfg)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error for condition %q, got nil", tc.condition)
				}
				for _, sub := range tc.errSubstrs {
					if !strings.Contains(err.Error(), sub) {
						t.Errorf("error %q missing expected substring %q", err.Error(), sub)
					}
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error for condition %q: %v", tc.condition, err)
			}
		})
	}
}

// TestValidateComplexityContracts_NoComplexityConditions ensures decisions
// without any complexity conditions are unaffected.
func TestValidateComplexityContracts_NoComplexityConditions(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Decisions: []Decision{{
				Name: "route",
				Rules: RuleNode{Operator: "AND", Conditions: []RuleNode{
					{Type: SignalTypeDomain, Name: "business"},
				}},
			}},
		},
	}
	if err := validateComplexityContracts(cfg); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}
