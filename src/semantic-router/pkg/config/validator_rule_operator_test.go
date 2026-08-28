package config

import (
	"strings"
	"testing"
)

func TestNormalizeRuleOperator_DefaultsOmittedOperatorToAND(t *testing.T) {
	// Reproduces the issue #2937 divergence: a rule node with two
	// conditions and no operator must mean AND everywhere, not OR.
	node := &RuleNode{
		Conditions: []RuleNode{
			{Type: SignalTypeDomain, Name: "business"},
			{Type: SignalTypeKeyword, Name: "urgent"},
		},
	}

	if err := NormalizeRuleOperator(node); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if node.Operator != "AND" {
		t.Fatalf("expected omitted operator to default to AND, got %q", node.Operator)
	}
}

func TestNormalizeRuleOperator_TrimsAndUppercases(t *testing.T) {
	cases := []string{"and", "And", " AND", "and ", " AnD "}
	for _, raw := range cases {
		node := &RuleNode{
			Operator:   raw,
			Conditions: []RuleNode{{Type: SignalTypeKeyword, Name: "kw"}},
		}
		if err := NormalizeRuleOperator(node); err != nil {
			t.Fatalf("operator %q: unexpected error: %v", raw, err)
		}
		if node.Operator != "AND" {
			t.Fatalf("operator %q: expected normalized AND, got %q", raw, node.Operator)
		}
	}
}

func TestNormalizeRuleOperator_PreservesEmptyRootFallback(t *testing.T) {
	node := &RuleNode{}

	if err := NormalizeRuleOperator(node); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !node.IsEmpty() {
		t.Fatalf("expected the completely empty node to remain the match-all fallback, got %+v", node)
	}
}

func TestNormalizeRuleOperator_RejectsUnsupportedOperator(t *testing.T) {
	node := &RuleNode{
		Operator:   "XOR",
		Conditions: []RuleNode{{Type: SignalTypeKeyword, Name: "kw"}},
	}

	err := NormalizeRuleOperator(node)
	if err == nil {
		t.Fatal("expected an error for an unsupported operator")
	}
	if !strings.Contains(err.Error(), "XOR") {
		t.Fatalf("expected error to name the offending operator, got: %v", err)
	}
}

func TestNormalizeRuleOperator_RecursesIntoNestedConditions(t *testing.T) {
	node := &RuleNode{
		Operator: "and",
		Conditions: []RuleNode{
			{Type: SignalTypeKeyword, Name: "kw"},
			{
				Conditions: []RuleNode{
					{Type: SignalTypeDomain, Name: "business"},
					{Type: SignalTypeDomain, Name: "urgent"},
				},
			},
		},
	}

	if err := NormalizeRuleOperator(node); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if node.Operator != "AND" {
		t.Fatalf("expected root operator AND, got %q", node.Operator)
	}
	if node.Conditions[1].Operator != "AND" {
		t.Fatalf("expected nested omitted operator to default to AND, got %q", node.Conditions[1].Operator)
	}
}

func TestNormalizeRuleOperator_NestedInvalidOperatorNamesItsPath(t *testing.T) {
	node := &RuleNode{
		Operator: "AND",
		Conditions: []RuleNode{
			{Type: SignalTypeKeyword, Name: "kw"},
			{Operator: "MAYBE", Conditions: []RuleNode{{Type: SignalTypeKeyword, Name: "kw2"}}},
		},
	}

	err := NormalizeRuleOperator(node)
	if err == nil {
		t.Fatal("expected an error for a nested unsupported operator")
	}
	if !strings.Contains(err.Error(), "conditions[1]") {
		t.Fatalf("expected error to name the nested path, got: %v", err)
	}
}

func TestValidateRuleOperatorContracts_NormalizesDecisionsAndComposers(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				ComplexityRules: []ComplexityRule{
					{
						Name: "needs_reasoning",
						Composer: &RuleNode{
							Operator: "or",
							Conditions: []RuleNode{
								{Type: SignalTypeKeyword, Name: "kw"},
								{Type: SignalTypeDomain, Name: "dom"},
							},
						},
					},
				},
			},
			Decisions: []Decision{
				{
					Name: "route",
					Rules: RuleNode{
						Conditions: []RuleNode{
							{Type: SignalTypeKeyword, Name: "kw"},
							{Type: SignalTypeDomain, Name: "dom"},
						},
					},
				},
			},
		},
	}

	if err := validateRuleOperatorContracts(cfg); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Decisions[0].Rules.Operator != "AND" {
		t.Fatalf("expected decision rule operator to default to AND, got %q", cfg.Decisions[0].Rules.Operator)
	}
	if cfg.ComplexityRules[0].Composer.Operator != "OR" {
		t.Fatalf("expected composer operator to normalize to OR, got %q", cfg.ComplexityRules[0].Composer.Operator)
	}
}

func TestValidateRuleOperatorContracts_RejectsInvalidDecisionOperator(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Decisions: []Decision{
				{
					Name: "route",
					Rules: RuleNode{
						Operator:   "XOR",
						Conditions: []RuleNode{{Type: SignalTypeKeyword, Name: "kw"}},
					},
				},
			},
		},
	}

	err := validateRuleOperatorContracts(cfg)
	if err == nil {
		t.Fatal("expected an error for an invalid decision rule operator")
	}
	if !strings.Contains(err.Error(), `"route"`) {
		t.Fatalf("expected error to name the decision, got: %v", err)
	}
}
