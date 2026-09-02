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

// ruleTreeShapeCase is one entry in the rule-tree shape corpus shared with the
// decision engine parity test (pkg/decision/rule_tree_shape_test.go). Every
// tree here is asserted through the loader-wide validator so config, router,
// and CRD-converted trees agree on the same accepted and rejected shapes.
type ruleTreeShapeCase struct {
	name    string
	rules   RuleNode
	wantErr string // substring of the validation error; empty means accepted
}

func ruleTreeShapeCorpus() []ruleTreeShapeCase {
	kw := func(name string) RuleNode { return RuleNode{Type: SignalTypeKeyword, Name: name} }
	return []ruleTreeShapeCase{
		// Accepted shapes.
		{name: "omitted rules root", rules: RuleNode{}},
		{name: "root childless AND is match-all", rules: RuleNode{Operator: "AND"}},
		{name: "root childless and lowercase is match-all", rules: RuleNode{Operator: "and"}},
		{name: "single leaf root", rules: kw("a")},
		{name: "omitted operator defaults to AND", rules: RuleNode{Conditions: []RuleNode{kw("a"), kw("b")}}},
		{name: "AND", rules: RuleNode{Operator: "AND", Conditions: []RuleNode{kw("a"), kw("b")}}},
		{name: "OR", rules: RuleNode{Operator: "OR", Conditions: []RuleNode{kw("a"), kw("b")}}},
		{name: "NOT one child", rules: RuleNode{Operator: "NOT", Conditions: []RuleNode{kw("a")}}},
		{name: "padded lowercase not", rules: RuleNode{Operator: " not ", Conditions: []RuleNode{kw("a")}}},
		{name: "nested NOT around OR", rules: RuleNode{Operator: "AND", Conditions: []RuleNode{
			kw("a"),
			{Operator: "NOT", Conditions: []RuleNode{{Operator: "OR", Conditions: []RuleNode{kw("b"), kw("c")}}}},
		}}},
		{name: "nested omitted operator", rules: RuleNode{Operator: "OR", Conditions: []RuleNode{
			{Conditions: []RuleNode{kw("a"), kw("b")}},
			kw("c"),
		}}},

		// Rejected operators.
		{name: "root XOR", rules: RuleNode{Operator: "XOR", Conditions: []RuleNode{kw("a")}},
			wantErr: `rules: unsupported operator "XOR"`},
		{name: "root typo ADN", rules: RuleNode{Operator: "ADN", Conditions: []RuleNode{kw("a"), kw("b")}},
			wantErr: `rules: unsupported operator "ADN"`},
		{name: "nested NOR", rules: RuleNode{Operator: "AND", Conditions: []RuleNode{
			kw("a"),
			{Operator: "NOR", Conditions: []RuleNode{kw("b")}},
		}}, wantErr: `rules.conditions[1]: unsupported operator "NOR"`},

		// Rejected NOT arity.
		{name: "root NOT zero children", rules: RuleNode{Operator: "NOT"},
			wantErr: "rules: NOT requires exactly one child condition, got 0"},
		{name: "root NOT two children", rules: RuleNode{Operator: "NOT", Conditions: []RuleNode{kw("a"), kw("b")}},
			wantErr: "rules: NOT requires exactly one child condition, got 2"},
		{name: "nested NOT two children", rules: RuleNode{Operator: "OR", Conditions: []RuleNode{
			kw("a"),
			{Operator: "NOT", Conditions: []RuleNode{kw("b"), kw("c")}},
		}}, wantErr: "rules.conditions[1]: NOT requires exactly one child condition, got 2"},

		// Rejected childless combinations.
		{name: "root childless OR never matches", rules: RuleNode{Operator: "OR"},
			wantErr: "rules: OR combination requires at least one child condition"},
		{name: "nested childless AND", rules: RuleNode{Operator: "OR", Conditions: []RuleNode{kw("a"), {Operator: "AND"}}},
			wantErr: "rules.conditions[1]: AND combination requires at least one child condition"},
		{name: "nested empty node", rules: RuleNode{Operator: "AND", Conditions: []RuleNode{kw("a"), {}}},
			wantErr: "rules.conditions[1]: combination condition requires an operator and at least one child condition"},

		// Rejected node shapes.
		{name: "leaf with operator", rules: RuleNode{Operator: "AND", Conditions: []RuleNode{
			{Type: SignalTypeKeyword, Name: "a", Operator: "OR"},
		}}, wantErr: "rules.conditions[0]: condition must be either a leaf (type/name) or a combination (operator/conditions), not both"},
		{name: "leaf with children", rules: RuleNode{Type: SignalTypeKeyword, Name: "a", Conditions: []RuleNode{kw("b")}},
			wantErr: "rules: condition must be either a leaf (type/name) or a combination (operator/conditions), not both"},
		{name: "name without type", rules: RuleNode{Operator: "AND", Conditions: []RuleNode{{Name: "a"}}},
			wantErr: "rules.conditions[0]: leaf condition requires a type"},
		{name: "root label only satisfies IsEmpty", rules: RuleNode{Label: "positive"},
			wantErr: "rules: leaf condition requires a type"},
		{name: "root on_error only", rules: RuleNode{OnError: "match"},
			wantErr: "rules: leaf condition requires a type"},
		{name: "predicate without type", rules: RuleNode{Operator: "AND", Conditions: []RuleNode{
			{Name: "a", Predicate: &NumericPredicate{}},
		}}, wantErr: "rules.conditions[0]: leaf condition requires a type"},
	}
}

func TestNormalizeRuleOperator_ShapeCorpus(t *testing.T) {
	for _, tc := range ruleTreeShapeCorpus() {
		t.Run(tc.name, func(t *testing.T) {
			node := tc.rules
			err := NormalizeRuleOperator(&node)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("expected tree to be accepted, got: %v", err)
				}
				assertRuleTreeNormalized(t, &node, true)
				return
			}
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected error containing %q, got: %v", tc.wantErr, err)
			}
		})
	}
}

// assertRuleTreeNormalized checks the post-normalization invariants every
// evaluator relies on: each combination node carries a canonical operator, NOT
// is unary, and no nested combination is childless.
func assertRuleTreeNormalized(t *testing.T, node *RuleNode, root bool) {
	t.Helper()
	if node.IsLeaf() {
		return
	}
	if root && node.IsEmpty() {
		return
	}
	if !IsRuleTreeOperator(node.Operator) {
		t.Fatalf("expected a canonical operator after normalization, got %q", node.Operator)
	}
	if node.Operator == RuleOperatorNot && len(node.Conditions) != 1 {
		t.Fatalf("expected NOT to be unary after normalization, got %d children", len(node.Conditions))
	}
	if !root && len(node.Conditions) == 0 {
		t.Fatalf("expected nested combination to carry children after normalization")
	}
	for i := range node.Conditions {
		assertRuleTreeNormalized(t, &node.Conditions[i], false)
	}
}

func TestValidateRuleOperatorContracts_ShapeCorpusNamesDecisionAndPath(t *testing.T) {
	for _, tc := range ruleTreeShapeCorpus() {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &RouterConfig{
				IntelligentRouting: IntelligentRouting{
					Decisions: []Decision{{Name: "route", Rules: tc.rules}},
				},
			}
			err := validateRuleOperatorContracts(cfg)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("expected tree to be accepted, got: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantErr)
			}
			want := `decision "route": ` + tc.wantErr
			if !strings.Contains(err.Error(), want) {
				t.Fatalf("expected error containing %q, got: %v", want, err)
			}
		})
	}
}

func TestValidateRuleOperatorContracts_ComposerRejectsNotArity(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				ComplexityRules: []ComplexityRule{{
					Name: "needs_reasoning",
					Composer: &RuleNode{
						Operator: "NOT",
						Conditions: []RuleNode{
							{Type: SignalTypeKeyword, Name: "kw"},
							{Type: SignalTypeDomain, Name: "dom"},
						},
					},
				}},
			},
		},
	}

	err := validateRuleOperatorContracts(cfg)
	if err == nil {
		t.Fatal("expected an error for a multi-child composer NOT")
	}
	want := `complexity rule "needs_reasoning": composer: NOT requires exactly one child condition, got 2`
	if !strings.Contains(err.Error(), want) {
		t.Fatalf("expected error containing %q, got: %v", want, err)
	}
}

func TestParseYAMLBytes_RejectsMultiChildNotInDecisionRules(t *testing.T) {
	yamlConfig := `
version: v0.3
providers:
  defaults:
    default_model: gpt-worker
  models:
    - name: gpt-worker
      provider_model_id: openai/gpt-5.5
      backend_refs:
        - name: openrouter
          base_url: https://openrouter.ai/api/v1
          provider: openai
routing:
  modelCards:
    - name: gpt-worker
  signals:
    keywords:
      - name: urgent
        operator: OR
        keywords: ["urgent"]
      - name: billing
        operator: OR
        keywords: ["invoice"]
  decisions:
    - name: not_urgent_billing
      priority: 10
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: billing
          - operator: NOT
            conditions:
              - type: keyword
                name: urgent
              - type: keyword
                name: billing
      modelRefs:
        - model: gpt-worker
          use_reasoning: false
`
	_, err := ParseYAMLBytes([]byte(yamlConfig))
	if err == nil {
		t.Fatal("expected ParseYAMLBytes to reject a multi-child NOT")
	}
	want := `decision "not_urgent_billing": rules.conditions[1]: NOT requires exactly one child condition, got 2`
	if !strings.Contains(err.Error(), want) {
		t.Fatalf("expected error containing %q, got: %v", want, err)
	}
}
