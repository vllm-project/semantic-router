package config

import "testing"

func TestValidateAgenticFactsSignalContracts(t *testing.T) {
	reviewer := "reviewer"
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{AgenticFactsRules: []AgenticFactsRule{{
			Name:      "reviewer-role",
			Field:     AgenticFactsFieldDelegatedRole,
			Predicate: AgenticFactsPredicate{Equals: &reviewer},
		}}},
	}}
	if err := validateAgenticFactsSignalContracts(cfg); err != nil {
		t.Fatalf("validateAgenticFactsSignalContracts() error = %v", err)
	}
}

func TestValidateAgenticFactsSignalContractsAcceptsInPredicate(t *testing.T) {
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{AgenticFactsRules: []AgenticFactsRule{{
			Name:      "execution-phase",
			Field:     AgenticFactsFieldTaskPhase,
			Predicate: AgenticFactsPredicate{In: []string{"execute", "verify"}},
		}}},
	}}
	if err := validateAgenticFactsSignalContracts(cfg); err != nil {
		t.Fatalf("validateAgenticFactsSignalContracts() error = %v", err)
	}
}

func TestValidateAgenticFactsSignalContractsRejectsUnknownField(t *testing.T) {
	value := "x"
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{AgenticFactsRules: []AgenticFactsRule{{
			Name:      "bad-field",
			Field:     "lineage_depth",
			Predicate: AgenticFactsPredicate{Equals: &value},
		}}},
	}}
	if err := validateAgenticFactsSignalContracts(cfg); err == nil {
		t.Fatal("validateAgenticFactsSignalContracts() expected unsupported field error")
	}
}

func TestValidateAgenticFactsSignalContractsRejectsAmbiguousPredicate(t *testing.T) {
	value := "reviewer"
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{AgenticFactsRules: []AgenticFactsRule{{
			Name:  "bad",
			Field: AgenticFactsFieldDelegatedRole,
			Predicate: AgenticFactsPredicate{
				Equals: &value,
				In:     []string{"reviewer", "auditor"},
			},
		}}},
	}}
	if err := validateAgenticFactsSignalContracts(cfg); err == nil {
		t.Fatal("validateAgenticFactsSignalContracts() expected ambiguous predicate error")
	}
}

func TestValidateAgenticFactsSignalContractsRejectsNoPredicate(t *testing.T) {
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{AgenticFactsRules: []AgenticFactsRule{{
			Name:  "bad",
			Field: AgenticFactsFieldDelegatedRole,
		}}},
	}}
	if err := validateAgenticFactsSignalContracts(cfg); err == nil {
		t.Fatal("validateAgenticFactsSignalContracts() expected missing predicate error")
	}
}

func TestValidateAgenticFactsSignalContractsRejectsWhitespaceName(t *testing.T) {
	value := "reviewer"
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{AgenticFactsRules: []AgenticFactsRule{{
			Name:      " reviewer-role ",
			Field:     AgenticFactsFieldDelegatedRole,
			Predicate: AgenticFactsPredicate{Equals: &value},
		}}},
	}}
	if err := validateAgenticFactsSignalContracts(cfg); err == nil {
		t.Fatal("expected surrounding whitespace validation error")
	}
}

func TestValidateAgenticFactsSignalContractsRejectsDuplicateName(t *testing.T) {
	value := "reviewer"
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{AgenticFactsRules: []AgenticFactsRule{
			{Name: "reviewer-role", Field: AgenticFactsFieldDelegatedRole, Predicate: AgenticFactsPredicate{Equals: &value}},
			{Name: "Reviewer-Role", Field: AgenticFactsFieldDelegatedRole, Predicate: AgenticFactsPredicate{Equals: &value}},
		}},
	}}
	if err := validateAgenticFactsSignalContracts(cfg); err == nil {
		t.Fatal("expected duplicate name validation error")
	}
}

func TestDecisionLeafRejectsUnknownAgenticFactsReference(t *testing.T) {
	err := validateDecisionLeafNode(
		&RouterConfig{},
		"agentic-facts-route",
		&RuleNode{Type: SignalTypeAgenticFacts, Name: "missing"},
	)
	if err == nil {
		t.Fatal("expected unknown agentic_facts reference error")
	}
}
