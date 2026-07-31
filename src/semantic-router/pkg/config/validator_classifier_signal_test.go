package config

import "testing"

func TestValidateClassifierSignalContracts(t *testing.T) {
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{ClassifierRules: []ClassifierSignalRule{{
			Name:      "phishing",
			Type:      "local",
			ModelPath: "models/phishing",
			Labels:    []string{"BENIGN", "PHISHING"},
			UseCPU:    true,
		}}},
	}}
	if err := validateClassifierSignalContracts(cfg); err != nil {
		t.Fatalf("validateClassifierSignalContracts() error = %v", err)
	}
}

func TestValidateClassifierSignalContractsRejectsUnknownLLM(t *testing.T) {
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{ClassifierRules: []ClassifierSignalRule{{
			Name:         "risk",
			Type:         "llm",
			Model:        "missing",
			Labels:       []string{"SAFE", "RISKY"},
			Instructions: "Choose.",
		}}},
	}}
	if err := validateClassifierSignalContracts(cfg); err == nil {
		t.Fatal("expected missing external model error")
	}
}

func TestLocalClassifierDecisionPredicateRequiresWinningConfidence(t *testing.T) {
	upperBound := 0.4
	err := validateClassifierDecisionLeaf(
		&RouterConfig{IntelligentRouting: IntelligentRouting{
			Signals: Signals{ClassifierRules: []ClassifierSignalRule{{
				Name:   "risk",
				Type:   "local",
				Labels: []string{"SAFE", "RISKY"},
			}}},
		}},
		"risk-route",
		&RuleNode{
			Type:      SignalTypeClassifier,
			Name:      "risk",
			Label:     "RISKY",
			Predicate: &NumericPredicate{LTE: &upperBound},
		},
	)
	if err == nil {
		t.Fatal("expected local classifier lower-bound predicate error")
	}
}
