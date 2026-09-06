package config

import "testing"

func TestValidateMetadataContracts(t *testing.T) {
	denied := "denied"
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{MetadataRules: []MetadataRule{{
			Name:      "consent-denied",
			Key:       "consent",
			Predicate: MetadataPredicate{Equals: &denied},
		}}},
	}}
	if err := validateMetadataContracts(cfg); err != nil {
		t.Fatalf("validateMetadataContracts() error = %v", err)
	}
}

func TestValidateMetadataContractsRejectsAmbiguousPredicate(t *testing.T) {
	denied := "denied"
	exists := true
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{MetadataRules: []MetadataRule{{
			Name: "bad",
			Key:  "consent",
			Predicate: MetadataPredicate{
				Equals: &denied,
				Exists: &exists,
			},
		}}},
	}}
	if err := validateMetadataContracts(cfg); err == nil {
		t.Fatal("validateMetadataContracts() expected ambiguous predicate error")
	}
}

func TestValidateMetadataContractsRejectsWhitespaceName(t *testing.T) {
	value := "canary"
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{
		Signals: Signals{MetadataRules: []MetadataRule{{
			Name:      " canary ",
			Key:       "cohort",
			Predicate: MetadataPredicate{Equals: &value},
		}}},
	}}
	if err := validateMetadataContracts(cfg); err == nil {
		t.Fatal("expected surrounding whitespace validation error")
	}
}

func TestDecisionLeafRejectsUnknownMetadataReference(t *testing.T) {
	err := validateDecisionLeafNode(
		&RouterConfig{},
		"metadata-route",
		&RuleNode{Type: SignalTypeMetadata, Name: "missing"},
	)
	if err == nil {
		t.Fatal("expected unknown metadata reference error")
	}
}
