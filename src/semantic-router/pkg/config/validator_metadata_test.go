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
