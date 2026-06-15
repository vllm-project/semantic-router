package config

import (
	"strings"
	"testing"
)

func validConversationRule() ConversationRule {
	return ConversationRule{
		Name: "multi_turn",
		Feature: ConversationFeature{
			Type:   "count",
			Source: ConversationSource{Type: "message", Role: "user"},
		},
		Predicate: &NumericPredicate{GTE: ptrFloat64(2)},
	}
}

func ptrFloat64(v float64) *float64 { return &v }

func TestValidateConversationRuleContract_ValidRule(t *testing.T) {
	if err := ValidateConversationRuleContract(validConversationRule()); err != nil {
		t.Fatalf("expected valid rule to pass, got: %v", err)
	}
}

func TestValidateConversationRuleContract_EmptyName(t *testing.T) {
	rule := validConversationRule()
	rule.Name = ""
	err := ValidateConversationRuleContract(rule)
	if err == nil {
		t.Fatal("expected error for empty name")
	}
	if !strings.Contains(err.Error(), "name is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateConversationRuleContract_BadFeatureType(t *testing.T) {
	rule := validConversationRule()
	rule.Feature.Type = "char_count"
	err := ValidateConversationRuleContract(rule)
	if err == nil {
		t.Fatal("expected error for unsupported feature.type")
	}
	if !strings.Contains(err.Error(), "unsupported feature.type") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateConversationRuleContract_BadSourceType(t *testing.T) {
	rule := validConversationRule()
	rule.Feature.Source.Type = "paragraph"
	err := ValidateConversationRuleContract(rule)
	if err == nil {
		t.Fatal("expected error for unsupported source.type")
	}
	if !strings.Contains(err.Error(), "unsupported feature.source.type") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateConversationRuleContract_BadSourceRole(t *testing.T) {
	rule := validConversationRule()
	rule.Feature.Source.Role = "moderator"
	err := ValidateConversationRuleContract(rule)
	if err == nil {
		t.Fatal("expected error for unsupported source.role")
	}
	if !strings.Contains(err.Error(), "unsupported source.role") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateConversationRuleContract_RoleOnNonMessage(t *testing.T) {
	rule := validConversationRule()
	rule.Feature.Source = ConversationSource{Type: "tool_definition", Role: "user"}
	err := ValidateConversationRuleContract(rule)
	if err == nil {
		t.Fatal("expected error when role is set on non-message source")
	}
	if !strings.Contains(err.Error(), "source.role is only valid") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateConversationRuleContract_ExistsWithPredicate(t *testing.T) {
	rule := validConversationRule()
	rule.Feature.Type = "exists"
	rule.Predicate = &NumericPredicate{GTE: ptrFloat64(1)}
	err := ValidateConversationRuleContract(rule)
	if err == nil {
		t.Fatal("expected error for exists with predicate")
	}
	if !strings.Contains(err.Error(), "does not accept a predicate") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateConversationRuleContract_ConflictingPredicates(t *testing.T) {
	rule := validConversationRule()
	rule.Predicate = &NumericPredicate{GT: ptrFloat64(1), GTE: ptrFloat64(2)}
	err := ValidateConversationRuleContract(rule)
	if err == nil {
		t.Fatal("expected error for gt+gte conflict")
	}
	if !strings.Contains(err.Error(), "both gt and gte") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateConversationContracts_DuplicateNames(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				ConversationRules: []ConversationRule{
					validConversationRule(),
					validConversationRule(),
				},
			},
		},
	}
	err := validateConversationContracts(cfg)
	if err == nil {
		t.Fatal("expected error for duplicate rule names")
	}
	if !strings.Contains(err.Error(), "duplicate name") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateConversationRuleContract_ExistsWithoutPredicate(t *testing.T) {
	rule := ConversationRule{
		Name: "has_dev",
		Feature: ConversationFeature{
			Type:   "exists",
			Source: ConversationSource{Type: "message", Role: "developer"},
		},
	}
	if err := ValidateConversationRuleContract(rule); err != nil {
		t.Fatalf("expected exists without predicate to pass, got: %v", err)
	}
}
