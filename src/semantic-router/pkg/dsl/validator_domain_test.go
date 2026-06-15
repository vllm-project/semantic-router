package dsl

import (
	"strings"
	"testing"
)

func TestValidateDomainSignalRejectsUnsupportedImplicitDomainName(t *testing.T) {
	input := `
SIGNAL domain balance_demo_compact {}
PROJECTION partition domain_partition {
  semantics: "softmax_exclusive"
  temperature: 0.1
  members: ["balance_demo_compact"]
  default: "balance_demo_compact"
}
ROUTE r1 { PRIORITY 100 WHEN domain("balance_demo_compact") MODEL "m1" }
`
	diags, _ := Validate(input)
	if !hasConstraintContaining(diags, "supported routing domain name") {
		t.Fatalf("expected supported routing domains constraint, got %v", diags)
	}
}

func TestValidateDomainSignalRejectsUnsupportedMMLUCategory(t *testing.T) {
	input := `
SIGNAL domain compact { mmlu_categories: ["computer_science"] }
ROUTE r1 { PRIORITY 100 WHEN domain("compact") MODEL "m1" }
`
	diags, _ := Validate(input)
	if !hasConstraintContaining(diags, "computer_science") {
		t.Fatalf("expected unsupported mmlu_categories constraint, got %v", diags)
	}
}

func TestCompileDomainSignalRejectsUnsupportedMMLUCategory(t *testing.T) {
	input := `
SIGNAL domain compact { mmlu_categories: ["computer_science"] }
ROUTE r1 { PRIORITY 100 WHEN domain("compact") MODEL "m1" }
`
	_, errs := Compile(input)
	if len(errs) == 0 {
		t.Fatal("expected compile error for unsupported mmlu category")
	}
}

func TestCompileDomainSignalRejectsUnsupportedSoftmaxGroupImplicitDomainName(t *testing.T) {
	input := `
SIGNAL domain balance_demo_compact {}
PROJECTION partition domain_partition {
  semantics: "softmax_exclusive"
  temperature: 0.1
  members: ["balance_demo_compact"]
  default: "balance_demo_compact"
}
ROUTE r1 { PRIORITY 100 WHEN domain("balance_demo_compact") MODEL "m1" }
`
	_, errs := Compile(input)
	if len(errs) == 0 {
		t.Fatal("expected compile error for unsupported softmax group domain member")
	}
}

func TestCompileDomainSignalAllowsAliasWithSupportedMMLUCategory(t *testing.T) {
	input := `
SIGNAL domain compact { mmlu_categories: ["computer science"] }
ROUTE r1 { PRIORITY 100 WHEN domain("compact") MODEL "m1" }
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("unexpected compile errors: %v", errs)
	}
	if len(cfg.Categories) != 1 || cfg.Categories[0].Name != "compact" {
		t.Fatalf("unexpected categories: %+v", cfg.Categories)
	}
}

func hasConstraintContaining(diags []Diagnostic, needle string) bool {
	for _, diag := range diags {
		if diag.Level == DiagConstraint && strings.Contains(diag.Message, needle) {
			return true
		}
	}
	return false
}
