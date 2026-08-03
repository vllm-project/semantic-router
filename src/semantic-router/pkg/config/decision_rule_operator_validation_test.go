package config

import (
	"strings"
	"testing"
)

func decisionOperatorConfig(rulesBlock string) []byte {
	return []byte(`
version: v0.3
listeners: []
providers:
  defaults:
    default_model: m1
  models:
    - name: m1
      backend_refs:
        - endpoint: 127.0.0.1:8000
          api_key: k
routing:
  modelCards:
    - name: m1
  decisions:
    - name: d1
      priority: 1
      rules:
` + rulesBlock + `
      modelRefs:
        - model: m1
          use_reasoning: false
`)
}

// A decision rule operator is a closed set (evalComposerNode handles AND/OR/NOT;
// anything else is silently treated as AND). An invalid, non-empty operator must
// be rejected at config load rather than silently changing routing semantics.
func TestDecisionRuleOperatorRejectsInvalid(t *testing.T) {
	cfg := decisionOperatorConfig("        operator: XOR\n        conditions: []")
	_, err := ParseYAMLBytes(cfg)
	if err == nil {
		t.Fatal("expected invalid rule operator XOR to be rejected, got nil")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "operator") {
		t.Fatalf("error should mention operator, got: %v", err)
	}
}

func TestDecisionRuleOperatorAcceptsValid(t *testing.T) {
	for _, op := range []string{"AND", "OR", "NOT", "and", "or"} {
		cfg := decisionOperatorConfig("        operator: " + op + "\n        conditions: []")
		if _, err := ParseYAMLBytes(cfg); err != nil {
			t.Fatalf("operator %q should be accepted, got: %v", op, err)
		}
	}
}

// An omitted operator on a combination node defaults to AND at runtime and must
// remain valid.
func TestDecisionRuleOperatorEmptyAccepted(t *testing.T) {
	cfg := decisionOperatorConfig("        conditions: []")
	if _, err := ParseYAMLBytes(cfg); err != nil {
		t.Fatalf("omitted operator should be accepted, got: %v", err)
	}
}

// Invalid operators on nested combination nodes must also be rejected.
func TestDecisionRuleOperatorNestedRejected(t *testing.T) {
	cfg := decisionOperatorConfig("        operator: AND\n        conditions:\n          - operator: NONSENSE\n            conditions: []")
	_, err := ParseYAMLBytes(cfg)
	if err == nil {
		t.Fatal("expected nested invalid rule operator to be rejected, got nil")
	}
}
