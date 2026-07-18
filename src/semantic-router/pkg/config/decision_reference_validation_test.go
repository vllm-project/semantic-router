package config

import (
	"strings"
	"testing"
)

// buildDecisionRefConfig wraps a decisions block in an otherwise-valid v0.3
// canonical config with a single backed model card "m1".
func buildDecisionRefConfig(decisions string) []byte {
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
          api_key: secret
routing:
  modelCards:
    - name: m1
  decisions:
` + decisions)
}

// Sanity: confirm validateCanonicalContract runs on the ParseYAMLBytes path
// (an unknown default_model is already rejected today).
func TestUnknownDefaultModelRejected_Sanity(t *testing.T) {
	yaml := []byte(`
version: v0.3
listeners: []
providers:
  defaults:
    default_model: ghost
  models:
    - name: m1
      backend_refs:
        - endpoint: 127.0.0.1:8000
          api_key: secret
routing:
  modelCards:
    - name: m1
  decisions:
    - name: d1
      priority: 1
      rules: {operator: AND, conditions: []}
      modelRefs:
        - model: m1
          use_reasoning: false
`)
	if _, err := ParseYAMLBytes(yaml); err == nil {
		t.Fatal("expected unknown default_model to be rejected, got nil error")
	}
}

// G1: a decision modelRef pointing at a model not in modelCards must be
// rejected at config load. Today this is only checked when a lora_name is
// also set, so a plain modelRef to an unknown model slips through and only
// fails at request time with a misleading upstream 401.
func TestDecisionModelRefUnknownModelRejected(t *testing.T) {
	cfg := buildDecisionRefConfig(`    - name: d1
      priority: 1
      rules: {operator: AND, conditions: []}
      modelRefs:
        - model: ghost-model
          use_reasoning: false
`)
	_, err := ParseYAMLBytes(cfg)
	if err == nil {
		t.Fatal("expected unknown modelRef model to be rejected, got nil error")
	}
	if !strings.Contains(err.Error(), "ghost-model") {
		t.Fatalf("error should name the unknown model, got: %v", err)
	}
}

// G1 baseline: a modelRef to a known model card must still be accepted.
func TestDecisionModelRefKnownModelAccepted(t *testing.T) {
	cfg := buildDecisionRefConfig(`    - name: d1
      priority: 1
      rules: {operator: AND, conditions: []}
      modelRefs:
        - model: m1
          use_reasoning: false
`)
	if _, err := ParseYAMLBytes(cfg); err != nil {
		t.Fatalf("known modelRef model should be accepted, got: %v", err)
	}
}

// G2: two decisions sharing the same name are ambiguous and must be rejected.
func TestDuplicateDecisionNamesRejected(t *testing.T) {
	cfg := buildDecisionRefConfig(`    - name: dup
      priority: 100
      rules: {operator: AND, conditions: []}
      modelRefs:
        - model: m1
          use_reasoning: false
    - name: dup
      priority: 50
      rules: {operator: AND, conditions: []}
      modelRefs:
        - model: m1
          use_reasoning: false
`)
	_, err := ParseYAMLBytes(cfg)
	if err == nil {
		t.Fatal("expected duplicate decision names to be rejected, got nil error")
	}
	if !strings.Contains(err.Error(), "dup") {
		t.Fatalf("error should name the duplicate decision, got: %v", err)
	}
}

// G2 baseline: distinct decision names must be accepted.
func TestDistinctDecisionNamesAccepted(t *testing.T) {
	cfg := buildDecisionRefConfig(`    - name: d1
      priority: 100
      rules: {operator: AND, conditions: []}
      modelRefs:
        - model: m1
          use_reasoning: false
    - name: d2
      priority: 50
      rules: {operator: AND, conditions: []}
      modelRefs:
        - model: m1
          use_reasoning: false
`)
	if _, err := ParseYAMLBytes(cfg); err != nil {
		t.Fatalf("distinct decision names should be accepted, got: %v", err)
	}
}
