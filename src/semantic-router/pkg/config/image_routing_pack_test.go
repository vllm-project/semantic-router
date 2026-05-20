package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

// TestImageRoutingPack_StructuralContract loads the shipped image-modality
// embedding pack and asserts the contract operators rely on when inlining it
// into a recipe: rule names, threshold, aggregation method, query modality,
// candidate count. Drift in any of these would change downstream routing
// behavior silently, so they're pinned here.
func TestImageRoutingPack_StructuralContract(t *testing.T) {
	rules := loadImageRoutingPackRules(t)

	expectedNames := map[string]bool{
		"identifier_document_imagery": false,
		"code_or_terminal_imagery":    false,
		"ambient_office_imagery":      false,
	}
	if got, want := len(rules), len(expectedNames); got != want {
		t.Fatalf("image-routing.yaml: rule count = %d, want %d", got, want)
	}
	for _, rule := range rules {
		if _, ok := expectedNames[rule.Name]; !ok {
			t.Errorf("unexpected rule name %q in image-routing.yaml", rule.Name)
			continue
		}
		expectedNames[rule.Name] = true

		if got, want := rule.QueryModality, QueryModalityImage; got != want {
			t.Errorf("rule %q: query_modality = %q, want %q", rule.Name, got, want)
		}
		// Threshold is operator-tunable and is calibrated against the
		// shipped multimodal embedding model; range-bound it rather
		// than pinning an exact value so future calibration changes
		// don't trip an unrelated structural test. The bounds catch
		// silent regression to the original 0.70 (no rules fire) or
		// accidental zero (every rule fires).
		if rule.SimilarityThreshold <= 0 || rule.SimilarityThreshold >= 0.5 {
			t.Errorf("rule %q: threshold = %v, want (0, 0.5) consistent with the calibrated image-modality range", rule.Name, rule.SimilarityThreshold)
		}
		if got, want := rule.AggregationMethodConfiged, AggregationMethodMax; got != want {
			t.Errorf("rule %q: aggregation_method = %q, want %q", rule.Name, got, want)
		}
		// This pack ships 8 candidates per rule. Pinning the floor at
		// 8 catches silent reduction during future edits. Other shipped
		// fragment files (e.g., config/signal/embedding/support.yaml)
		// ship fewer candidates and are not subject to this pin.
		if got := len(rule.Candidates); got < 8 {
			t.Errorf("rule %q: candidate count = %d, want at least 8 (this pack's shipped per-rule floor)", rule.Name, got)
		}
	}
	for name, seen := range expectedNames {
		if !seen {
			t.Errorf("image-routing.yaml is missing expected rule %q", name)
		}
	}
}

// TestImageRoutingPack_ValidatorAcceptsUnderMultimodal confirms the shipped
// pack passes the embedding-modality validator when paired with a multimodal
// embedding model (the supported configuration).
func TestImageRoutingPack_ValidatorAcceptsUnderMultimodal(t *testing.T) {
	rules := loadImageRoutingPackRules(t)
	if err := validateEmbeddingRuleModalities(rules, "multimodal"); err != nil {
		t.Fatalf("image-routing.yaml: validator should accept under model_type=multimodal, got: %v", err)
	}
}

// TestImageRoutingPack_ValidatorRejectsUnderTextOnlyModel confirms the
// shipped pack triggers the validator's image-rule-without-multimodal-model
// rejection path. This is the init-rejection contract documented in
// embedding.md and exercised here against the actual shipped rules so the
// rejection surface stays load-bearing under config drift.
func TestImageRoutingPack_ValidatorRejectsUnderTextOnlyModel(t *testing.T) {
	rules := loadImageRoutingPackRules(t)
	err := validateEmbeddingRuleModalities(rules, "qwen3")
	if err == nil {
		t.Fatal("image-routing.yaml: validator should reject image rules paired with non-multimodal model_type, got nil")
	}
	msg := err.Error()
	for _, want := range []string{"identifier_document_imagery", "code_or_terminal_imagery", "ambient_office_imagery", "model_type=multimodal"} {
		if !strings.Contains(msg, want) {
			t.Errorf("rejection error should mention %q, got: %s", want, msg)
		}
	}
}

// loadImageRoutingPackRules reads config/signal/embedding/image-routing.yaml
// from the repo root and returns the parsed embedding rules. Test-internal
// helper; production code paths use the full canonical config loader.
func loadImageRoutingPackRules(t *testing.T) []EmbeddingRule {
	t.Helper()
	root := repoRootFromTestFile(t)
	path := filepath.Join(root, "config", "signal", "embedding", "image-routing.yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read %s: %v", path, err)
	}
	var doc CanonicalConfig
	if err := yaml.Unmarshal(data, &doc); err != nil {
		t.Fatalf("failed to parse %s as canonical config fragment: %v", path, err)
	}
	if len(doc.Routing.Signals.Embeddings) == 0 {
		t.Fatalf("%s parsed cleanly but contains zero embedding rules under routing.signals.embeddings", path)
	}
	return doc.Routing.Signals.Embeddings
}
