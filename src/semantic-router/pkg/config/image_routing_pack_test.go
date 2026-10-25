package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
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
		// accidental zero (every rule fires) and a value no image can
		// reach (no rule ever fires).
		if rule.SimilarityThreshold <= 0 || rule.SimilarityThreshold >= 0.9 {
			t.Errorf("rule %q: threshold = %v, want (0, 0.9) consistent with the calibrated image-modality range", rule.Name, rule.SimilarityThreshold)
		}
		if got, want := rule.AggregationMethodConfiged, AggregationMethodMax; got != want {
			t.Errorf("rule %q: aggregation_method = %q, want %q", rule.Name, got, want)
		}
		// The pack is calibrated with every cross-modal anchor retained and
		// raw-max scoring. A per-rule override changes that policy.
		if rule.PrototypeScoring != nil {
			t.Errorf("rule %q: carries a prototype_scoring override; calibrated thresholds assume the image default", rule.Name)
		}
		policy := rule.EffectivePrototypeScoring(PrototypeScoringConfig{}.WithDefaults())
		if policy.IsEnabled() || policy.BestWeight != 1 || policy.TopM != 1 {
			t.Errorf("rule %q: image policy must preserve all candidates and score their raw maximum", rule.Name)
		}
		if rule.Name == "code_or_terminal_imagery" {
			if len(rule.Candidates) != 0 || len(rule.ImageCandidates) != 7 || len(rule.NegativeImageCandidates) != 178 {
				t.Errorf("code rule must preserve the frozen development image banks: %+v", rule)
			}
		} else if len(rule.Candidates) != 8 {
			t.Errorf("rule %q must retain eight positive text anchors", rule.Name)
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
	if err := validateEmbeddingRuleModalities(rules, "multimodal", false); err != nil {
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
	err := validateEmbeddingRuleModalities(rules, "qwen3", false)
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

func TestImageRoutingPack_MatchesMultimodalE2EProfile(t *testing.T) {
	canonical := loadImageRoutingPackRules(t)
	root := repoRootFromTestFile(t)
	path := filepath.Join(root, "e2e", "profiles", "multimodal-routing", "crds", "intelligentroute.yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read %s: %v", path, err)
	}
	var document struct {
		Spec struct {
			Signals struct {
				Embeddings []struct {
					Name                    string   `yaml:"name"`
					QueryModality           string   `yaml:"queryModality"`
					Threshold               float32  `yaml:"threshold"`
					AggregationMethod       string   `yaml:"aggregationMethod"`
					Candidates              []string `yaml:"candidates"`
					ImageCandidates         []string `yaml:"imageCandidates"`
					NegativeCandidates      []string `yaml:"negativeCandidates"`
					NegativeImageCandidates []string `yaml:"negativeImageCandidates"`
					// Any override block, even empty, is a mismatch with
					// the pack the gate calibrated.
					PrototypeScoring map[string]interface{} `yaml:"prototypeScoring"`
				} `yaml:"embeddings"`
			} `yaml:"signals"`
		} `yaml:"spec"`
	}
	if err := yaml.Unmarshal(data, &document); err != nil {
		t.Fatalf("failed to parse %s: %v", path, err)
	}
	if got, want := len(document.Spec.Signals.Embeddings), len(canonical); got != want {
		t.Fatalf("E2E profile embedding rule count = %d, want %d", got, want)
	}
	for i, rule := range canonical {
		mirror := document.Spec.Signals.Embeddings[i]
		if mirror.Name != rule.Name || mirror.QueryModality != string(rule.QueryModality) ||
			mirror.Threshold != rule.SimilarityThreshold ||
			mirror.AggregationMethod != string(rule.AggregationMethodConfiged) ||
			!slices.Equal(mirror.Candidates, rule.Candidates) || !slices.Equal(mirror.ImageCandidates, rule.ImageCandidates) || !slices.Equal(mirror.NegativeCandidates, rule.NegativeCandidates) || !slices.Equal(mirror.NegativeImageCandidates, rule.NegativeImageCandidates) {
			t.Errorf("E2E profile rule %q does not match canonical image-routing.yaml", rule.Name)
		}
		if mirror.PrototypeScoring != nil {
			t.Errorf("E2E profile rule %q carries a prototypeScoring override; calibrated thresholds assume the image default", rule.Name)
		}
	}
}

// loadImageRoutingPackRules reads config/fragments/signal/embedding/image-routing.yaml
// from the repo root and returns the parsed embedding rules. Test-internal
// helper; production code paths use the full canonical config loader.
func loadImageRoutingPackRules(t *testing.T) []EmbeddingRule {
	t.Helper()
	root := repoRootFromTestFile(t)
	path := filepath.Join(root, "config", "fragments", "signal", "embedding", "image-routing.yaml")
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

// The shipped banks are tied to the frozen split by content identity, not just
// path names. Real asset packaging verifies each file against these digests.
func TestImageRoutingPrototypesExcludeValidationContent(t *testing.T) {
	root := repoRootFromTestFile(t)
	var manifest struct {
		RuntimeDirectory string `json:"runtime_directory"`
		Assets           []struct{ Source, SHA256, Asset, Role string }
	}
	data, err := os.ReadFile(filepath.Join(root, "config/assets/image-routing/manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	var protocol struct {
		Split         struct{ Assignment map[string]string }
		ContentSHA256 map[string]string `json:"content_sha256"`
	}
	data, err = os.ReadFile(filepath.Join(root, "tools/calibration/image-routing/testdata/prototype-protocol.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(data, &protocol); err != nil {
		t.Fatal(err)
	}
	validation := map[string]bool{}
	for path, split := range protocol.Split.Assignment {
		if split == "validation" {
			validation[protocol.ContentSHA256[path]] = true
		}
	}
	references := map[string]string{}
	positive, negative := 0, 0
	for _, asset := range manifest.Assets {
		if protocol.Split.Assignment[asset.Source] != "development" || validation[asset.SHA256] || protocol.ContentSHA256[asset.Source] != asset.SHA256 {
			t.Fatalf("prototype leaks held-out content or differs from protocol: %+v", asset)
		}
		references[filepath.Join(manifest.RuntimeDirectory, asset.Asset)] = asset.Role
		if asset.Role == "positive" {
			positive++
		} else {
			negative++
		}
	}
	if positive != 7 || negative != 178 {
		t.Fatalf("frozen bank inventory differs: %d/%d", positive, negative)
	}
	for _, rule := range loadImageRoutingPackRules(t) {
		for _, ref := range rule.ImageCandidates {
			if references[ref] != "positive" {
				t.Fatalf("positive reference not packaged: %s", ref)
			}
		}
		for _, ref := range rule.NegativeImageCandidates {
			if references[ref] != "negative" {
				t.Fatalf("negative reference not packaged: %s", ref)
			}
		}
	}
}
