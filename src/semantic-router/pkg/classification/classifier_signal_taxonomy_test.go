package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestTaxonomySignalMatchConfidenceTierUsesBestMatchedTier(t *testing.T) {
	result := &TaxonomyClassifyResult{
		BestCategory:          "generic_coding",
		BestSimilarity:        0.99,
		BestTier:              "local_standard",
		BestMatchedCategory:   "generic_coding",
		BestMatchedSimilarity: 0.99,
		BestMatchedTier:       "local_standard",
		MatchedCategories:     []string{"prompt_injection", "generic_coding"},
		MatchedTiers:          []string{"security_containment", "local_standard"},
		CategoryTiers: map[string]string{
			"prompt_injection": "security_containment",
			"generic_coding":   "local_standard",
		},
		CategoryConfidences: map[string]float64{
			"prompt_injection": 0.80,
			"generic_coding":   0.99,
		},
	}

	securityRule := config.TaxonomySignalRule{
		Name:       "security_containment",
		Classifier: "privacy_classifier",
		Bind: config.TaxonomySignalBind{
			Kind:  config.TaxonomyBindKindTier,
			Value: "security_containment",
		},
	}
	if confidence, matched := taxonomySignalMatchConfidence(securityRule, result); matched || confidence != 0 {
		t.Fatalf("security tier should not match when best matched tier is local_standard, got matched=%v confidence=%.2f", matched, confidence)
	}

	localRule := config.TaxonomySignalRule{
		Name:       "local_standard",
		Classifier: "privacy_classifier",
		Bind: config.TaxonomySignalBind{
			Kind:  config.TaxonomyBindKindTier,
			Value: "local_standard",
		},
	}
	confidence, matched := taxonomySignalMatchConfidence(localRule, result)
	if !matched {
		t.Fatal("local_standard tier should match when it is the best matched tier")
	}
	if confidence != 0.99 {
		t.Fatalf("local_standard confidence = %.2f, want 0.99", confidence)
	}
}

func TestTaxonomySignalMatchConfidenceTierRequiresThresholdMatchedCategory(t *testing.T) {
	result := &TaxonomyClassifyResult{
		BestCategory:        "generic_coding",
		BestSimilarity:      0.42,
		BestTier:            "local_standard",
		MatchedCategories:   nil,
		MatchedTiers:        nil,
		CategoryTiers:       map[string]string{"generic_coding": "local_standard"},
		CategoryConfidences: map[string]float64{"generic_coding": 0.42},
	}

	rule := config.TaxonomySignalRule{
		Name:       "local_standard",
		Classifier: "privacy_classifier",
		Bind: config.TaxonomySignalBind{
			Kind:  config.TaxonomyBindKindTier,
			Value: "local_standard",
		},
	}
	if confidence, matched := taxonomySignalMatchConfidence(rule, result); matched || confidence != 0 {
		t.Fatalf("tier match should require a threshold-qualified best matched category, got matched=%v confidence=%.2f", matched, confidence)
	}
}
