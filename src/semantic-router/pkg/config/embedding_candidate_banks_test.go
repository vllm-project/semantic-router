package config

import "testing"

func TestEmbeddingCandidateBanksRequirementsAndValidation(t *testing.T) {
	rule := EmbeddingRule{Name: "image-bank", ImageCandidates: []string{"./positive.png"}, NegativeImageCandidates: []string{"./negative.png"}, SimilarityThreshold: -0.1}
	if err := validateEmbeddingRuleModalities([]EmbeddingRule{rule}, "mmbert", false); err == nil {
		t.Fatal("image prototypes need an image encoder even with text query")
	}
	if err := validateEmbeddingRuleModalities([]EmbeddingRule{rule}, "mmbert", true); err != nil {
		t.Fatal(err)
	}
	cfg := &RouterConfig{}
	cfg.EmbeddingRules = []EmbeddingRule{rule}
	requirements := EmbeddingRequirements(cfg, "owned", false)
	foundImage, foundText := false, false
	for _, r := range requirements {
		if r.Model != "owned" {
			t.Fatalf("lost provider identity: %+v", r)
		}
		foundImage = foundImage || r.Modality == "image"
		foundText = foundText || r.Modality == "text"
	}
	if !foundImage || !foundText {
		t.Fatalf("missing candidate/query requirements: %+v", requirements)
	}
	rule.ImageCandidates = nil
	if err := validateEmbeddingRuleModalities([]EmbeddingRule{rule}, "multimodal", false); err == nil {
		t.Fatal("negative-only bank accepted")
	}
}
