package config

import "testing"

func TestEmbeddingPrototypePolicyFollowsModalityAndExplicitOverride(t *testing.T) {
	family := PrototypeScoringConfig{BestWeight: 0.8, MaxPrototypes: 2, TopM: 3}.WithDefaults()
	for _, modality := range []QueryModality{"", QueryModalityText, QueryModalityImage, QueryModalityAudio} {
		rule := EmbeddingRule{QueryModality: modality}
		got := rule.EffectivePrototypeScoring(family)
		media := modality == QueryModalityImage || modality == QueryModalityAudio
		if media {
			if got.IsEnabled() || got.BestWeight != 1 || got.TopM != 1 {
				t.Fatalf("%s must retain all cross-modal anchors and raw-max score: %+v", modality, got)
			}
		} else if !got.IsEnabled() || got.BestWeight != family.BestWeight || got.TopM != family.TopM || got.MaxPrototypes != family.MaxPrototypes {
			t.Fatalf("text policy must inherit the family config: %+v", got)
		}
		rule.PrototypeScoring = &PrototypeScoringConfig{}
		got = rule.EffectivePrototypeScoring(family)
		if !got.IsEnabled() || got.BestWeight != 0.75 || got.TopM != 2 {
			t.Fatalf("explicit rule override must opt into its complete policy: %+v", got)
		}
	}
}

func TestEmbeddingTextQueryPreservesImageAnchors(t *testing.T) {
	rule := EmbeddingRule{Candidates: []string{"positive"}, NegativeImageCandidates: []string{"./negative.png"}}
	if rule.EffectivePrototypeScoring(PrototypeScoringConfig{}).IsEnabled() {
		t.Fatal("cross-modal candidate compression must be explicit")
	}
	rule.PrototypeScoring = &PrototypeScoringConfig{}
	if !rule.EffectivePrototypeScoring(PrototypeScoringConfig{}).IsEnabled() {
		t.Fatal("explicit compression ignored")
	}
}
