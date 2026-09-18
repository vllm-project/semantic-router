package classification

import (
	"fmt"
	"math"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Orthogonal anchors make the cap's dropped tail observable without inference.
func prototypeRuleVectors() ([]string, map[string][]float32) {
	candidates := make([]string, 10)
	vectors := make(map[string][]float32)
	for i := range candidates {
		name := fmt.Sprintf("anchor%02d", i)
		vector := make([]float32, len(candidates))
		vector[i] = 1
		candidates[i] = name
		vectors[name] = vector
	}
	vectors["query"] = vectors[candidates[9]]
	return candidates, vectors
}

func TestRecipeEmbeddingPrototypeOverrideControlsBankAndScore(t *testing.T) {
	candidates, vectors := prototypeRuleVectors()
	stubEmbeddingLookup(t, vectors)
	stubMultiModalImageLookup(t, map[string][]float32{"query-image": vectors["query"]})
	disabled := false
	for _, modality := range []config.QueryModality{config.QueryModalityText, config.QueryModalityImage} {
		for _, threshold := range []float32{0.8, 0.9} {
			cfg := &config.RouterConfig{}
			cfg.EmbeddingConfig = config.HNSWConfig{PrototypeScoring: config.PrototypeScoringConfig{MaxPrototypes: 1, BestWeight: 1, TopM: 1}}
			for _, name := range []config.RecipeName{"retain", "inherit"} {
				var override *config.PrototypeScoringConfig
				if name == "retain" {
					override = &config.PrototypeScoringConfig{Enabled: &disabled, BestWeight: 0.75, TopM: 2}
				}
				cfg.Recipes = append(cfg.Recipes, config.RoutingRecipe{Name: name, Profile: config.RoutingProfile{Signals: config.Signals{
					EmbeddingRules: []config.EmbeddingRule{{
						Name: "intent", QueryModality: modality, SimilarityThreshold: threshold,
						Candidates: append(append([]string(nil), candidates...), candidates[9]), PrototypeScoring: override,
					}},
				}}})
			}
			for i := range cfg.Recipes {
				scoped := cfg.ConfigForRecipe(&cfg.Recipes[i])
				classifier := newTestEmbeddingClassifier(t, scoped.EmbeddingRules, scoped.EmbeddingConfig)
				var result *EmbeddingClassificationResult
				var err error
				if modality == config.QueryModalityImage {
					result, err = classifier.ClassifyDetailedMultimodal(modality, "query-image")
				} else {
					result, err = classifier.ClassifyDetailed("query")
				}
				if err != nil {
					t.Fatal(err)
				}
				wantCount, wantScore := 1, 0.0
				if cfg.Recipes[i].Name == "retain" {
					wantCount, wantScore = 10, 0.875
				}
				if len(result.Scores) != 1 || result.Scores[0].PrototypeCount != wantCount || math.Abs(result.Scores[0].Score-wantScore) > 1e-6 {
					t.Fatalf("%s/%s scores = %+v; want count %d score %g", cfg.Recipes[i].Name, modality, result.Scores, wantCount, wantScore)
				}
				wantMatch := wantScore >= float64(threshold)
				if (len(result.Matches) == 1) != wantMatch {
					t.Fatalf("threshold %g: matches = %+v, want match %v", threshold, result.Matches, wantMatch)
				}
			}
		}
	}
}

func TestComplexityPrototypeOverrideAppliesToAllFourBanksAndScores(t *testing.T) {
	hard, vectors := prototypeRuleVectors()
	easy := make([]string, len(hard))
	for i, name := range hard {
		easy[i] = "easy-" + name
		vector := make([]float32, len(hard))
		vector[i] = -1
		vectors[easy[i]] = vector
	}
	stubEmbeddingLookup(t, vectors)
	stubMultiModalImageLookup(t, vectors)
	original := getMultiModalTextEmbedding
	getMultiModalTextEmbedding = func(string, int) ([]float32, error) { return vectors["query"], nil }
	t.Cleanup(func() { getMultiModalTextEmbedding = original })
	disabled := false
	rules := []config.ComplexityRule{
		{Name: "retain", Threshold: 0.8, Hard: config.ComplexityCandidates{Candidates: hard, ImageCandidates: hard}, Easy: config.ComplexityCandidates{Candidates: easy, ImageCandidates: easy}, PrototypeScoring: &config.PrototypeScoringConfig{Enabled: &disabled, BestWeight: 0.75, TopM: 2}},
		{Name: "inherit", Threshold: 0.8, Hard: config.ComplexityCandidates{Candidates: hard, ImageCandidates: hard}, Easy: config.ComplexityCandidates{Candidates: easy, ImageCandidates: easy}},
	}
	classifier, err := NewComplexityClassifier(rules, "qwen3", config.PrototypeScoringConfig{MaxPrototypes: 1, BestWeight: 1, TopM: 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, rule := range rules {
		want := 1
		if rule.Name == "retain" {
			want = 10
		}
		for _, banks := range []map[string]*prototypeBank{classifier.hardPrototypeBanks, classifier.easyPrototypeBanks, classifier.imageHardPrototypeBanks, classifier.imageEasyPrototypeBanks} {
			if got := len(banks[rule.Name].prototypes); got != want {
				t.Fatalf("%s bank has %d prototypes, want %d", rule.Name, got, want)
			}
		}
	}
	for _, image := range []string{"", "query"} {
		results, classifyErr := classifier.ClassifyDetailedWithImage("query", image)
		if classifyErr != nil {
			t.Fatal(classifyErr)
		}
		if len(results) != 2 {
			t.Fatalf("got %d results", len(results))
		}
		if results[0].Difficulty != "hard" || math.Abs(results[0].TextHardScore-0.875) > 1e-6 || results[1].Difficulty != "medium" {
			t.Fatalf("bank/score settings disagree: %+v", results)
		}
		if image != "" && math.Abs(results[0].ImageHardScore-0.875) > 1e-6 {
			t.Fatalf("image score did not use rule override: %+v", results[0])
		}
	}
}
