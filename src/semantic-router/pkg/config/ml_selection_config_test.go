package config

import (
	"strings"
	"testing"
)

func TestValidateMLSelectionAlgorithmConfigRequiresMatchingFamily(t *testing.T) {
	tests := []struct {
		name     string
		typeName string
		cfg      *MLSelectionConfig
		want     string
	}{
		{name: "missing block", typeName: DecisionAlgorithmKNN, want: "requires algorithm.ml"},
		{
			name: "missing family", typeName: DecisionAlgorithmKNN,
			cfg: &MLSelectionConfig{}, want: "requires algorithm.ml.knn",
		},
		{
			name: "wrong family", typeName: DecisionAlgorithmKNN,
			cfg: &MLSelectionConfig{SVM: &MLSVMConfig{}}, want: "requires only algorithm.ml.knn",
		},
		{
			name: "multiple families", typeName: DecisionAlgorithmKNN,
			cfg: &MLSelectionConfig{KNN: &MLKNNConfig{}, SVM: &MLSVMConfig{}}, want: "requires only algorithm.ml.knn",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := ValidateMLSelectionAlgorithmConfig(test.typeName, test.cfg)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("ValidateMLSelectionAlgorithmConfig() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestParseV03MLSelectionLivesOnRecipeDecision(t *testing.T) {
	document := strings.Replace(
		humanAuthoringFixture,
		"          rules: {operator: AND, conditions: []}",
		`          rules: {operator: AND, conditions: []}
          algorithm:
            type: knn
            ml:
              models_path: /models/selection
              embedding_dim: 1024
              knn:
                k: 5
                pretrained_path: /models/selection/knn_model.json`,
		1,
	)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	ml, err := MLSelectionConfigForRoutingProfile(cfg.ConfigForRecipe(&cfg.Recipes[0]))
	if err != nil || ml == nil || ml.KNN == nil || ml.KNN.K != 5 {
		t.Fatalf("parsed Recipe ML config = %#v, %v", ml, err)
	}
}

func TestMLSelectionConfigForRoutingProfileMergesFamilies(t *testing.T) {
	cfg := mlRoutingProfile(
		mlDecision("nearby", DecisionAlgorithmKNN, "/models/a", 1024, &MLSelectionConfig{KNN: &MLKNNConfig{K: 5}}),
		mlDecision("boundary", DecisionAlgorithmSVM, "/models/a", 1024, &MLSelectionConfig{SVM: &MLSVMConfig{Kernel: "rbf", Gamma: 1}}),
	)

	ml, err := MLSelectionConfigForRoutingProfile(scopedRoutingProfileForTest(cfg))
	if err != nil {
		t.Fatalf("MLSelectionConfigForRoutingProfile() error = %v", err)
	}
	if ml == nil || ml.KNN == nil || ml.SVM == nil || ml.KMeans != nil || ml.MLP != nil {
		t.Fatalf("aggregate ML config = %#v", ml)
	}
	if ml.ModelsPath != "/models/a" || ml.EmbeddingDim != 1024 || ml.KNN.K != 5 || ml.SVM.Kernel != "rbf" {
		t.Fatalf("aggregate ML values = %#v", ml)
	}
}

func TestMLSelectionConfigForRoutingProfileRejectsConflicts(t *testing.T) {
	tests := []struct {
		name      string
		decisions []Decision
		want      string
	}{
		{
			name: "shared settings",
			decisions: []Decision{
				mlDecision("nearby", DecisionAlgorithmKNN, "/models/a", 1024, &MLSelectionConfig{KNN: &MLKNNConfig{K: 5}}),
				mlDecision("boundary", DecisionAlgorithmSVM, "/models/b", 1024, &MLSelectionConfig{SVM: &MLSVMConfig{Kernel: "rbf"}}),
			},
			want: "conflicting algorithm.ml shared settings",
		},
		{
			name: "same family",
			decisions: []Decision{
				mlDecision("nearby-a", DecisionAlgorithmKNN, "/models/a", 1024, &MLSelectionConfig{KNN: &MLKNNConfig{K: 5}}),
				mlDecision("nearby-b", DecisionAlgorithmKNN, "/models/a", 1024, &MLSelectionConfig{KNN: &MLKNNConfig{K: 9}}),
			},
			want: "conflicting algorithm.ml.knn settings",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := MLSelectionConfigForRoutingProfile(mlRoutingProfile(test.decisions...))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("MLSelectionConfigForRoutingProfile() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestConfigForRecipeKeepsMLSelectionIsolated(t *testing.T) {
	cfg := &RouterConfig{Recipes: []RoutingRecipe{
		{
			Name: "fast",
			Profile: RoutingProfile{Decisions: []Decision{
				mlDecision("choose", DecisionAlgorithmKNN, "/models/fast", 384, &MLSelectionConfig{KNN: &MLKNNConfig{K: 3}}),
			}},
		},
		{
			Name: "deep",
			Profile: RoutingProfile{Decisions: []Decision{
				mlDecision("choose", DecisionAlgorithmKNN, "/models/deep", 1024, &MLSelectionConfig{KNN: &MLKNNConfig{K: 11}}),
			}},
		},
	}}

	fast, fastErr := MLSelectionConfigForRoutingProfile(cfg.ConfigForRecipe(&cfg.Recipes[0]))
	deep, deepErr := MLSelectionConfigForRoutingProfile(cfg.ConfigForRecipe(&cfg.Recipes[1]))
	if fastErr != nil || deepErr != nil {
		t.Fatalf("recipe-scoped ML config errors = (%v, %v)", fastErr, deepErr)
	}
	if fast.ModelsPath != "/models/fast" || fast.KNN.K != 3 || fast.EmbeddingDim != 384 {
		t.Fatalf("fast ML config = %#v", fast)
	}
	if deep.ModelsPath != "/models/deep" || deep.KNN.K != 11 || deep.EmbeddingDim != 1024 {
		t.Fatalf("deep ML config = %#v", deep)
	}
}

func mlRoutingProfile(decisions ...Decision) *RouterConfig {
	return &RouterConfig{IntelligentRouting: IntelligentRouting{Decisions: decisions}}
}

func mlDecision(name string, algorithmType string, modelsPath string, embeddingDim int, ml *MLSelectionConfig) Decision {
	ml.ModelsPath = modelsPath
	ml.EmbeddingDim = embeddingDim
	return Decision{Name: name, Algorithm: &AlgorithmConfig{Type: algorithmType, ML: ml}}
}
