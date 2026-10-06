package config

import (
	"maps"
	"testing"
)

func TestEmbeddingModelsNeededUsesMLRequest(t *testing.T) {
	for _, algorithm := range []string{"knn", "kmeans", "svm", "mlp"} {
		t.Run(algorithm, func(t *testing.T) {
			cfg := &RouterConfig{RouterOptions: RouterOptions{AutoModelNames: []string{"auto"}}}
			cfg.Decisions = []Decision{{Algorithm: &AlgorithmConfig{Type: algorithm}}}
			cfg.ModelSelection.ML.ModelsPath = "models/ml"
			cfg.ModelSelection.ML.ModelType = "  Qwen3  "
			if got := EmbeddingModelsNeeded(cfg, "mmbert", false); !maps.Equal(got, map[string]bool{"qwen3": true}) {
				t.Fatalf("ML embedding needs = %v, want qwen3", got)
			}
			cfg.Tools.Enabled = true
			if got := EmbeddingModelsNeeded(cfg, "mmbert", true); !maps.Equal(got, map[string]bool{"qwen3": true, "mmbert": true}) {
				t.Fatalf("shared primary consumer lost its embedding: %v", got)
			}
			if cfg.ModelSelection.ML.ModelType != "  Qwen3  " {
				t.Fatal("preparation mutated the configured model type")
			}
		})
	}
}

func TestEmbeddingModelsNeededMLDefaultsAndInactiveConfig(t *testing.T) {
	cases := []struct {
		name      string
		algorithm string
		ml        MLSelectionConfig
		want      map[string]bool
	}{
		{name: "empty config", algorithm: "knn", want: map[string]bool{"mmbert": true}},
		{name: "legacy dimension only", algorithm: "svm", ml: MLSelectionConfig{ModelsPath: "models/ml", EmbeddingDim: 1024}, want: map[string]bool{"mmbert": true}},
		{name: "model without artifacts", algorithm: "svm", ml: MLSelectionConfig{ModelType: "qwen3"}, want: map[string]bool{"mmbert": true}},
		{name: "whitespace model", algorithm: "svm", ml: MLSelectionConfig{ModelsPath: "models/ml", ModelType: "  "}, want: map[string]bool{"mmbert": true}},
		{name: "unused ML config", algorithm: "static", ml: MLSelectionConfig{ModelsPath: "models/ml", ModelType: "qwen3"}, want: map[string]bool{}},
		{name: "other embedding consumer", algorithm: "router_dc", ml: MLSelectionConfig{ModelsPath: "models/ml", ModelType: "qwen3"}, want: map[string]bool{"mmbert": true}},
		{name: "KNN artifact only", algorithm: "knn", ml: MLSelectionConfig{ModelType: "qwen3", KNN: MLKNNConfig{PretrainedPath: "knn.json"}}, want: map[string]bool{"qwen3": true}},
		{name: "KMeans artifact only", algorithm: "kmeans", ml: MLSelectionConfig{ModelType: "qwen3", KMeans: MLKMeansConfig{PretrainedPath: "kmeans.json"}}, want: map[string]bool{"qwen3": true}},
		{name: "SVM artifact only", algorithm: "svm", ml: MLSelectionConfig{ModelType: "qwen3", SVM: MLSVMConfig{PretrainedPath: "svm.json"}}, want: map[string]bool{"qwen3": true}},
		{name: "MLP artifact only", algorithm: "mlp", ml: MLSelectionConfig{ModelType: "qwen3", MLP: MLMLPConfig{PretrainedPath: "mlp.json"}}, want: map[string]bool{"qwen3": true}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &RouterConfig{RouterOptions: RouterOptions{AutoModelNames: []string{"auto"}}}
			cfg.Decisions = []Decision{{Algorithm: &AlgorithmConfig{Type: tc.algorithm}}}
			cfg.ModelSelection.ML = tc.ml
			if got := EmbeddingModelsNeeded(cfg, "mmbert", false); !maps.Equal(got, tc.want) {
				t.Fatalf("embedding needs = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestEmbeddingModelsNeededKeepsConfiguredPrimaryWarmup(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.ModelSelection.Enabled = true
	cfg.ModelSelection.ML.ModelsPath = "models/ml"
	cfg.ModelSelection.ML.ModelType = "qwen3"
	if got := EmbeddingModelsNeeded(cfg, "mmbert", true); !maps.Equal(got, map[string]bool{"mmbert": true}) {
		t.Fatalf("configured primary warmup changed without an ML decision: %v", got)
	}
}

func TestEmbeddingModelsNeededMLRecipeIsolation(t *testing.T) {
	cfg := &RouterConfig{RouterOptions: RouterOptions{AutoModelNames: []string{}}}
	cfg.ModelSelection.ML.ModelsPath = "models/ml"
	cfg.ModelSelection.ML.ModelType = "qwen3"
	cfg.Decisions = []Decision{{Algorithm: &AlgorithmConfig{Type: "svm"}}}
	cfg.Recipes = []RoutingRecipe{
		{Name: DefaultRecipeName, Profile: RoutingProfile{Decisions: cfg.Decisions}},
		{Name: "ml", Profile: RoutingProfile{Decisions: cfg.Decisions}},
		{Name: "static", Profile: RoutingProfile{Decisions: []Decision{{Algorithm: &AlgorithmConfig{Type: "static"}}}}},
	}
	if got := EmbeddingModelsNeeded(cfg, "mmbert", false); len(got) != 0 {
		t.Fatalf("dormant default should not prepare recipe-local ML: %v", got)
	}
	if got := EmbeddingModelsNeeded(cfg.ConfigForRecipe(&cfg.Recipes[1]), "mmbert", false); !maps.Equal(got, map[string]bool{"qwen3": true}) {
		t.Fatalf("ML recipe embedding needs = %v, want qwen3", got)
	}
	if got := EmbeddingModelsNeeded(cfg.ConfigForRecipe(&cfg.Recipes[2]), "mmbert", false); len(got) != 0 {
		t.Fatalf("static recipe prepared another recipe's ML model: %v", got)
	}
}
