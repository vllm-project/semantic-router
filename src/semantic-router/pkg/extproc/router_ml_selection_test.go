package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestBuildMLSelectionConfigUsesOnlyScopedRecipe(t *testing.T) {
	cfg := &config.RouterConfig{Recipes: []config.RoutingRecipe{
		{
			Name: "nearest",
			Profile: config.RoutingProfile{Decisions: []config.Decision{{
				Name: "choose",
				Algorithm: &config.AlgorithmConfig{
					Type: config.DecisionAlgorithmKNN,
					ML: &config.MLSelectionConfig{
						EmbeddingDim: 384,
						KNN:          &config.MLKNNConfig{K: 3},
					},
				},
			}}},
		},
		{
			Name: "boundary",
			Profile: config.RoutingProfile{Decisions: []config.Decision{{
				Name: "choose",
				Algorithm: &config.AlgorithmConfig{
					Type: config.DecisionAlgorithmSVM,
					ML: &config.MLSelectionConfig{
						EmbeddingDim: 1024,
						SVM:          &config.MLSVMConfig{Kernel: "linear"},
					},
				},
			}}},
		},
	}}

	nearest := buildMLSelectionConfig(cfg.ConfigForRecipe(&cfg.Recipes[0]))
	boundary := buildMLSelectionConfig(cfg.ConfigForRecipe(&cfg.Recipes[1]))
	if nearest == nil || nearest.KNN == nil || nearest.KNN.K != 3 || nearest.SVM != nil || nearest.EmbeddingDim != 384 {
		t.Fatalf("nearest Recipe selector config = %#v", nearest)
	}
	if boundary == nil || boundary.SVM == nil || boundary.SVM.Kernel != "linear" || boundary.KNN != nil || boundary.EmbeddingDim != 1024 {
		t.Fatalf("boundary Recipe selector config = %#v", boundary)
	}
}

func TestRecipeModelSelectorRegistriesContainOnlyConfiguredMLFamilies(t *testing.T) {
	cfg := &config.RouterConfig{Recipes: []config.RoutingRecipe{
		{
			Name: "nearest",
			Profile: config.RoutingProfile{Decisions: []config.Decision{{
				Name: "choose",
				Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmKNN, ML: &config.MLSelectionConfig{
					KNN: &config.MLKNNConfig{K: 3},
				}},
			}}},
		},
		{
			Name: "boundary",
			Profile: config.RoutingProfile{Decisions: []config.Decision{{
				Name: "choose",
				Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmSVM, ML: &config.MLSelectionConfig{
					SVM: &config.MLSVMConfig{Kernel: "linear"},
				}},
			}}},
		},
	}}

	registries, _, _, cancel := createModelSelectorRegistries(cfg, nil)
	if cancel != nil {
		defer cancel()
	}
	nearest := registries["nearest"]
	boundary := registries["boundary"]
	if nearest == nil || boundary == nil {
		t.Fatalf("Recipe registries = %#v", registries)
	}
	if _, ok := nearest.Get(selection.MethodKNN); !ok {
		t.Fatal("nearest Recipe did not register KNN")
	}
	if _, ok := nearest.Get(selection.MethodSVM); ok {
		t.Fatal("nearest Recipe unexpectedly registered SVM")
	}
	if _, ok := boundary.Get(selection.MethodSVM); !ok {
		t.Fatal("boundary Recipe did not register SVM")
	}
	if _, ok := boundary.Get(selection.MethodKNN); ok {
		t.Fatal("boundary Recipe unexpectedly registered KNN")
	}
}
