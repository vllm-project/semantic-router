package modelruntime

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBatchedEmbeddingNeedsFollowsReachableRecipeMLSelection(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{EmbeddingModels: config.EmbeddingModels{Qwen3ModelPath: "/models/qwen3"}},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"backend": {ResourceID: "mdl-backend", ResourceRevision: 1},
		}},
		Recipes: []config.RoutingRecipe{{
			ID: "rcp-default", Revision: 1, Name: config.DefaultRecipeName,
			Profile: config.RoutingProfile{Decisions: []config.Decision{{
				ID: "dec-choose", Name: "choose",
				Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmKNN, ML: &config.MLSelectionConfig{
					KNN: &config.MLKNNConfig{K: 5},
				}},
			}}},
		}},
		Entrypoints: []config.EntrypointMapping{{
			ID: "ep-default", Revision: 1, Name: "default", ModelNames: []string{"router/default"},
			Rules: []config.EntrypointRule{{
				ID: "rule-default", Name: "default",
				Action: config.EntrypointRuleAction{
					RecipeID: "rcp-default", RecipeRevision: 1, Recipe: config.DefaultRecipeName,
					Assignments: map[string]config.RoutingAssignmentSet{
						"dec-choose": {Models: []config.RoutingModelAssignment{{
							ModelID: "mdl-backend", ModelRevision: 1, ModelName: "backend", Weight: "1",
						}}},
					},
				},
			}},
		}},
	}
	if err := cfg.PrepareEntrypointRecipes(); err != nil {
		t.Fatalf("PrepareEntrypointRecipes() error = %v", err)
	}

	semanticCache, ml := batchedEmbeddingNeeds(cfg, "/models/qwen3")
	if semanticCache || !ml {
		t.Fatalf("batchedEmbeddingNeeds() = (%v, %v), want (false, true)", semanticCache, ml)
	}

	cfg.Recipes[0].Profile.Decisions[0].Algorithm = &config.AlgorithmConfig{Type: config.DecisionAlgorithmStatic}
	if err := cfg.PrepareEntrypointRecipes(); err != nil {
		t.Fatalf("PrepareEntrypointRecipes() after policy update error = %v", err)
	}
	_, ml = batchedEmbeddingNeeds(cfg, "/models/qwen3")
	if ml {
		t.Fatal("static Recipe unexpectedly required ML batched embeddings")
	}
}
