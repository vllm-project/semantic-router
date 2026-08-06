package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestRouterLearningTierCandidatesStayInsideRecipe(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		Recipes: []config.RoutingRecipe{
			{
				Name: "recipe-a",
				Profile: config.RoutingProfile{Decisions: []config.Decision{{
					Name:      "a-tier-one",
					Tier:      1,
					ModelRefs: []config.ModelRef{{Model: "model-a"}, {Model: "model-a2"}},
				}}},
			},
			{
				Name: "recipe-b",
				Profile: config.RoutingProfile{Decisions: []config.Decision{{
					Name:      "b-tier-one",
					Tier:      1,
					ModelRefs: []config.ModelRef{{Model: "model-b"}},
				}}},
			},
		},
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"model-a":  {},
				"model-a2": {},
				"model-b":  {},
			},
		},
	}}
	decision := &router.Config.Recipes[0].Profile.Decisions[0]
	ctx := &RequestContext{VSRSelectedDecision: decision}
	selCtx := &selection.SelectionContext{
		RecipeName:      "recipe-a",
		DecisionName:    decision.Name,
		CandidateModels: decision.ModelRefs,
	}

	candidates := router.learningCandidateModels(
		selCtx,
		ctx,
		config.RouterLearningCandidateSetTier,
	)

	if len(candidates) != 2 ||
		candidates[0].Model != "model-a" ||
		candidates[1].Model != "model-a2" {
		t.Fatalf("tier candidates escaped recipe-a: %#v", candidates)
	}
}
