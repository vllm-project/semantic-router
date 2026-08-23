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

func TestRouterLearningTierCandidatesStayInsideEntrypointBindings(t *testing.T) {
	cfg, err := parseExtProcAuthoringConfig(t, `
version: v0.4
models:
  - name: base-a
    card: {}
    connections: [{provider: vllm, endpoint: http://127.0.0.1:8000, model: base-a}]
  - name: base-b
    card: {}
    connections: [{provider: vllm, endpoint: http://127.0.0.1:8001, model: base-b}]
  - name: edge-a
    card: {}
    connections: [{provider: vllm, endpoint: http://127.0.0.1:8002, model: edge-a}]
  - name: edge-b
    card: {}
    connections: [{provider: vllm, endpoint: http://127.0.0.1:8003, model: edge-b}]
  - name: other-a
    card: {}
    connections: [{provider: vllm, endpoint: http://127.0.0.1:8004, model: other-a}]
  - name: other-b
    card: {}
    connections: [{provider: vllm, endpoint: http://127.0.0.1:8005, model: other-b}]
recipes:
  - name: shared
    document:
      decisions:
        - name: simple
          tier: 1
          rules: {}
        - name: peer
          tier: 1
          rules: {}
entrypoints:
  - name: mom/edge
    recipe: shared
    assignments:
      simple: {models: [{model: edge-a}]}
      peer: {models: [{model: edge-b}]}
  - name: mom/other
    recipe: shared
    assignments:
      simple: {models: [{model: other-a}]}
      peer: {models: [{model: other-b}]}
`)
	if err != nil {
		t.Fatalf("parse entrypoint-bound learning fixture: %v", err)
	}
	recipe, ok := cfg.RecipeForRequestModel("mom/edge")
	if !ok {
		t.Fatal("resolve edge entrypoint recipe")
	}
	decision := &recipe.Profile.Decisions[0]
	router := &OpenAIRouter{Config: cfg}
	candidates := router.learningCandidateModels(
		&selection.SelectionContext{
			RecipeName:      recipe.RuntimeScope(),
			DecisionName:    decision.Name,
			CandidateModels: decision.ModelRefs,
		},
		&RequestContext{VSRSelectedDecision: decision},
		config.RouterLearningCandidateSetTier,
	)

	assertModelRefs(t, candidates, []string{"edge-a", "edge-b"})
}
