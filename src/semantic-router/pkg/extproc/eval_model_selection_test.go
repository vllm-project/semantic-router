package extproc

import (
	"context"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

type evalRuntimeScopeSelector struct {
	wantScope config.RecipeName
	model     string
}

func (s evalRuntimeScopeSelector) Select(_ context.Context, input *selection.SelectionContext) (*selection.SelectionResult, error) {
	if input == nil || input.RecipeName != s.wantScope {
		return nil, fmt.Errorf("Eval selector scope = %q, want %q", input.RecipeName, s.wantScope)
	}
	return &selection.SelectionResult{
		SelectedModel: s.model,
		Method:        selection.MethodLatencyAware,
	}, nil
}

func (evalRuntimeScopeSelector) Method() selection.SelectionMethod {
	return selection.MethodLatencyAware
}

func (evalRuntimeScopeSelector) UpdateFeedback(context.Context, *selection.Feedback) error {
	return nil
}

func (evalRuntimeScopeSelector) Tier() selection.AlgorithmTier {
	return selection.TierSupported
}

func (evalRuntimeScopeSelector) ExternalDependencies() []selection.Dependency {
	return nil
}

func TestSelectModelForEvalUsesLiveMultiFactorPolicy(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"lower-quality":  {QualityScore: 0.2},
				"higher-quality": {QualityScore: 0.95},
			},
		},
	}}
	decision := &config.Decision{
		Name: "quality-route",
		ModelRefs: []config.ModelRef{
			{Model: "lower-quality"},
			{Model: "higher-quality"},
		},
		Algorithm: &config.AlgorithmConfig{
			Type: "multi_factor",
			MultiFactor: &config.MultiFactorSelectionConfig{
				Weights: &config.MultiFactorWeightsConfig{Quality: 1},
			},
		},
	}

	result := router.SelectModelForEval(services.EvalModelSelectionInput{
		Decision: decision,
		Query:    "Compare two designs.",
	})
	if result.Status != services.EvalSelectionSelected || result.SelectedModel != "higher-quality" {
		t.Fatalf("Eval selection = %+v", result)
	}
	if result.Method != "multi_factor" {
		t.Fatalf("Eval selection method = %q", result.Method)
	}
}

func TestSelectModelForEvalDoesNotPretendLooperCandidateIsFinal(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	decision := &config.Decision{
		Name:      "fusion-route",
		ModelRefs: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		Algorithm: &config.AlgorithmConfig{Type: "fusion"},
	}

	result := router.SelectModelForEval(services.EvalModelSelectionInput{Decision: decision})
	if result.Status != services.EvalSelectionExecutionRequired || result.SelectedModel != "" {
		t.Fatalf("looper Eval selection = %+v", result)
	}
}

func TestSelectModelForEvalReportsConfiguredLooperFinalModel(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	decision := &config.Decision{
		Name:      "fusion-route",
		ModelRefs: []config.ModelRef{{Model: "panel-a"}, {Model: "panel-b"}},
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model: "judge-model",
			},
		},
	}

	result := router.SelectModelForEval(services.EvalModelSelectionInput{Decision: decision})
	if result.Status != services.EvalSelectionPlannedFinal || result.SelectedModel != "judge-model" {
		t.Fatalf("configured Looper final selection = %+v", result)
	}
}

func TestSelectModelForEvalDoesNotClaimBaseSelectorIsFinalWhenLearningCanChangeIt(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		RouterLearning: config.RouterLearningConfig{Enabled: true},
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"model-a": {QualityScore: 0.9},
				"model-b": {QualityScore: 0.1},
			},
		},
	}}
	decision := &config.Decision{
		Name:      "adaptive-route",
		ModelRefs: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmMultiFactor,
			MultiFactor: &config.MultiFactorSelectionConfig{
				Weights: &config.MultiFactorWeightsConfig{Quality: 1},
			},
		},
	}

	result := router.SelectModelForEval(services.EvalModelSelectionInput{Decision: decision})
	if result.Status != services.EvalSelectionExecutionRequired || result.SelectedModel != "" {
		t.Fatalf("learning-aware Eval selection = %+v, want no fabricated final model", result)
	}
}

func evalRuntimeScopeConfig(recipeName config.RecipeName, recipeID, decisionID string) *config.RouterConfig {
	model := func(id string) config.ModelParams {
		return config.ModelParams{ResourceID: id, ResourceRevision: 1}
	}
	assignment := func(id, name string) config.RoutingModelAssignment {
		return config.RoutingModelAssignment{
			ModelID: id, ModelRevision: 1, ModelName: name, Weight: "1",
		}
	}
	entrypoint := func(id, alias string, models []config.RoutingModelAssignment) config.EntrypointMapping {
		return config.EntrypointMapping{
			ID: id, Revision: 1, Name: alias, ModelNames: []string{"router/" + alias},
			Rules: []config.EntrypointRule{{
				ID: "rule-" + alias, Name: "default",
				Action: config.EntrypointRuleAction{
					RecipeID: recipeID, RecipeRevision: 1, Recipe: recipeName,
					Assignments: map[string]config.RoutingAssignmentSet{
						decisionID: {Models: models},
					},
				},
			}},
		}
	}
	return &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"base-first": model("model-base-first"), "base-second": model("model-base-second"),
			"edge-first": model("model-edge-first"), "edge-second": model("model-edge-second"),
			"local-first": model("model-local-first"), "local-second": model("model-local-second"),
		}},
		Recipes: []config.RoutingRecipe{{
			ID: recipeID, Revision: 1, Name: recipeName,
			Profile: config.RoutingProfile{Decisions: []config.Decision{{
				ID: decisionID, Name: "choose",
				ModelRefs: []config.ModelRef{{Model: "base-first"}, {Model: "base-second"}},
				Algorithm: &config.AlgorithmConfig{Type: string(selection.MethodLatencyAware)},
			}}},
		}},
		Entrypoints: []config.EntrypointMapping{
			entrypoint("entrypoint-edge", "edge", []config.RoutingModelAssignment{
				assignment("model-edge-first", "edge-first"),
				assignment("model-edge-second", "edge-second"),
			}),
			entrypoint("entrypoint-local", "local", []config.RoutingModelAssignment{
				assignment("model-local-first", "local-first"),
				assignment("model-local-second", "local-second"),
			}),
		},
	}
}

func TestSelectModelForEvalUsesEntrypointRuntimeScope(t *testing.T) {
	const recipeName config.RecipeName = "shared"
	const (
		recipeID   = "recipe-shared"
		decisionID = "decision-choose"
	)
	configForTest := evalRuntimeScopeConfig(recipeName, recipeID, decisionID)
	if err := configForTest.PrepareEntrypointRecipes(); err != nil {
		t.Fatalf("PrepareEntrypointRecipes() error = %v", err)
	}

	edge, edgeOK := configForTest.RecipeForRequestModel("router/edge")
	local, localOK := configForTest.RecipeForRequestModel("router/local")
	if !edgeOK || !localOK || edge.RuntimeScope() == local.RuntimeScope() {
		t.Fatalf("entrypoint scopes are unavailable or shared: edge=%q local=%q", edge.RuntimeScope(), local.RuntimeScope())
	}

	registries := map[config.RecipeName]*selection.Registry{}
	for _, entrypoint := range []struct {
		recipe *config.RoutingRecipe
		model  string
	}{{edge, "edge-second"}, {local, "local-second"}} {
		registry := selection.NewRegistry()
		registry.Register(selection.MethodLatencyAware, evalRuntimeScopeSelector{
			wantScope: entrypoint.recipe.RuntimeScope(),
			model:     entrypoint.model,
		})
		registries[entrypoint.recipe.RuntimeScope()] = registry
	}
	router := &OpenAIRouter{Config: configForTest, RecipeModelSelectors: registries}

	for _, entrypoint := range []struct {
		name   string
		recipe *config.RoutingRecipe
		want   string
	}{{"edge", edge, "edge-second"}, {"local", local, "local-second"}} {
		t.Run(entrypoint.name, func(t *testing.T) {
			result := router.SelectModelForEval(services.EvalModelSelectionInput{
				Recipe:       recipeName,
				RuntimeScope: entrypoint.recipe.RuntimeScope(),
				Decision:     &entrypoint.recipe.Profile.Decisions[0],
				Query:        "route this request",
			})
			if result.Status != services.EvalSelectionSelected || result.SelectedModel != entrypoint.want {
				t.Fatalf("Eval selection = %+v, want model %q", result, entrypoint.want)
			}
		})
	}
}
