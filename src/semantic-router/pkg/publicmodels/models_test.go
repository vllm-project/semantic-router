package publicmodels

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestNewOpenAIModelListUsesSourceMetadata(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{
			IncludeConfigModelsInList: true,
		},
		Entrypoints: []config.EntrypointMapping{
			{
				ID: "entrypoint-balanced", Revision: 1, Name: "partner/balanced", ModelNames: []string{"partner/balanced"},
				Rules: []config.EntrypointRule{{
					ID: "rule-balanced", Name: "default",
					Action: config.EntrypointRuleAction{
						RecipeID: "recipe-balanced", RecipeRevision: 1, Recipe: "balanced",
						Assignments: map[string]config.RoutingAssignmentSet{
							"decision-balanced": {Models: []config.RoutingModelAssignment{{
								ModelID: "model-partner-backend", ModelRevision: 1, ModelName: "partner/backend", Weight: "1",
							}}},
						},
					},
				}},
			},
		},
		Recipes: []config.RoutingRecipe{
			{
				ID:          "recipe-balanced",
				Revision:    1,
				Name:        "balanced",
				Description: "Intelligent Router for Mixture-of-Models",
				Profile: config.RoutingProfile{Decisions: []config.Decision{{
					ID: "decision-balanced", Name: "balanced",
				}}},
			},
		},
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"partner/backend": {ResourceID: "model-partner-backend", ResourceRevision: 1},
			},
		},
	}
	if err := cfg.PrepareEntrypointRecipes(); err != nil {
		t.Fatalf("prepare public model fixture: %v", err)
	}

	modelList := NewOpenAIModelList(cfg, 123)
	modelsByID := make(map[string]OpenAIModel, len(modelList.Data))
	for _, model := range modelList.Data {
		modelsByID[model.ID] = model
	}

	assertPublicModel(
		t,
		modelsByID["partner/balanced"],
		routerOwner,
		selectableVirtualRoute("balanced"),
		"Intelligent Router for Mixture-of-Models",
	)
	assertPublicModel(
		t,
		modelsByID["partner/backend"],
		upstreamEndpointOwner,
		passthroughRoute(),
		"",
	)
}

func TestNewOpenAIModelListDoesNotInventDefaultAliases(t *testing.T) {
	modelList := NewOpenAIModelList(nil, 123)
	if len(modelList.Data) != 0 {
		t.Fatalf("nil config produced hidden routing aliases: %+v", modelList.Data)
	}
}

func TestBackendCandidatesRequirePublicFlagOrExplicitFilteringPipeline(t *testing.T) {
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"private/backend": {ResourceID: "model-private-backend", ResourceRevision: 1},
		}},
	}

	publicList := NewOpenAIModelList(cfg, 123)
	if len(publicList.Data) != 0 {
		t.Fatalf("public catalog ignored disabled backend visibility: %+v", publicList.Data)
	}

	candidates := NewOpenAIModelListWithOptions(cfg, 123, ModelListBuildOptions{
		IncludeBackendModelCandidates: true,
	})
	if len(candidates.Data) != 1 || candidates.Data[0].ID != "private/backend" {
		t.Fatalf("authorized filtering candidates = %+v", candidates.Data)
	}
	if cfg.IncludeConfigModelsInList {
		t.Fatal("candidate construction mutated the public visibility policy")
	}
}

func assertPublicModel(
	t *testing.T,
	model OpenAIModel,
	wantOwner string,
	wantRouting RoutingMetadata,
	wantDescription string,
) {
	t.Helper()
	if model.ID == "" {
		t.Fatal("expected model to exist")
	}
	if model.Object != "model" || model.Created != 123 {
		t.Fatalf("model envelope = %+v", model)
	}
	if model.OwnedBy != wantOwner {
		t.Fatalf("%s owned_by = %q, want %q", model.ID, model.OwnedBy, wantOwner)
	}
	if model.Routing != wantRouting {
		t.Fatalf("%s routing metadata = %+v, want %+v", model.ID, model.Routing, wantRouting)
	}
	if model.Description != wantDescription {
		t.Fatalf("%s description = %q, want %q", model.ID, model.Description, wantDescription)
	}
}
