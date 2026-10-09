package publicmodels

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestNewOpenAIModelListUsesSourceMetadata(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{
			ListBackendModels: true,
		},
		Entrypoints: []config.EntrypointMapping{
			{ModelNames: []string{"router/custom"}, Recipe: config.DefaultRecipeName},
			{ModelNames: []string{"partner/balanced"}, Recipe: "balanced"},
		},
		Recipes: []config.RoutingRecipe{
			{
				Name:        "balanced",
				Description: "Intelligent Router for Mixture-of-Models",
			},
		},
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"partner/backend": {},
			},
		},
	}

	modelList := NewOpenAIModelList(cfg, 123)
	modelsByID := make(map[string]OpenAIModel, len(modelList.Data))
	for _, model := range modelList.Data {
		modelsByID[model.ID] = model
	}

	assertPublicModel(
		t,
		modelsByID["router/custom"],
		routerOwner,
		selectableVirtualRoute(config.DefaultRecipeName, true),
		"Entrypoint for the default routing recipe",
	)
	assertPublicModel(
		t,
		modelsByID["partner/balanced"],
		routerOwner,
		selectableVirtualRoute("balanced", false),
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

func TestNewOpenAIModelListKeepsDefaultAliasesGeneric(t *testing.T) {
	modelList := NewOpenAIModelList(nil, 123)
	if len(modelList.Data) != 1 {
		t.Fatalf("default model count = %d, want %d", len(modelList.Data), 1)
	}
	for _, model := range modelList.Data {
		assertPublicModel(t, model, routerOwner, selectableVirtualRoute(config.DefaultRecipeName, true), "Entrypoint for the default routing recipe")
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
	wantRouting.API = config.ChatAPI
	wantRouting.Source = model.Routing.Source
	if model.Routing != wantRouting {
		t.Fatalf("%s routing metadata = %+v, want %+v", model.ID, model.Routing, wantRouting)
	}
	if model.Description != wantDescription {
		t.Fatalf("%s description = %q, want %q", model.ID, model.Description, wantDescription)
	}
}

func TestModelDiscoveryExposesEffectiveSourceAndSelectableBackend(t *testing.T) {
	cfg := &config.RouterConfig{RouterOptions: config.RouterOptions{ListBackendModels: true}, BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{"backend": {}}}}
	models := NewOpenAIModelList(cfg, 0).Data
	if len(models) != 2 || models[0].ID != config.DefaultEntrypointModel || models[0].Routing.Source != config.EntrypointBuiltin || !models[0].Routing.DefaultRoute {
		t.Fatalf("effective default metadata=%+v", models)
	}
	if !models[1].Routing.Selectable || models[1].Routing.Resolution != ResolutionPassthrough || models[1].Routing.API != config.ChatAPI {
		t.Fatalf("backend metadata=%+v", models[1])
	}
	cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{"ours"}, Recipe: config.DefaultRecipeName}}
	models = NewOpenAIModelList(cfg, 0).Data
	if models[0].ID != "ours" || models[0].Routing.Source != config.EntrypointExplicit || !models[0].Routing.DefaultRoute {
		t.Fatalf("explicit default metadata=%+v", models[0])
	}
}
