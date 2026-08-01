package publicmodels

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestNewOpenAIModelListUsesSourceMetadata(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{
			AutoModelNames:            []string{"router/custom"},
			IncludeConfigModelsInList: true,
		},
		Entrypoints: []config.EntrypointMapping{
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

	assertPublicModel(t, modelsByID["router/custom"], routerOwner, RoutingTypeAutoAlias, autoModelDescription)
	assertPublicModel(
		t,
		modelsByID["partner/balanced"],
		routerOwner,
		RoutingTypeEntrypoint,
		"Intelligent Router for Mixture-of-Models",
	)
	assertPublicModel(t, modelsByID["partner/backend"], upstreamEndpointOwner, RoutingTypeBackend, "")
}

func TestNewOpenAIModelListKeepsDefaultAliasesGeneric(t *testing.T) {
	modelList := NewOpenAIModelList(nil, 123)
	if len(modelList.Data) != len(config.DefaultAutoModelNames()) {
		t.Fatalf("default model count = %d, want %d", len(modelList.Data), len(config.DefaultAutoModelNames()))
	}
	for _, model := range modelList.Data {
		assertPublicModel(t, model, routerOwner, RoutingTypeAutoAlias, autoModelDescription)
	}
}

func assertPublicModel(
	t *testing.T,
	model OpenAIModel,
	wantOwner string,
	wantRoutingType RoutingType,
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
	if model.RoutingType != wantRoutingType {
		t.Fatalf("%s routing_type = %q, want %q", model.ID, model.RoutingType, wantRoutingType)
	}
	if model.Description != wantDescription {
		t.Fatalf("%s description = %q, want %q", model.ID, model.Description, wantDescription)
	}
}
