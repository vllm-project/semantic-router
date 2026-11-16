package publicmodels

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestChatDiscoveryExcludesNativeOnlyAliases(t *testing.T) {
	cfg := &config.RouterConfig{RouterOptions: config.RouterOptions{ListBackendModels: true}, BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
		"chat-model": {}, "decision-model": {APIFormat: config.APIFormatSystemOne},
	}}}
	models := NewOpenAIModelList(cfg, 123)
	found := false
	for _, model := range models.Data {
		if model.ID == "decision-model" {
			t.Fatal("native-only model leaked into Chat discovery")
		}
		found = found || model.ID == "chat-model"
	}
	if !found {
		t.Fatal("Chat model was omitted")
	}
}

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

// With list_backend_models the list names exactly what a direct request is
// served for, sorted: a LoRA adapter one model declares is listed, a name two
// models declare is not, even as a backendless model, and a model without a
// backend is listed only when an external gateway serves it.
func TestBackendModelListingMatchesDirectRequests(t *testing.T) {
	const document = `
version: v0.5
global:
  router:
    list_backend_models: true
providers:
  defaults:
    model: zeta
  models:
    - name: zeta
      loras: [{name: shared}]
      backend_refs: [{provider: vllm, endpoint: "10.0.0.2:8000"}]
    - name: alpha
      loras: [{name: shared}, {name: only-alpha}]
      backend_refs: [{provider: vllm, endpoint: "10.0.0.1:8000"}]
    - name: notes-only
      context_window_size: 32768
    - name: shared
      context_window_size: 32768
recipes:
  - name: default
`
	listed := func(document string) string {
		cfg, err := config.ParseYAMLBytes([]byte(document))
		if err != nil {
			t.Fatal(err)
		}
		var ids []string
		for _, model := range NewOpenAIModelList(cfg, 0).Data {
			ids = append(ids, model.ID)
		}
		return strings.Join(ids, ",")
	}
	if got, want := listed(document), "vllm-sr/auto,alpha,only-alpha,zeta"; got != want {
		t.Fatalf("listing = %s, want %s", got, want)
	}
	// With no listener and no model selection, an external gateway serves a
	// model that has no backend here.
	external := strings.Replace(document, "    list_backend_models: true\n",
		"    list_backend_models: true\n    model_selection: {enabled: false}\nlisteners: []\n", 1)
	if got, want := listed(external), "vllm-sr/auto,alpha,notes-only,only-alpha,zeta"; got != want {
		t.Fatalf("external-gateway listing = %s, want %s", got, want)
	}
}

// An entrypoint another one sends its reviews or memory calls to is listed for
// those callers, under its role, but not offered for chat.
func TestDelegateEntrypointsAreNotOfferedForChat(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.5
providers:
  models:
    - name: m
      backend_refs: [{provider: vllm, endpoint: "10.0.0.1:8000"}]
entrypoints:
  - {model_names: [vllm-sr/auto], recipe: default, default: true, review_model: vllm-sr/review, memory_model: vllm-sr/memory}
  - {model_names: [vllm-sr/review], recipe: default}
  - {model_names: [vllm-sr/memory, memory-alias], recipe: default}
recipes:
  - name: default
`))
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]RoutingMetadata{}
	for _, model := range NewOpenAIModelList(cfg, 0).Data {
		got[model.ID] = model.Routing
	}
	for id, want := range map[string]string{"vllm-sr/auto": "", "vllm-sr/review": "review", "vllm-sr/memory": "memory", "memory-alias": "memory"} {
		if got[id].Delegate != want || got[id].Selectable != (want == "") {
			t.Errorf("%s routing = %+v, want delegate %q", id, got[id], want)
		}
	}
}
