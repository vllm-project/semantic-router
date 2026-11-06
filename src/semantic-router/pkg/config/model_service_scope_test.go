package config

import "testing"

func TestGlobalModelServicesUseOnlyReachableEnabledPlugins(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.Recipes = []RoutingRecipe{
		{Name: "dormant", Profile: RoutingProfile{Decisions: []Decision{{Plugins: []DecisionPlugin{
			{Type: "memory", Configuration: MustStructuredPayload(MemoryPluginConfig{Enabled: true})},
		}}}}},
		{Name: "active", Profile: RoutingProfile{Decisions: []Decision{{Plugins: []DecisionPlugin{
			{Type: "tool_selection", Configuration: MustStructuredPayload(ToolSelectionPluginConfig{Enabled: true})},
			{Type: "memory", Configuration: MustStructuredPayload(MemoryPluginConfig{Enabled: false})},
		}}}}},
	}
	cfg.Entrypoints = []EntrypointMapping{{Recipe: "active", ModelNames: []string{"auto"}}}
	services := cfg.ConfigForGlobalModelServices()
	if !services.Tools.Enabled || services.Memory.Enabled {
		t.Fatalf("service demand = tools %v / memory %v", services.Tools.Enabled, services.Memory.Enabled)
	}
	if cfg.Tools.Enabled || cfg.Memory.Enabled {
		t.Fatal("preparation demand changed global authoring")
	}
}

func TestRecipeToolConsumerUsesGlobalModuleWithoutGlobalBinding(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.EmbeddingConfig.ModelType = "bert"
	cfg.Decisions = []Decision{{Plugins: []DecisionPlugin{{Type: "tool_selection", Configuration: MustStructuredPayload(ToolSelectionPluginConfig{Enabled: true})}}}}
	if EmbeddingModelsNeeded(cfg, "bert", false)["bert"] {
		t.Fatal("recipe prepared a second service-owned tools model")
	}
	service := cfg.ConfigForGlobalModelServices()
	if !EmbeddingModelsNeeded(service, "bert", true)["bert"] {
		t.Fatal("global scope lost tool consumer")
	}
}
