package config

import "testing"

func TestGlobalModelServicesDoNotBorrowDefaultRecipeOverride(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.EmbeddingConfig.ModelType = "mmbert"
	cfg.ModelDeployments = map[string]ModelDeployment{
		"global": {Provider: "ort", Device: "rocm:0", Artifact: "models/encoder"},
		"local":  {Provider: "ort", Device: "cpu", Artifact: "models/encoder"},
	}
	cfg.GlobalModelBindings = map[string]ModelBinding{"embedding": {Deployment: "global", Adapter: "mmbert", Contract: "embedding.v1"}}
	cfg.ModelBindings = map[string]ModelBinding{"embedding": {Deployment: "local", Adapter: "mmbert", Contract: "embedding.v1"}}
	cfg.Tools.Enabled = true
	cfg.Memory.Enabled = true
	cfg.EmbeddingRules = []EmbeddingRule{{Name: "route", Candidates: []string{"hello"}}}
	services := cfg.ConfigForGlobalModelServices()
	plan, err := CompileModelBindings(services)
	if err != nil {
		t.Fatal(err)
	}
	global, _ := plan.LookupGlobal("embedding")
	if global.Deployment.Device != "rocm:0" || services.ModelBindings["embedding"].Deployment != "global" {
		t.Fatal("global service borrowed the CPU recipe override")
	}
	if len(services.EmbeddingRules) != 0 || len(services.Decisions) != 0 || !services.Tools.Enabled || !services.Memory.Enabled {
		t.Fatal("service scope mixed routing and shared consumers")
	}
	if cfg.ModelBindings["embedding"].Deployment != "local" || len(cfg.EmbeddingRules) != 1 {
		t.Fatal("service projection mutated recipe authoring")
	}
}

func TestGlobalModelServicesUseOnlyReachableEnabledPlugins(t *testing.T) {
	cfg := &RouterConfig{RouterOptions: RouterOptions{AutoModelNames: []string{}}}
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
