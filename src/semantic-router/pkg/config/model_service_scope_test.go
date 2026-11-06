package config

import "testing"

func TestGlobalModelServicesUseOnlyReachableEnabledPlugins(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.Recipes = []RoutingRecipe{
		{Name: "dormant", Profile: RoutingProfile{Decisions: []Decision{{Plugins: []DecisionPlugin{
			{Type: "memory", Configuration: MustStructuredPayload(MemoryPluginConfig{Enabled: true})},
		}}}}},
		{Name: "active", Profile: RoutingProfile{Decisions: []Decision{{Plugins: []DecisionPlugin{
			{Type: "memory", Configuration: MustStructuredPayload(MemoryPluginConfig{Enabled: false})},
		}}}}},
	}
	cfg.Entrypoints = []EntrypointMapping{{Recipe: "active", ModelNames: []string{"auto"}}}
	if services := cfg.ConfigForGlobalModelServices(); services.Memory.Enabled {
		t.Fatal("neither an unreachable nor a disabled plugin demands the memory service")
	}
	cfg.Entrypoints = []EntrypointMapping{{Recipe: "dormant", ModelNames: []string{"auto"}}}
	if services := cfg.ConfigForGlobalModelServices(); !services.Memory.Enabled {
		t.Fatal("a reachable enabled plugin demands the memory service")
	}
	if cfg.Memory.Enabled {
		t.Fatal("preparation demand changed global authoring")
	}
}
