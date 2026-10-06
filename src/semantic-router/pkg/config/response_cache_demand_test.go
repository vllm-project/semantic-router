package config

import "testing"

func cacheDemandDecision(mode string, enabled bool) Decision {
	return Decision{Name: "route", Plugins: []DecisionPlugin{{Type: "response_cache", Configuration: MustStructuredPayload(ResponseCachePluginConfig{Enabled: enabled, Mode: mode})}}}
}

func TestResponseCacheDemandUsesReachableConsumersAndPreservesExactOnly(t *testing.T) {
	for _, tc := range []struct {
		name            string
		decisions       []Decision
		store, semantic bool
	}{
		{"legacy global without decisions", nil, true, true},
		{"no plugin", []Decision{{Name: "route"}}, false, false},
		{"disabled plugin", []Decision{cacheDemandDecision("semantic", false)}, false, false},
		{"exact only", []Decision{cacheDemandDecision("exact", true)}, true, false},
		{"semantic", []Decision{cacheDemandDecision("semantic", true)}, true, true},
		{"both", []Decision{cacheDemandDecision("exact_then_semantic", true)}, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &RouterConfig{}
			cfg.SemanticCache.Enabled = true
			cfg.SemanticCache.EmbeddingModel = "mmbert"
			cfg.Decisions = tc.decisions
			store, semantic := cfg.ResponseCacheDemand()
			if store != tc.store || semantic != tc.semantic {
				t.Fatalf("demand = %v/%v", store, semantic)
			}
			if got := EmbeddingModelsNeeded(cfg, "mmbert", true)["mmbert"]; got != tc.semantic {
				t.Fatalf("embedding demand=%v", got)
			}
		})
	}
	cfg := &RouterConfig{RouterOptions: RouterOptions{AutoModelNames: []string{}}}
	cfg.SemanticCache.Enabled = true
	cfg.SemanticCache.EmbeddingModel = "mmbert"
	cfg.Recipes = []RoutingRecipe{{Name: "default"}, {Name: "cached", Profile: RoutingProfile{Decisions: []Decision{cacheDemandDecision("semantic", true)}}}, {Name: "uncached", Profile: RoutingProfile{Decisions: []Decision{{Name: "plain"}}}}}
	cfg.Entrypoints = []EntrypointMapping{{ModelNames: []string{"public"}, Recipe: "uncached"}}
	if store, _ := cfg.ResponseCacheDemand(); store {
		t.Fatal("dormant recipe provisioned shared cache")
	}
	cfg.Entrypoints[0].Recipe = "cached"
	if !EmbeddingModelsNeeded(cfg, "mmbert", true)["mmbert"] {
		t.Fatal("named recipe's shared consumer was lost by default-scope filtering")
	}
	if EmbeddingModelsNeeded(cfg.ConfigForRecipe(&cfg.Recipes[1]), "mmbert", false)["mmbert"] {
		t.Fatal("recipe attempted to own shared cache")
	}
}
