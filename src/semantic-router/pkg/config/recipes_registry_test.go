package config

import (
	"testing"
)

func TestCanonicalRecipeSignalsMergeIntoGlobalRegistry(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(recipeTestBaseYAML + recipeTestPrivacyBlockYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}

	registry := make(map[string]bool, len(cfg.Signals.KeywordRules))
	for _, rule := range cfg.Signals.KeywordRules {
		registry[rule.Name] = true
	}
	if !registry["urgent_keywords"] || !registry["pii_keywords"] {
		t.Fatalf("expected the flat signal registry to union all recipes, got %+v", registry)
	}

	canonical := CanonicalConfigFromRouterConfig(cfg)
	if len(canonical.Routing.Signals.Keywords) != 1 || canonical.Routing.Signals.Keywords[0].Name != "urgent_keywords" {
		t.Fatalf("expected the exported top-level routing block to keep only the default profile, got %+v", canonical.Routing.Signals.Keywords)
	}

	if got := len(cfg.AllRoutingDecisions()); got != 2 {
		t.Fatalf("expected AllRoutingDecisions to cover both recipes, got %d", got)
	}
}

func TestUsesSignalTypeInRoutingCoversRecipeDecisions(t *testing.T) {
	contextRecipeYAML := `
recipes:
  - name: privacy
    routing:
      signals:
        context:
          - name: short_context
            max_tokens: 1K
      decisions:
        - name: privacy_route
          rules:
            operator: AND
            conditions:
              - type: context
                name: short_context
          modelRefs:
            - model: model-b
              use_reasoning: false
`
	cfg, err := ParseYAMLBytes([]byte(recipeTestBaseYAML + contextRecipeYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}

	if !cfg.UsesSignalTypeInRouting("context") {
		t.Fatal("expected a signal type used only by a recipe decision to count as used in routing")
	}
	if cfg.UsesSignalTypeInRouting("embedding") {
		t.Fatal("expected an unused signal type to stay unused")
	}
}

func TestEntrypointRecipeDescription(t *testing.T) {
	cfg := &RouterConfig{
		Recipes: []RoutingRecipe{
			{Name: DefaultRecipeName},
			{Name: "privacy", Description: "privacy profile"},
		},
	}

	if got := cfg.EntrypointRecipeDescription("privacy"); got != "privacy profile" {
		t.Fatalf("expected the recipe's own description, got %q", got)
	}
	if got := cfg.EntrypointRecipeDescription(DefaultRecipeName); got != "Entrypoint for the default routing recipe" {
		t.Fatalf("expected the generic fallback description, got %q", got)
	}
}

func TestAllRoutingDecisionsFallsBackToFlatDecisions(t *testing.T) {
	// Configs built without the canonical loader (DSL fragments, hand-built
	// test configs) carry no recipes; the flat decisions are the only profile.
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Decisions: []Decision{{Name: "flat_route"}},
		},
	}

	decisions := cfg.AllRoutingDecisions()
	if len(decisions) != 1 || decisions[0].Name != "flat_route" {
		t.Fatalf("expected the flat decisions to back the routing view, got %+v", decisions)
	}
}
