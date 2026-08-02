package config

import (
	"strings"
	"testing"
)

func TestCanonicalRecipeSignalsRemainIsolated(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(recipeTestBaseYAML + recipeTestPrivacyBlockYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}

	registry := make(map[string]bool, len(cfg.Signals.KeywordRules))
	for _, rule := range cfg.Signals.KeywordRules {
		registry[rule.Name] = true
	}
	if !registry["urgent_keywords"] || registry["pii_keywords"] {
		t.Fatalf("expected the flat registry to contain only default signals, got %+v", registry)
	}
	privacy, _ := cfg.RecipeByName("privacy")
	if privacy == nil || len(privacy.Profile.Signals.KeywordRules) != 1 || privacy.Profile.Signals.KeywordRules[0].Name != "pii_keywords" {
		t.Fatalf("expected the named recipe to own pii_keywords, got %+v", privacy)
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

func TestDefaultRecipeFallsBackToFlatRoutingProfile(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals:   Signals{KeywordRules: []KeywordRule{{Name: "default_signal"}}},
			Decisions: []Decision{{Name: "default_route"}},
			Strategy:  "confidence",
		},
	}

	recipe := cfg.DefaultRecipe()
	if recipe == nil {
		t.Fatal("expected a synthetic default recipe for a flat config")
	}
	if recipe.Name != DefaultRecipeName || recipe.Profile.Strategy != RoutingStrategyConfidence {
		t.Fatalf("unexpected default recipe metadata: %+v", recipe)
	}
	if len(recipe.Profile.Signals.KeywordRules) != 1 || recipe.Profile.Signals.KeywordRules[0].Name != "default_signal" {
		t.Fatalf("expected the flat signals in the default recipe, got %+v", recipe.Profile.Signals)
	}
	if len(recipe.Profile.Decisions) != 1 || recipe.Profile.Decisions[0].Name != "default_route" {
		t.Fatalf("expected the flat decisions in the default recipe, got %+v", recipe.Profile.Decisions)
	}

	resolved, ok := cfg.RecipeForRoutingModel(DefaultVSRAutoModelName)
	if !ok || resolved == nil || resolved.Name != DefaultRecipeName {
		t.Fatalf("expected the auto alias to resolve the synthetic default recipe, got %+v, %v", resolved, ok)
	}
}

func TestConfigForRecipeKeepsOnlyReferencedKnowledgeBases(t *testing.T) {
	cfg := &RouterConfig{
		KnowledgeBases: []KnowledgeBaseConfig{
			{Name: "privacy_kb"},
			{Name: "mmlu_kb"},
		},
	}
	recipe := &RoutingRecipe{
		Name: "privacy",
		Profile: RoutingProfile{
			Signals: Signals{KBRules: []KBSignalRule{{KB: "privacy_kb"}}},
		},
	}

	scoped := cfg.ConfigForRecipe(recipe)
	if len(scoped.KnowledgeBases) != 1 || scoped.KnowledgeBases[0].Name != "privacy_kb" {
		t.Fatalf("expected only the recipe-referenced KB, got %+v", scoped.KnowledgeBases)
	}
	if len(cfg.KnowledgeBases) != 2 {
		t.Fatalf("scoping mutated the shared KB catalog: %+v", cfg.KnowledgeBases)
	}
}

func TestConfigForRecipeKeepsKnowledgeBasesUsedByProjectionMetrics(t *testing.T) {
	cfg := &RouterConfig{
		KnowledgeBases: []KnowledgeBaseConfig{{Name: "quality_kb"}},
	}
	recipe := &RoutingRecipe{
		Name: "quality",
		Profile: RoutingProfile{
			Projections: Projections{Scores: []ProjectionScore{{
				Name: "quality_score",
				Inputs: []ProjectionScoreInput{{
					Type: ProjectionInputKBMetric,
					KB:   "quality_kb",
				}},
			}}},
		},
	}

	scoped := cfg.ConfigForRecipe(recipe)
	if len(scoped.KnowledgeBases) != 1 || scoped.KnowledgeBases[0].Name != "quality_kb" {
		t.Fatalf("expected the projection-referenced KB, got %+v", scoped.KnowledgeBases)
	}
}

func TestRoutingNamespaceKeyIsCollisionSafe(t *testing.T) {
	left := RoutingNamespaceKey("a::b", "c")
	right := RoutingNamespaceKey("a", "b::c")
	if left == right {
		t.Fatalf("distinct recipe/local pairs produced the same key %q", left)
	}
	if got := RoutingDecisionKey("privacy-first", "protected_route"); got != "privacy-first::protected_route" {
		t.Fatalf("simple routing key lost its readable form: %q", got)
	}
	if got := RoutingNamespaceScope("a::b"); got != "a%3A%3Ab" {
		t.Fatalf("routing scope was not escaped for storage: %q", got)
	}
}

func TestRecipeRoutingStrategyIsValidatedLocally(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(recipeTestBaseYAML + `
recipes:
  - name: privacy
    routing:
      strategy: random
      decisions:
        - name: privacy_route
          rules:
            operator: AND
            conditions:
              - type: keyword
                name: urgent_keywords
          modelRefs:
            - model: model-b
              use_reasoning: false
`))
	if err == nil {
		t.Fatal("expected an unsupported recipe strategy to fail validation")
	}
	if got := err.Error(); !strings.Contains(got, `routing recipe "privacy"`) || !strings.Contains(got, `routing.strategy must be "priority" or "confidence"`) {
		t.Fatalf("unexpected validation error: %v", err)
	}
}
