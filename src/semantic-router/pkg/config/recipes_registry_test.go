package config

import (
	"strings"
	"testing"
)

func TestCanonicalRecipeSignalsRemainIsolated(t *testing.T) {
	cfg, err := parseRecipeFixtureYAML(t, []byte(recipeTestPrivacyYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}

	defaultRecipe := cfg.DefaultRecipe()
	if defaultRecipe == nil {
		t.Fatal("default Recipe was not compiled")
	}
	registry := make(map[string]bool, len(defaultRecipe.Profile.Signals.KeywordRules))
	for _, rule := range defaultRecipe.Profile.Signals.KeywordRules {
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
	var exportedPrivacy *CanonicalRecipe
	for index := range canonical.Recipes {
		if canonical.Recipes[index].Name == "privacy" {
			exportedPrivacy = &canonical.Recipes[index]
		}
	}
	if len(canonical.Recipes) != 2 || exportedPrivacy == nil || len(exportedPrivacy.Routing.Signals.Keywords) != 1 || exportedPrivacy.Routing.Signals.Keywords[0].Name != "pii_keywords" {
		t.Fatalf("expected the exported named Recipe to preserve only its local signals, got %+v", canonical.Recipes)
	}

	if got := len(cfg.AllRoutingDecisions()); got != 2 {
		t.Fatalf("expected AllRoutingDecisions to cover both recipes, got %d", got)
	}
}

func TestUsesSignalTypeInRoutingCoversRecipeDecisions(t *testing.T) {
	contextRecipeYAML := strings.Replace(recipeTestPrivacyYAML,
		"      signals:\n        keywords:\n          - {name: pii_keywords, operator: OR, keywords: [ssn]}\n",
		"      signals:\n        context:\n          - {name: short_context, max_tokens: 1K}\n", 1)
	contextRecipeYAML = strings.Replace(contextRecipeYAML,
		"{type: keyword, name: pii_keywords}", "{type: context, name: short_context}", 1)
	cfg, err := parseRecipeFixtureYAML(t, []byte(contextRecipeYAML))
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

func TestAllRoutingDecisionsUsesFlatDecisionsOnlyForScopedView(t *testing.T) {
	cfg := &RouterConfig{
		RoutingScope: "scoped",
		IntelligentRouting: IntelligentRouting{
			Decisions: []Decision{{Name: "flat_route"}},
		},
	}

	decisions := cfg.AllRoutingDecisions()
	if len(decisions) != 1 || decisions[0].Name != "flat_route" {
		t.Fatalf("expected the scoped decisions to back the routing view, got %+v", decisions)
	}
	cfg.RoutingScope = ""
	if decisions := cfg.AllRoutingDecisions(); len(decisions) != 0 {
		t.Fatalf("root runtime config accepted scoped flat decisions: %+v", decisions)
	}
}

func TestDefaultRecipeNeverSynthesizesFlatRoutingProfile(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals:   Signals{KeywordRules: []KeywordRule{{Name: "default_signal"}}},
			Decisions: []Decision{{Name: "default_route"}},
			Strategy:  "confidence",
		},
	}

	if recipe := cfg.DefaultRecipe(); recipe != nil {
		t.Fatalf("root runtime config synthesized a Recipe from scoped fields: %+v", recipe)
	}

	if resolved, ok := cfg.RecipeForRequestModel("vllm-sr/auto"); ok || resolved != nil {
		t.Fatalf("unmapped model resolved without an explicit Entrypoint: %+v, %v", resolved, ok)
	}
}

func TestRoutingProfileVisitorNeverTreatsRootFieldsAsARecipe(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Decisions: []Decision{{Name: "flat_route"}},
		},
	}
	visits := 0
	if err := visitRoutingProfileConfigs(cfg, func(*RouterConfig) error {
		visits++
		return nil
	}); err != nil {
		t.Fatalf("visitRoutingProfileConfigs() error = %v", err)
	}
	if visits != 0 {
		t.Fatalf("root scoped fields were visited as an implicit Recipe %d times", visits)
	}

	cfg.RoutingScope = "authoring"
	if err := visitRoutingProfileConfigs(cfg, func(*RouterConfig) error {
		visits++
		return nil
	}); err != nil {
		t.Fatalf("visitRoutingProfileConfigs(scoped) error = %v", err)
	}
	if visits != 1 {
		t.Fatalf("explicit scoped view visits = %d, want 1", visits)
	}
}

func TestReachableRoutingRecipesIncludesOnlyEntrypointRecipes(t *testing.T) {
	cfg := &RouterConfig{
		Recipes: []RoutingRecipe{
			{Name: DefaultRecipeName},
			{Name: "mapped"},
			{Name: "unmapped"},
		},
	}
	cfg.Entrypoints = []EntrypointMapping{testCompiledEntrypoint("vllm-sr/mapped", &cfg.Recipes[1])}

	reachable := cfg.ReachableRoutingRecipes()
	if len(reachable) != 1 || reachable[0].Name != "mapped" {
		t.Fatalf("reachable recipes = %+v, want only mapped", reachable)
	}
	if cfg.IsRecipeReachableForRouting(DefaultRecipeName) {
		t.Fatal("default Recipe became reachable without an Entrypoint")
	}
	if cfg.IsRecipeReachableForRouting("unmapped") {
		t.Fatal("unmapped named recipe unexpectedly reported reachable")
	}
}

func TestReachableRoutingRecipesRequiresAnEntrypoint(t *testing.T) {
	cfg := &RouterConfig{
		Recipes: []RoutingRecipe{{Name: DefaultRecipeName}},
	}
	if got := cfg.ReachableRoutingRecipes(); len(got) != 0 {
		t.Fatalf("reachable recipes = %+v, a declared Recipe must not be callable without an Entrypoint", got)
	}

	cfg.Entrypoints = []EntrypointMapping{testCompiledEntrypoint("vllm-sr/default", &cfg.Recipes[0])}
	if got := cfg.ReachableRoutingRecipes(); len(got) != 1 || got[0].Name != DefaultRecipeName {
		t.Fatalf("default entrypoint did not restore reachability: %+v", got)
	}
}

func testCompiledEntrypoint(alias string, recipe *RoutingRecipe) EntrypointMapping {
	rule := EntrypointRule{
		ID: "rule-test", Name: "default",
		Action:        EntrypointRuleAction{RecipeID: recipe.ID, RecipeRevision: recipe.Revision, Recipe: recipe.Name},
		derivedRecipe: recipe,
	}
	return EntrypointMapping{
		ID: "ep-test", Revision: 1, Name: alias, ModelNames: []string{alias},
		Rules: []EntrypointRule{rule}, Recipe: recipe.Name, derivedRecipe: recipe,
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
	document := strings.Replace(recipeTestPrivacyYAML,
		"    routing:\n      signals:\n        keywords:\n          - {name: pii_keywords",
		"    routing:\n      strategy: random\n      signals:\n        keywords:\n          - {name: pii_keywords", 1)
	_, err := parseRecipeFixtureYAML(t, []byte(document))
	if err == nil {
		t.Fatal("expected an unsupported recipe strategy to fail validation")
	}
	if got := err.Error(); !strings.Contains(got, `routing recipe "privacy"`) || !strings.Contains(got, `routing.strategy must be "priority" or "confidence"`) {
		t.Fatalf("unexpected validation error: %v", err)
	}
}
