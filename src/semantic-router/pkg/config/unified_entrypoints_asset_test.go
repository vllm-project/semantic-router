package config

import "testing"

func TestUnifiedEntrypointsRecipeDefinesObjectiveProfiles(t *testing.T) {
	cfg, err := ParseYAMLBytes(mustReadRepoFile(t, "deploy/recipes/unified-entrypoints.yaml"))
	if err != nil {
		t.Fatalf("parse unified entrypoints recipe: %v", err)
	}

	assertUnifiedEntrypointMappings(t, cfg)
	assertUnifiedReasoningFamilies(t, cfg)
	assertUnifiedObjectiveRecipes(t, cfg)
}

func assertUnifiedEntrypointMappings(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	expectedEntrypoints := map[string]string{
		"vllm-sr/mom-balanced-v1": "balanced",
		"vllm-sr/mom-flash-v1":    "speed-first",
		"vllm-sr/mom-economy-v1":  "cost-first",
		"vllm-sr/mom-frontier-v1": "accuracy-first",
		"vllm-sr/mom-private-v1":  "privacy-first",
	}
	if len(cfg.Entrypoints) != len(expectedEntrypoints) {
		t.Fatalf("entrypoint count = %d, want %d", len(cfg.Entrypoints), len(expectedEntrypoints))
	}
	if len(cfg.Recipes) != len(expectedEntrypoints)+1 {
		t.Fatalf("normalized recipe count = %d, want %d named recipes plus internal default", len(cfg.Recipes), len(expectedEntrypoints))
	}
	if defaultRecipe := cfg.DefaultRecipe(); defaultRecipe == nil || len(defaultRecipe.Profile.Decisions) != 0 {
		t.Fatalf("expected decisionless internal default recipe, got %+v", defaultRecipe)
	}
	if cfg.AutoModelNames == nil || len(cfg.EffectiveAutoModelNames()) != 0 {
		t.Fatalf("expected entrypoint-only public catalog, got auto aliases %#v", cfg.EffectiveAutoModelNames())
	}
	for modelName, recipeName := range expectedEntrypoints {
		recipe, ok := cfg.RecipeForRequestModel(modelName)
		if !ok {
			t.Fatalf("entrypoint %q did not resolve", modelName)
		}
		if recipe.Name != RecipeName(recipeName) {
			t.Fatalf("entrypoint %q resolved to %q, want %q", modelName, recipe.Name, recipeName)
		}
		if len(recipe.Profile.Decisions) == 0 {
			t.Fatalf("entrypoint %q resolved to a recipe without decisions", modelName)
		}
	}
}

func assertUnifiedReasoningFamilies(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	if got := cfg.ModelConfig["qwen/qwen3.5-rocm"].ReasoningFamily; got != "qwen3" {
		t.Fatalf("qwen reasoning family = %q, want qwen3", got)
	}
	for _, modelName := range []string{
		"google/gemini-2.5-flash-lite",
		"google/gemini-3.1-pro",
		"openai/gpt5.4",
		"anthropic/claude-opus-4.6",
	} {
		if got := cfg.ModelConfig[modelName].ReasoningFamily; got != "reasoning-effort" {
			t.Fatalf("%s reasoning family = %q, want reasoning-effort", modelName, got)
		}
	}
	if family := cfg.ReasoningFamilies["reasoning-effort"]; family.Type != "reasoning_effort" {
		t.Fatalf("reasoning-effort family lost effort encoding: %+v", family)
	}
}

func assertUnifiedObjectiveRecipes(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	assertUnifiedEfficiencyRecipes(t, cfg)
	assertUnifiedAccuracyRecipe(t, cfg)
	assertUnifiedPrivacyRecipe(t, cfg)
}

func assertUnifiedEfficiencyRecipes(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	speed, _ := cfg.RecipeByName("speed-first")
	speedAlgorithm := speed.Profile.Decisions[0].Algorithm
	if speedAlgorithm == nil || speedAlgorithm.Type != "multi_factor" ||
		speedAlgorithm.MultiFactor == nil || speedAlgorithm.MultiFactor.Weights == nil ||
		speedAlgorithm.MultiFactor.Weights.Latency != 0.85 {
		t.Fatalf("speed-first recipe lost its latency-first selector: %+v", speedAlgorithm)
	}

	cost, _ := cfg.RecipeByName("cost-first")
	if refs := cost.Profile.Decisions[0].ModelRefs; len(refs) != 1 || refs[0].Model != "qwen/qwen3.5-rocm" {
		t.Fatalf("cost-first recipe must remain on the self-hosted model: %+v", refs)
	}
}

func assertUnifiedAccuracyRecipe(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	accuracy, _ := cfg.RecipeByName("accuracy-first")
	accuracyAlgorithm := accuracy.Profile.Decisions[0].Algorithm
	if accuracyAlgorithm == nil || accuracyAlgorithm.MultiFactor == nil ||
		accuracyAlgorithm.MultiFactor.Weights == nil ||
		accuracyAlgorithm.MultiFactor.Weights.Quality != 1.0 {
		t.Fatalf("accuracy-first recipe lost its quality-only selector: %+v", accuracyAlgorithm)
	}
}

func assertUnifiedPrivacyRecipe(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	privacy, _ := cfg.RecipeByName("privacy-first")
	if len(privacy.Profile.Signals.JailbreakRules) != 1 || len(privacy.Profile.Signals.PIIRules) != 1 {
		t.Fatalf("privacy-first recipe must keep jailbreak and PII signals: %+v", privacy.Profile.Signals)
	}
	for _, decision := range privacy.Profile.Decisions {
		for _, ref := range decision.ModelRefs {
			if ref.Model != "qwen/qwen3.5-rocm" {
				t.Fatalf("privacy-first decision %q routes to non-local model %q", decision.Name, ref.Model)
			}
		}
	}
}
