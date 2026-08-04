package config

import (
	"slices"
	"testing"
)

func TestMultiObjectiveRecipeDefinesObjectiveProfiles(t *testing.T) {
	cfg, err := ParseYAMLBytes(mustReadRepoFile(t, "config/recipes/multi-objective/config.yaml"))
	if err != nil {
		t.Fatalf("parse multi-objective recipe: %v", err)
	}

	assertMultiObjectiveMappings(t, cfg)
	assertMultiObjectiveSharedBackendReasoningCapability(t, cfg)
	assertMultiObjectiveRecipes(t, cfg)
}

func assertMultiObjectiveMappings(t *testing.T, cfg *RouterConfig) {
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
		if _, exists := cfg.ModelConfig[modelName]; exists {
			t.Fatalf("entrypoint alias %q must not acquire provider capability metadata", modelName)
		}
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

func assertMultiObjectiveSharedBackendReasoningCapability(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	for _, modelName := range []string{
		"qwen/qwen3.5-rocm",
		"google/gemini-2.5-flash-lite",
		"google/gemini-3.1-pro",
		"openai/gpt5.4",
		"anthropic/claude-opus-4.6",
	} {
		if got := cfg.ModelConfig[modelName].ReasoningFamily; got != "qwen3" {
			t.Fatalf("%s shared-backend reasoning family = %q, want qwen3", modelName, got)
		}
	}
	if family := cfg.ReasoningFamilies["qwen3"]; family.Type != "chat_template_kwargs" {
		t.Fatalf("shared qwen3 backend lost chat-template reasoning capability: %+v", family)
	}
}

func assertMultiObjectiveRecipes(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	assertMultiObjectiveBalancedRecipe(t, cfg)
	assertMultiObjectiveEfficiencyRecipes(t, cfg)
	assertMultiObjectiveAccuracyRecipe(t, cfg)
	assertMultiObjectivePrivacyRecipe(t, cfg)
}

func assertMultiObjectiveBalancedRecipe(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	balanced, _ := cfg.RecipeByName("balanced")
	if len(balanced.Profile.Projections.Scores) == 0 || len(balanced.Profile.Projections.Mappings) == 0 {
		t.Fatal("balanced recipe must derive an effort projection from recipe-local signals")
	}
	recovery := multiObjectiveDecision(t, balanced, "unified_balance_recovery")
	if !recovery.HasPlugin(DecisionPluginSystemPrompt) {
		t.Fatal("balanced recovery route must include corrective response behavior")
	}
	assertMultiObjectiveLanguageCodes(t, balanced, []string{"zh", "es", "fr", "ja", "de"})
}

func assertMultiObjectiveEfficiencyRecipes(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	speed, _ := cfg.RecipeByName("speed-first")
	speedAlgorithm := multiObjectiveDecision(t, speed, "unified_speed_first_route").Algorithm
	if speedAlgorithm == nil || speedAlgorithm.Type != DecisionAlgorithmMultiFactor ||
		speedAlgorithm.MultiFactor == nil || speedAlgorithm.MultiFactor.Weights == nil ||
		speedAlgorithm.MultiFactor.Weights.Latency != 0.85 {
		t.Fatalf("speed-first recipe lost its latency-first selector: %+v", speedAlgorithm)
	}
	heavyAlgorithm := multiObjectiveDecision(t, speed, "unified_speed_heavy_route").Algorithm
	if heavyAlgorithm == nil || heavyAlgorithm.Type != DecisionAlgorithmLatencyAware {
		t.Fatalf("speed-first heavy route must separate TTFT and TPOT: %+v", heavyAlgorithm)
	}

	cost, _ := cfg.RecipeByName("cost-first")
	for _, decision := range cost.Profile.Decisions {
		if refs := decision.ModelRefs; len(refs) != 1 || refs[0].Model != "qwen/qwen3.5-rocm" {
			t.Fatalf("cost-first decision %q must remain on the self-hosted model: %+v", decision.Name, refs)
		}
	}
	if !multiObjectiveDecision(t, cost, "unified_cost_first_route").HasPlugin(DecisionPluginSemanticCache) {
		t.Fatal("cost-first direct route must reuse semantically equivalent answers")
	}
}

func assertMultiObjectiveAccuracyRecipe(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	accuracy, _ := cfg.RecipeByName("accuracy-first")
	accuracyAlgorithm := multiObjectiveDecision(t, accuracy, "unified_accuracy_first_route").Algorithm
	if accuracyAlgorithm == nil || accuracyAlgorithm.MultiFactor == nil ||
		accuracyAlgorithm.MultiFactor.Weights == nil ||
		accuracyAlgorithm.MultiFactor.Weights.Quality != 1.0 {
		t.Fatalf("accuracy-first recipe lost its quality-only selector: %+v", accuracyAlgorithm)
	}
	for decisionName, algorithmType := range map[string]string{
		"unified_frontier_verified_answer": DecisionAlgorithmConfidence,
		"unified_frontier_workflow":        DecisionAlgorithmWorkflows,
		"unified_frontier_fusion":          DecisionAlgorithmFusion,
		"unified_frontier_remom":           DecisionAlgorithmReMoM,
	} {
		algorithm := multiObjectiveDecision(t, accuracy, decisionName).Algorithm
		if algorithm == nil || algorithm.Type != algorithmType {
			t.Fatalf("accuracy-first decision %q algorithm = %+v, want %q", decisionName, algorithm, algorithmType)
		}
	}
	for _, decision := range accuracy.Profile.Decisions {
		if decision.HasPlugin(DecisionPluginHallucination) {
			t.Fatalf(
				"accuracy-first decision %q must not advertise an unavailable hallucination plugin",
				decision.Name,
			)
		}
	}
	assertMultiObjectiveLanguageCodes(t, accuracy, []string{"zh", "es", "fr", "ja", "de"})
}

func assertMultiObjectiveLanguageCodes(t *testing.T, recipe *RoutingRecipe, expected []string) {
	t.Helper()
	if recipe == nil {
		t.Fatal("recipe is unavailable")
	}
	actual := make([]string, 0, len(recipe.Profile.Signals.LanguageRules))
	for _, rule := range recipe.Profile.Signals.LanguageRules {
		actual = append(actual, rule.Name)
	}
	for _, code := range expected {
		if !slices.Contains(actual, code) {
			t.Fatalf("recipe %q language rules = %v, missing ISO code %q", recipe.Name, actual, code)
		}
	}
}

func assertMultiObjectivePrivacyRecipe(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	privacy, _ := cfg.RecipeByName("privacy-first")
	if len(privacy.Profile.Signals.JailbreakRules) != 1 || len(privacy.Profile.Signals.PIIRules) != 1 {
		t.Fatalf("privacy-first recipe must keep jailbreak and PII signals: %+v", privacy.Profile.Signals)
	}
	if len(privacy.Profile.Projections.Scores) == 0 || len(privacy.Profile.Signals.KBRules) != 1 {
		t.Fatal("privacy-first recipe must derive local policy from detector and KB evidence")
	}
	for _, decision := range privacy.Profile.Decisions {
		for _, ref := range decision.ModelRefs {
			if ref.Model != "qwen/qwen3.5-rocm" {
				t.Fatalf("privacy-first decision %q routes to non-local model %q", decision.Name, ref.Model)
			}
		}
	}
}

func multiObjectiveDecision(t *testing.T, recipe *RoutingRecipe, name string) *Decision {
	t.Helper()
	if recipe == nil {
		t.Fatalf("recipe for decision %q is unavailable", name)
	}
	for i := range recipe.Profile.Decisions {
		if recipe.Profile.Decisions[i].Name == name {
			return &recipe.Profile.Decisions[i]
		}
	}
	t.Fatalf("decision %q not found in recipe %q", name, recipe.Name)
	return nil
}
