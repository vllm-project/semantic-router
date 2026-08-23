package config

import (
	"slices"
	"testing"
)

func TestMultiObjectiveRecipeDefinesObjectiveProfiles(t *testing.T) {
	cfg, err := testAuthoringParser(t).ParseYAMLBytes(mustReadRepoFile(t, "config/recipes/multi-objective/config.yaml"))
	if err != nil {
		t.Fatalf("parse multi-objective recipe: %v", err)
	}

	assertMultiObjectiveMappings(t, cfg)
	assertMultiObjectiveModelReasoningCapability(t, cfg)
	assertMultiObjectiveRecipes(t, cfg)
}

func assertMultiObjectiveMappings(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	expectedEntrypoints := map[string]string{
		"vllm-sr/mom-v1-blend": "balanced",
		"vllm-sr/mom-v1-flash": "speed-first",
		"vllm-sr/mom-v1-lite":  "cost-first",
		"vllm-sr/mom-v1-ultra": "accuracy-first",
		"vllm-sr/mom-v1-vault": "privacy-first",
	}
	if len(cfg.Entrypoints) != len(expectedEntrypoints) {
		t.Fatalf("entrypoint count = %d, want %d", len(cfg.Entrypoints), len(expectedEntrypoints))
	}
	if len(cfg.Recipes) != len(expectedEntrypoints) {
		t.Fatalf("Recipe count = %d, want %d explicit Recipes", len(cfg.Recipes), len(expectedEntrypoints))
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

func assertMultiObjectiveModelReasoningCapability(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	for _, modelName := range []string{
		"local/qwen3.5-122b-frontier",
		"local/qwen3.5-9b-economy",
		"local/qwen3.5-9b-economy-replica",
		"local/qwen3.5-9b-private",
		"local/qwen3.6-27b-coder",
		"local/qwen3.6-35b-flash",
	} {
		familyName := cfg.ModelConfig[modelName].ReasoningFamily
		if familyName == "" {
			t.Fatalf("%s reasoning family is empty", modelName)
		}
		if family := cfg.ReasoningFamilies[familyName]; family.Type != "chat_template_kwargs" {
			t.Fatalf("%s reasoning family %q lost chat-template reasoning capability: %+v", modelName, familyName, family)
		}
	}
	for _, modelName := range []string{
		"local/gemma4-26b-balanced",
		"local/deepseek-v4-flash-analyst",
	} {
		if got := cfg.ModelConfig[modelName].ReasoningFamily; got != "" {
			t.Fatalf("%s must not inherit the Qwen reasoning contract, got %q", modelName, got)
		}
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
	balanced := multiObjectiveRecipe(t, cfg, "balanced")
	if len(balanced.Profile.Projections.Scores) == 0 || len(balanced.Profile.Projections.Mappings) == 0 {
		t.Fatal("balanced recipe must derive an effort projection from recipe-local signals")
	}
	recovery := multiObjectiveDecision(t, balanced, "unified_balance_recovery")
	if !recovery.HasPlugin(DecisionPluginSystemPrompt) {
		t.Fatal("balanced recovery route must include corrective response behavior")
	}
	if rules := balanced.Profile.Signals.FactCheckRules; len(rules) != 1 || rules[0].Name != "needs_fact_check" {
		t.Fatalf("balanced fact-check rules must use the runtime classifier label: %+v", rules)
	}
	if rules := balanced.Profile.Signals.UserFeedbackRules; len(rules) != 1 || rules[0].Name != "wrong_answer" {
		t.Fatalf("balanced feedback rules must use the runtime classifier label: %+v", rules)
	}
	if rules := balanced.Profile.Signals.ComplexityRules; len(rules) != 1 || rules[0].Threshold != 0.08 {
		t.Fatalf("balanced complexity threshold must use a calibrated margin: %+v", rules)
	}
	assertMultiObjectiveLanguageCodes(t, balanced, []string{"zh", "es", "fr", "ja", "de"})
}

func assertMultiObjectiveEfficiencyRecipes(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	speed := multiObjectiveRecipe(t, cfg, "speed-first")
	speedAlgorithm := multiObjectiveDecision(t, speed, "unified_speed_first_route").Algorithm
	speedWeights := requireMultiObjectiveMultiFactorWeights(t, speedAlgorithm, "speed-first")
	if speedWeights.Latency != 0.85 {
		t.Fatalf("speed-first recipe lost its latency-first selector: %+v", speedWeights)
	}
	heavyAlgorithm := multiObjectiveDecision(t, speed, "unified_speed_heavy_route").Algorithm
	requireMultiObjectiveAlgorithmType(t, heavyAlgorithm, DecisionAlgorithmLatencyAware, "speed-first heavy route")

	cost := multiObjectiveRecipe(t, cfg, "cost-first")
	for _, decision := range cost.Profile.Decisions {
		models := make([]string, 0, len(decision.ModelRefs))
		for _, ref := range decision.ModelRefs {
			models = append(models, ref.Model)
		}
		slices.Sort(models)
		if !slices.Equal(models, []string{"local/qwen3.5-9b-economy", "local/qwen3.5-9b-economy-replica"}) {
			t.Fatalf("cost-first decision %q must use both self-hosted economy replicas: %+v", decision.Name, decision.ModelRefs)
		}
		requireMultiObjectiveAlgorithmType(
			t,
			decision.Algorithm,
			DecisionAlgorithmMultiFactor,
			"cost-first decision "+decision.Name,
		)
	}
	if !multiObjectiveDecision(t, cost, "unified_cost_first_route").HasPlugin(DecisionPluginResponseCache) {
		t.Fatal("cost-first direct route must reuse semantically equivalent answers")
	}
}

func assertMultiObjectiveAccuracyRecipe(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	accuracy := multiObjectiveRecipe(t, cfg, "accuracy-first")
	if rules := accuracy.Profile.Signals.FactCheckRules; len(rules) != 0 {
		t.Fatalf("accuracy recipe must not let learned fact-check classification drive orchestration: %+v", rules)
	}
	if rules := accuracy.Profile.Signals.ComplexityRules; len(rules) != 1 || rules[0].Threshold != 0.15 {
		t.Fatalf("accuracy complexity threshold must use a calibrated margin: %+v", rules)
	}
	accuracyAlgorithm := multiObjectiveDecision(t, accuracy, "unified_frontier_direct").Algorithm
	accuracyWeights := requireMultiObjectiveMultiFactorWeights(t, accuracyAlgorithm, "accuracy-first")
	if accuracyWeights.Quality != 1.0 {
		t.Fatalf("accuracy-first recipe lost its quality-only selector: %+v", accuracyWeights)
	}
	assertMultiObjectiveVerifiedDecision(t, accuracy)
	for decisionName, algorithmType := range map[string]string{
		"unified_frontier_verified_answer": DecisionAlgorithmConfidence,
		"unified_frontier_workflow":        DecisionAlgorithmWorkflows,
		"unified_frontier_fusion":          DecisionAlgorithmFusion,
		"unified_frontier_remom":           DecisionAlgorithmReMoM,
	} {
		algorithm := multiObjectiveDecision(t, accuracy, decisionName).Algorithm
		requireMultiObjectiveAlgorithmType(t, algorithm, algorithmType, "accuracy-first decision "+decisionName)
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

func assertMultiObjectiveVerifiedDecision(t *testing.T, accuracy *RoutingRecipe) {
	t.Helper()
	verified := multiObjectiveDecision(t, accuracy, "unified_frontier_verified_answer")
	if len(verified.Rules.Conditions) != 1 ||
		verified.Rules.Conditions[0].Type != SignalTypeKeyword ||
		verified.Rules.Conditions[0].Name != "unified_frontier_verification_markers" {
		t.Fatalf("frontier verification must require explicit user intent: %+v", verified.Rules)
	}
	if verified.Algorithm == nil || verified.Algorithm.Confidence == nil ||
		verified.Algorithm.Confidence.ConfidenceMethod != "avg_logprob" {
		t.Fatalf("frontier verification must use supported streaming confidence: %+v", verified.Algorithm)
	}
	verifiedModels := make([]string, 0, len(verified.ModelRefs))
	for _, ref := range verified.ModelRefs {
		verifiedModels = append(verifiedModels, ref.Model)
	}
	if len(verifiedModels) != 4 || slices.Contains(verifiedModels, "local/gemma4-26b-balanced") {
		t.Fatalf("frontier verification contains a non-tool-compatible candidate: %v", verifiedModels)
	}
}

func requireMultiObjectiveAlgorithmType(t *testing.T, algorithm *AlgorithmConfig, expected, context string) {
	t.Helper()
	if algorithm == nil {
		t.Fatalf("%s algorithm is unavailable", context)
	}
	if algorithm.Type != expected {
		t.Fatalf("%s algorithm = %+v, want %q", context, algorithm, expected)
	}
}

func requireMultiObjectiveMultiFactorWeights(
	t *testing.T,
	algorithm *AlgorithmConfig,
	context string,
) *MultiFactorWeightsConfig {
	t.Helper()
	requireMultiObjectiveAlgorithmType(t, algorithm, DecisionAlgorithmMultiFactor, context)
	if algorithm.MultiFactor == nil {
		t.Fatalf("%s multi-factor configuration is unavailable", context)
	}
	if algorithm.MultiFactor.Weights == nil {
		t.Fatalf("%s multi-factor weights are unavailable", context)
	}
	return algorithm.MultiFactor.Weights
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
	privacy := multiObjectiveRecipe(t, cfg, "privacy-first")
	if len(privacy.Profile.Signals.JailbreakRules) != 1 || len(privacy.Profile.Signals.PIIRules) != 1 {
		t.Fatalf("privacy-first recipe must keep jailbreak and PII signals: %+v", privacy.Profile.Signals)
	}
	if len(privacy.Profile.Projections.Scores) == 0 || len(privacy.Profile.Signals.KBRules) != 1 {
		t.Fatal("privacy-first recipe must derive local policy from detector and KB evidence")
	}
	for _, decision := range privacy.Profile.Decisions {
		for _, ref := range decision.ModelRefs {
			if ref.Model != "local/qwen3.5-9b-private" {
				t.Fatalf("privacy-first decision %q routes to non-local model %q", decision.Name, ref.Model)
			}
		}
	}
}

func multiObjectiveRecipe(t *testing.T, cfg *RouterConfig, name RecipeName) *RoutingRecipe {
	t.Helper()
	for _, entrypoint := range cfg.Entrypoints {
		if entrypoint.Recipe != name || len(entrypoint.ModelNames) == 0 {
			continue
		}
		if recipe, ok := cfg.RecipeForRequestModel(entrypoint.ModelNames[0]); ok {
			return recipe
		}
	}
	t.Fatalf("no Entrypoint materializes recipe %q", name)
	return nil
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
