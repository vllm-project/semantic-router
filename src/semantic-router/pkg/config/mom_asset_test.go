package config

import (
	"slices"
	"testing"

	"gopkg.in/yaml.v2"
)

const momAsset = "config/recipes/built-in/latest/mom-v1/config.yaml"

func TestMoMRecipeDistributionContract(t *testing.T) {
	var distribution CanonicalConfig
	if err := yaml.UnmarshalStrict(mustReadRepoFile(t, momAsset), &distribution); err != nil {
		t.Fatalf("parse built-in Recipe distribution: %v", err)
	}
	if distribution.Version != "v0.4" {
		t.Fatalf("distribution version = %q, want v0.4", distribution.Version)
	}
	if len(distribution.Models) != 0 || len(distribution.Entrypoints) != 0 || distribution.Global != nil {
		t.Fatalf("built-in distribution must own Recipes only: %+v", distribution)
	}
	if err := validateAuthoringRecipes(distribution.Recipes); err != nil {
		t.Fatalf("validate built-in Recipes: %v", err)
	}

	wantNames := []string{"accuracy", "balance", "cost", "speed", "vault"}
	gotNames := make([]string, 0, len(distribution.Recipes))
	decisionCount := 0
	for _, recipe := range distribution.Recipes {
		gotNames = append(gotNames, recipe.Name)
		decisionCount += len(recipe.Document.Decisions)
		assertBuiltInOmniDecision(t, recipe)
		assertBuiltInNoSystemPrompt(t, recipe)
	}
	slices.Sort(gotNames)
	if !slices.Equal(gotNames, wantNames) {
		t.Fatalf("Recipe names = %v, want %v", gotNames, wantNames)
	}
	if decisionCount != 26 {
		t.Fatalf("decision count = %d, want 26", decisionCount)
	}

	accuracy := builtInRecipe(t, distribution.Recipes, "accuracy")
	assertBuiltInOrchestrationBudgets(t, accuracy)
}

func assertBuiltInOmniDecision(t *testing.T, recipe AuthoringRecipe) {
	t.Helper()
	decision := builtInDecision(t, recipe, "omni")
	if !momRuleReferences(decision.Rules, SignalTypeConversation, recipe.Name+"_has_images") {
		t.Fatalf("Recipe %q omni decision does not match image content: %+v", recipe.Name, decision.Rules)
	}
	if len(decision.ModelRefs) != 0 {
		t.Fatalf("Recipe %q embeds physical Models: %+v", recipe.Name, decision.ModelRefs)
	}
}

func assertBuiltInNoSystemPrompt(t *testing.T, recipe AuthoringRecipe) {
	t.Helper()
	for _, decision := range recipe.Document.Decisions {
		if decision.HasPlugin(DecisionPluginSystemPrompt) {
			t.Fatalf("built-in decision %q must not inject a system prompt", decision.Name)
		}
	}
}

func assertBuiltInOrchestrationBudgets(t *testing.T, recipe AuthoringRecipe) {
	t.Helper()
	workflow := builtInDecision(t, recipe, "orchestrate")
	if workflow.Algorithm == nil || workflow.Algorithm.Workflows == nil ||
		workflow.Algorithm.Workflows.MaxCompletionTokens != 2048 ||
		workflow.Algorithm.Workflows.Planner.MaxCompletionTokens != 2048 {
		t.Fatalf("accuracy workflow output budgets changed: %+v", workflow.Algorithm)
	}
	fusion := builtInDecision(t, recipe, "experts")
	if fusion.Algorithm == nil || fusion.Algorithm.Fusion == nil ||
		fusion.Algorithm.Fusion.MaxCompletionTokens != 2048 {
		t.Fatalf("accuracy Fusion output budget changed: %+v", fusion.Algorithm)
	}
}

func builtInRecipe(t *testing.T, recipes []AuthoringRecipe, name string) AuthoringRecipe {
	t.Helper()
	for _, recipe := range recipes {
		if recipe.Name == name {
			return recipe
		}
	}
	t.Fatalf("missing built-in Recipe %q", name)
	return AuthoringRecipe{}
}

func builtInDecision(t *testing.T, recipe AuthoringRecipe, name string) Decision {
	t.Helper()
	for _, decision := range recipe.Document.Decisions {
		if decision.Name == name {
			return decision
		}
	}
	t.Fatalf("Recipe %q is missing decision %q", recipe.Name, name)
	return Decision{}
}

func momRuleReferences(rule RuleNode, signalType, signalName string) bool {
	if rule.Type == signalType && rule.Name == signalName {
		return true
	}
	for _, condition := range rule.Conditions {
		if momRuleReferences(condition, signalType, signalName) {
			return true
		}
	}
	return false
}
