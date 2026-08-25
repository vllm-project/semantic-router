package config

import (
	"strings"
	"testing"
)

func TestNonDefaultRecipeDecisionsRemainReachableOnlyThroughRecipe(t *testing.T) {
	cfg, err := parseRecipeFixtureYAML(t, []byte(recipeTestPrivacyYAML))
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.HasRoutingDecisions() {
		t.Fatal("HasRoutingDecisions did not include compiled Recipes")
	}
	privacy, ok := cfg.RecipeByName("privacy")
	if !ok || privacy.Profile.Decisions[0].Name != "privacy_route" {
		t.Fatalf("privacy Recipe = %+v", privacy)
	}
	if cfg.GetDecisionByName("privacy_route") != nil {
		t.Fatal("default Recipe lookup escaped into another Recipe")
	}
}

func TestSignalDefinitionsWithTheSameNameStayRecipeLocal(t *testing.T) {
	document := strings.ReplaceAll(recipeTestPrivacyYAML, "pii_keywords", "urgent_keywords")
	cfg, err := parseRecipeFixtureYAML(t, []byte(document))
	if err != nil {
		t.Fatalf("same local signal name should be valid across Recipes: %v", err)
	}
	privacy, _ := cfg.RecipeByName("privacy")
	if got := privacy.Profile.Signals.KeywordRules[0].Keywords; len(got) != 1 || got[0] != "ssn" {
		t.Fatalf("privacy signal was overwritten: %+v", got)
	}
	if got := cfg.DefaultRecipe().Profile.Signals.KeywordRules[0].Keywords; len(got) != 1 || got[0] != "urgent" {
		t.Fatalf("default signal was overwritten: %+v", got)
	}
}

func TestSignalReferencesCannotCrossRecipeBoundaries(t *testing.T) {
	document := strings.Replace(recipeTestPrivacyYAML,
		"      signals:\n        keywords:\n          - {name: pii_keywords, operator: OR, keywords: [ssn]}\n",
		"", 1)
	document = strings.Replace(document, "{type: keyword, name: pii_keywords}", "{type: keyword, name: urgent_keywords}", 1)
	_, err := parseRecipeFixtureYAML(t, []byte(document))
	if err == nil || !strings.Contains(err.Error(), `routing recipe "privacy"`) || !strings.Contains(err.Error(), "not declared in this recipe") {
		t.Fatalf("error = %v", err)
	}
}

func TestSignalReferencesCannotCrossSignalFamilies(t *testing.T) {
	document := strings.Replace(recipeTestPrivacyYAML, "pii_keywords", "shared_name", 2)
	document = strings.Replace(document, "{type: keyword, name: shared_name}", "{type: authz, name: shared_name}", 1)
	_, err := parseRecipeFixtureYAML(t, []byte(document))
	if err == nil || !strings.Contains(err.Error(), "not declared in this recipe") {
		t.Fatalf("error = %v", err)
	}
}

func TestDuplicateDecisionNameAcrossRecipesIsIsolated(t *testing.T) {
	document := strings.ReplaceAll(recipeTestPrivacyYAML, "privacy_route", "default_route")
	cfg, err := parseRecipeFixtureYAML(t, []byte(document))
	if err != nil {
		t.Fatalf("same Decision name in separate Recipes was rejected: %v", err)
	}
	privacy, _ := cfg.RecipeByName("privacy")
	if privacy.Profile.Decisions[0].Name != "default_route" || cfg.DefaultRecipe().Profile.Decisions[0].Name != "default_route" {
		t.Fatalf("Decision names were not Recipe-local: %+v", cfg.Recipes)
	}
}

func TestEntrypointNameCollisionsRejected(t *testing.T) {
	document := strings.Replace(
		recipeTestPrivacyYAML,
		"  - model_names: [vllm-sr/privacy]\n",
		"  - model_names: [model-a]\n",
		1,
	)
	_, err := parseRecipeFixtureYAML(t, []byte(document))
	if err == nil || !strings.Contains(err.Error(), "already a configured model") {
		t.Fatalf("error = %v", err)
	}
}

func TestEntrypointNameCollidingWithLoRARejected(t *testing.T) {
	document := strings.Replace(recipeTestPrivacyYAML,
		"    - {name: model-a, description: default tier}\n",
		"    - name: model-a\n      description: default tier\n      loras: [{name: general-expert}]\n", 1)
	document = strings.Replace(
		document,
		"  - model_names: [vllm-sr/privacy]\n",
		"  - model_names: [general-expert]\n",
		1,
	)
	_, err := parseRecipeFixtureYAML(t, []byte(document))
	if err == nil || !strings.Contains(err.Error(), "already a configured LoRA adapter") {
		t.Fatalf("error = %v", err)
	}
}
