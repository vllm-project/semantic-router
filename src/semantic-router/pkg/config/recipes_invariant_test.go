package config

import (
	"reflect"
	"testing"
)

// The normalized state carries one intentional redundancy: DefaultDecisions
// mirrors the default recipe's decisions so single-profile read sites keep
// working. These tests pin that redundancy — and the deliberate asymmetry of
// the neighbouring Signals/Projections fields, which hold the GLOBAL registry
// rather than the default profile's rules.

const invariantTwoRecipeYAML = `
version: v0.3
routing:
  modelCards:
    - name: model-a
    - name: model-b
  signals:
    keywords:
      - name: urgent_keywords
        operator: OR
        keywords: ["urgent"]
  decisions:
    - name: default_route
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: urgent_keywords
      modelRefs:
        - model: model-a
recipes:
  - name: privacy
    routing:
      signals:
        keywords:
          - name: pii_keywords
            operator: OR
            keywords: ["ssn"]
      decisions:
        - name: privacy_route
          rules:
            operator: AND
            conditions:
              - type: keyword
                name: pii_keywords
          modelRefs:
            - model: model-b
entrypoints:
  - model_names: ["vllm-sr/privacy"]
    recipe: privacy
providers:
  defaults:
    default_model: model-a
  models:
    - name: model-a
      backend_refs:
        - endpoint: 127.0.0.1:8000
    - name: model-b
      backend_refs:
        - endpoint: 127.0.0.1:8001
`

func TestDefaultDecisionsMirrorTheDefaultRecipe(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(invariantTwoRecipeYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}

	defaultRecipe := cfg.DefaultRecipe()
	if defaultRecipe == nil {
		t.Fatal("expected a default recipe")
	}
	if !reflect.DeepEqual(cfg.DefaultDecisions, defaultRecipe.Decisions) {
		t.Fatalf("DefaultDecisions drifted from the default recipe:\n flat: %+v\n recipe: %+v",
			cfg.DefaultDecisions, defaultRecipe.Decisions)
	}

	// DefaultDecisions is default-only; AllRoutingDecisions spans every recipe.
	if len(cfg.DefaultDecisions) != 1 || cfg.DefaultDecisions[0].Name != "default_route" {
		t.Fatalf("expected only the default recipe's decision in the flat field, got %+v", cfg.DefaultDecisions)
	}
	all := cfg.AllRoutingDecisions()
	if len(all) != 2 {
		t.Fatalf("expected both profiles' decisions from AllRoutingDecisions, got %+v", all)
	}
}

func TestSignalRegistryIsGlobalWhileDecisionsStayScoped(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(invariantTwoRecipeYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}

	// Signals are the union: one classifier must be able to evaluate any
	// recipe's rules.
	names := make(map[string]bool, len(cfg.KeywordRules))
	for _, rule := range cfg.KeywordRules {
		names[rule.Name] = true
	}
	if !names["urgent_keywords"] || !names["pii_keywords"] {
		t.Fatalf("expected the global signal registry to hold both profiles' rules, got %v", names)
	}

	// ...but the default profile's own signals, used for canonical export,
	// must not carry the other recipe's rules.
	profileNames := make(map[string]bool)
	for _, rule := range cfg.RoutingProfileSignals().KeywordRules {
		profileNames[rule.Name] = true
	}
	if profileNames["pii_keywords"] {
		t.Fatal("the privacy recipe's signal leaked into the default routing profile")
	}
}

func TestRecipesOnlyConfigKeepsFlatFieldInSync(t *testing.T) {
	const recipesOnlyYAML = `
version: v0.3
routing:
  modelCards:
    - name: model-a
recipes:
  - name: default
    routing:
      signals:
        keywords:
          - name: urgent_keywords
            operator: OR
            keywords: ["urgent"]
      decisions:
        - name: default_route
          rules:
            operator: AND
            conditions:
              - type: keyword
                name: urgent_keywords
          modelRefs:
            - model: model-a
providers:
  defaults:
    default_model: model-a
  models:
    - name: model-a
      backend_refs:
        - endpoint: 127.0.0.1:8000
`

	cfg, err := ParseYAMLBytes([]byte(recipesOnlyYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	defaultRecipe := cfg.DefaultRecipe()
	if defaultRecipe == nil {
		t.Fatal("expected the explicit default recipe to normalize")
	}
	if !reflect.DeepEqual(cfg.DefaultDecisions, defaultRecipe.Decisions) {
		t.Fatalf("recipes-only layout left the flat field out of sync:\n flat: %+v\n recipe: %+v",
			cfg.DefaultDecisions, defaultRecipe.Decisions)
	}
	if !cfg.HasRoutingDecisions() {
		t.Fatal("expected HasRoutingDecisions to see the recipes-only decisions")
	}
}
