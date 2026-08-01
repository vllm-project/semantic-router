package config

import (
	"reflect"
	"testing"
)

// The normalized state carries one intentional redundancy: DefaultDecisions
// mirrors the default recipe's decisions so single-profile read sites keep
// working. These tests pin that redundancy as an alias, not merely as equal
// content: both views must share one backing array, so a mutation seen
// through one is seen through the other.

func assertDefaultDecisionsAliasDefaultRecipe(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	defaultRecipe := cfg.DefaultRecipe()
	if defaultRecipe == nil {
		t.Fatal("expected a default recipe")
	}
	if !reflect.DeepEqual(cfg.DefaultDecisions, defaultRecipe.Decisions) {
		t.Fatalf("DefaultDecisions drifted from the default recipe:\n flat: %+v\n recipe: %+v",
			cfg.DefaultDecisions, defaultRecipe.Decisions)
	}
	// DeepEqual alone would also pass on an equal copy; the invariant is
	// stronger. A defensive copy in the loader would break the alias while
	// keeping the content equal.
	if len(cfg.DefaultDecisions) > 0 && &cfg.DefaultDecisions[0] != &defaultRecipe.Decisions[0] {
		t.Fatal("DefaultDecisions and the default recipe hold equal copies, not one aliased slice")
	}
}

func TestDefaultDecisionsMirrorTheDefaultRecipe(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(recipeTestBaseYAML + recipeTestPrivacyBlockYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}

	assertDefaultDecisionsAliasDefaultRecipe(t, cfg)

	// DefaultDecisions is default-only; AllRoutingDecisions spans every recipe.
	if len(cfg.DefaultDecisions) != 1 || cfg.DefaultDecisions[0].Name != "default_route" {
		t.Fatalf("expected only the default recipe's decision in the flat field, got %+v", cfg.DefaultDecisions)
	}
	if got := len(cfg.AllRoutingDecisions()); got != 2 {
		t.Fatalf("expected both profiles' decisions from AllRoutingDecisions, got %d", got)
	}
}

func TestRecipesOnlyConfigKeepsFlatFieldInSync(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(recipeTestRecipesOnlyYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}

	assertDefaultDecisionsAliasDefaultRecipe(t, cfg)

	if !cfg.HasRoutingDecisions() {
		t.Fatal("expected HasRoutingDecisions to see the recipes-only decisions")
	}
}
