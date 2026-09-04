package config

import (
	"testing"

	yamlv3 "gopkg.in/yaml.v3"
)

const momAsset = "config/recipes/built-in/latest/mom-v1/config.yaml"

// The built-in MoM asset is intentionally a model-free Recipe document. The
// control plane materializes its selected Recipe with user-owned model
// assignments; it is not a standalone Router configuration.
func TestMoMRecipeDocumentContract(t *testing.T) {
	var document struct {
		Version string            `yaml:"version"`
		Recipes []CanonicalRecipe `yaml:"recipes"`
	}
	if err := yamlv3.Unmarshal(mustReadRepoFile(t, momAsset), &document); err != nil {
		t.Fatalf("parse built-in MoM Recipe document: %v", err)
	}
	if document.Version != "v0.3" {
		t.Fatalf("Recipe document version = %q, want v0.3", document.Version)
	}

	wantDecisions := map[string]int{
		"balance":  6,
		"speed":    5,
		"cost":     4,
		"accuracy": 7,
		"vault":    5,
	}
	if len(document.Recipes) != len(wantDecisions) {
		t.Fatalf("Recipe count = %d, want %d", len(document.Recipes), len(wantDecisions))
	}

	for _, recipe := range document.Recipes {
		wantCount, ok := wantDecisions[recipe.Name]
		if !ok {
			t.Fatalf("unexpected built-in Recipe %q", recipe.Name)
		}
		if got := len(recipe.Routing.Decisions); got != wantCount {
			t.Fatalf("Recipe %q decision count = %d, want %d", recipe.Name, got, wantCount)
		}
		assertModelFreeRecipe(t, recipe)
	}
}

func assertModelFreeRecipe(t *testing.T, recipe CanonicalRecipe) {
	t.Helper()
	if len(recipe.Routing.ModelCards) != 0 {
		t.Fatalf("Recipe %q must not embed model cards", recipe.Name)
	}
	for _, decision := range recipe.Routing.Decisions {
		if len(decision.ModelRefs) != 0 {
			t.Fatalf("Recipe %q decision %q must receive models through assignments", recipe.Name, decision.Name)
		}
		if decision.Algorithm == nil || decision.Algorithm.MinimumCandidates < 1 {
			t.Fatalf(
				"Recipe %q decision %q must declare algorithm.minimum_candidates",
				recipe.Name,
				decision.Name,
			)
		}
	}
}
