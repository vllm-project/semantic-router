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
	resolved, err := ParseYAMLBytes(mustReadRepoFile(t, momAsset))
	if err != nil {
		t.Fatal(err)
	}
	learning := resolved.RouterLearning
	if !learning.Enabled || learning.Adaptation.EffectiveEnabled() ||
		!learning.Protection.EffectiveEnabled() || learning.Protection.EffectiveScope() != RouterLearningScopeConversation {
		t.Fatalf("built-in defaults must enable conversation protection only: %+v", learning)
	}

	wantDecisions := map[string][]string{
		"balance":  {"reasoning", "simple", "medium"},
		"speed":    {"tools", "reasoning", "fast"},
		"cost":     {"tools", "reasoning", "economy"},
		"accuracy": {"agent", "review", "reasoning", "simple"},
		"vault":    {"guard", "sensitive", "private"},
	}
	if len(document.Recipes) != len(wantDecisions) {
		t.Fatalf("Recipe count = %d, want %d", len(document.Recipes), len(wantDecisions))
	}

	for _, recipe := range document.Recipes {
		wantNames, ok := wantDecisions[recipe.Name]
		if !ok {
			t.Fatalf("unexpected built-in Recipe %q", recipe.Name)
		}
		if got := len(recipe.Routing.Decisions); got != len(wantNames) {
			t.Fatalf("Recipe %q decision count = %d, want %d", recipe.Name, got, len(wantNames))
		}
		for i, decision := range recipe.Routing.Decisions {
			if decision.Name != wantNames[i] {
				t.Errorf("Recipe %q decision %d = %q, want %q", recipe.Name, i, decision.Name, wantNames[i])
			}
			wantMode := DecisionAdaptationModeApply
			if recipe.Name == "vault" || recipe.Name == "accuracy" && (decision.Name == "agent" || decision.Name == "review") {
				wantMode = DecisionAdaptationModeBypass
			}
			if got := decision.Adaptations.ProtectionMode(); got != wantMode {
				t.Errorf("Recipe %s decision %s protection mode = %s, want %s", recipe.Name, decision.Name, got, wantMode)
			}
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
		if recipe.Name == "vault" && decision.Name == "guard" {
			if decision.GetFastResponseConfig() == nil || decision.Algorithm != nil {
				t.Fatal("Vault guard must respond immediately without selecting a backend")
			}
			continue
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
