package dsl

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestNamedRecipeMergePreservesAdaptationsWithinScope(t *testing.T) {
	base := []byte(`version: v0.3
routing:
  decisions:
    - name: shared
      priority: 1
      adaptations: {mode: observe}
recipes:
  - name: alpha
    routing:
      decisions:
        - name: shared
          priority: 1
          adaptations: {mode: bypass}
        - name: removed
          priority: 2
          adaptations: {mode: bypass}
  - name: beta
    routing:
      decisions:
        - name: shared
          priority: 1
          adaptations:
            adaptation: {candidate_set: decision}
`)
	original, err := config.ParseYAMLBytes(base)
	if err != nil {
		t.Fatal(err)
	}
	text, err := Decompile(original)
	if err != nil {
		t.Fatal(err)
	}
	compiled, errs := Compile(text)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	alpha, _ := compiled.RecipeByName("alpha")
	for _, decision := range alpha.Profile.Decisions {
		if decision.Name == "shared" {
			alpha.Profile.Decisions = []config.Decision{decision}
			break
		}
	}
	compiled.Recipes = append(compiled.Recipes, config.RoutingRecipe{Name: "gamma", Profile: config.RoutingProfile{Decisions: []config.Decision{{Name: "shared", Priority: 1}}}})
	mergedBytes, err := MergeRoutingIntoBase(compiled, base)
	if err != nil {
		t.Fatal(err)
	}
	merged, err := config.ParseYAMLBytes(mergedBytes)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []config.RecipeName{config.DefaultRecipeName, "alpha", "beta"} {
		want, _ := original.RecipeByName(name)
		got, _ := merged.RecipeByName(name)
		if len(got.Profile.Decisions) != 1 || !reflect.DeepEqual(want.Profile.Decisions[0].Adaptations, got.Profile.Decisions[0].Adaptations) {
			t.Fatalf("adaptations lost or crossed recipe %s", name)
		}
	}
	gamma, _ := merged.RecipeByName("gamma")
	if !reflect.DeepEqual(gamma.Profile.Decisions[0].Adaptations, config.DecisionAdaptationsConfig{}) {
		t.Fatal("new recipe inherited another recipe's policy")
	}
	// An explicit replacement supplied by a non-DSL caller wins over base policy.
	alpha, _ = compiled.RecipeByName("alpha")
	alpha.Profile.Decisions[0].Adaptations = config.DecisionAdaptationsConfig{Mode: config.DecisionAdaptationModeObserve}
	overridden, err := MergeRoutingIntoBase(compiled, base)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := config.ParseYAMLBytes(overridden)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := parsed.RecipeByName("alpha")
	if got.Profile.Decisions[0].Adaptations.Mode != config.DecisionAdaptationModeObserve {
		t.Fatal("base overwrote an explicit replacement")
	}
}
