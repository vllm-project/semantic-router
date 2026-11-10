package dsl

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDSLMergePreservesDecisionFieldsTheDSLCannotExpress(t *testing.T) {
	base := []byte(`version: v0.5
recipes:
  - name: default
    decisions:
      - name: shared
        priority: 1
        reliability: {total_timeout: 90s, retry_on: reset}
        fallback: {enabled: true, max_attempts: 2}
        output_contract: Answer in one sentence.
        annotations: {policy_class: business, owner: routing-platform}
  - name: alpha
    decisions:
      - name: shared
        priority: 1
        reliability: {per_try_timeout: 5s}
        fallback: {enabled: false}
        output_contract_spec: {type: choice, choice_set: {values: ["yes", "no"]}}
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
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
	mergedBytes, err := MergeRoutingIntoBase(compiled, base)
	if err != nil {
		t.Fatal(err)
	}
	merged, err := config.ParseYAMLBytes(mergedBytes)
	if err != nil {
		t.Fatal(err)
	}
	baseOnly := func(d config.Decision) []interface{} {
		return []interface{}{d.Adaptations, d.Reliability, d.Fallback, d.OutputContract, d.OutputContractSpec, d.Annotations}
	}
	for _, name := range []config.RecipeName{config.DefaultRecipeName, "alpha"} {
		want, _ := original.RecipeByName(name)
		got, _ := merged.RecipeByName(name)
		if len(got.Profile.Decisions) != 1 || !reflect.DeepEqual(baseOnly(want.Profile.Decisions[0]), baseOnly(got.Profile.Decisions[0])) {
			t.Fatalf("recipe %s: a base-only decision field was lost or crossed in the DSL merge:\n%s", name, mergedBytes)
		}
	}
}

func TestNamedRecipeMergePreservesAdaptationsWithinScope(t *testing.T) {
	base := []byte(`version: v0.5
recipes:
  - name: default
    decisions:
      - name: shared
        priority: 1
        adaptations: {mode: observe}
  - name: alpha
    decisions:
      - name: shared
        priority: 1
        adaptations: {mode: bypass}
      - name: removed
        priority: 2
        adaptations: {mode: bypass}
  - name: beta
    decisions:
      - name: shared
        priority: 1
        adaptations:
          adaptation: {candidate_set: decision}
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
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
