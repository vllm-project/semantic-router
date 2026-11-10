package dsl

import (
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestModelBindingsRoundTripAcrossRecipes(t *testing.T) {
	source := `ROUTING {
  model_bindings: {jailbreak: {deployment: "shared"}}
}
RECIPE private {
  ROUTING {
    model_bindings: {pii: "private-pii"}
  }
}
ENTRYPOINT {model_names: ["private"] recipe: "private"}
`
	config, errs := Compile(source)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	output, err := DecompileConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	roundTrip, errs := Compile(output)
	if len(errs) > 0 {
		t.Fatalf("decompiled bindings do not parse: %v\n%s", errs, output)
	}
	if !reflect.DeepEqual(config.ModelBindings, roundTrip.ModelBindings) || !reflect.DeepEqual(config.Recipes, roundTrip.Recipes) {
		t.Fatalf("recipe bindings changed during round trip:\n%s", output)
	}
	if _, leaked := roundTrip.ModelBindings["pii"]; leaked {
		t.Fatal("private binding leaked into default routing")
	}
	program := DecompileRoutingToAST(config)
	jsonProgram := ProgramToJSON(program)
	if !reflect.DeepEqual(jsonProgram.ModelBindings, config.ModelBindings) {
		t.Fatal("builder AST lost model bindings")
	}
	yamlBytes, err := EmitRoutingYAMLFromConfig(roundTrip)
	if err != nil || !strings.Contains(string(yamlBytes), "model_bindings:") || !strings.Contains(string(yamlBytes), "jailbreak: shared") {
		t.Fatalf("routing YAML lost binding fields: %v\n%s", err, yamlBytes)
	}
}

func TestModelBindingsRejectInvalidDSLShape(t *testing.T) {
	for _, source := range []string{
		`ROUTING {model_bindings: "not-an-object"}`,
		`ROUTING {model_bindings: {pii: {deploymnt: "typo"}}}`,
		`ROUTING {model_bindings: {pii: {deployment: 12}}}`,
		`ROUTING {model_bindings: {pii: {deployment: "shared", contract: "token_spans.v1"}}}`,
	} {
		if _, errs := Compile(source); len(errs) == 0 {
			t.Errorf("invalid binding accepted: %s", source)
		}
	}
}

// Global bindings belong to global.model_catalog, which the DSL leaves to the
// base config: a round trip must not turn them into recipe content.
func TestGlobalModelBindingsStayGlobalThroughDSLRoundTrip(t *testing.T) {
	base := []byte(`version: v0.5
global:
  model_catalog:
    deployments:
      guard: {provider: model_runtime, artifact: vllm-sr/vela3-fast, device: cpu}
    bindings:
      jailbreak: {deployment: guard}
recipes:
  - name: main
    decisions:
      - name: catch_all
        priority: 1
`)
	original, err := config.ParseYAMLBytes(base)
	if err != nil {
		t.Fatal(err)
	}
	text, err := DecompileConfig(original)
	if err != nil {
		t.Fatal(err)
	}
	compiled, errs := Compile(text)
	if len(errs) > 0 {
		t.Fatalf("%v\n%s", errs, text)
	}
	mergedBytes, err := MergeRoutingIntoBase(compiled, base)
	if err != nil {
		t.Fatal(err)
	}
	merged, err := config.ParseYAMLBytes(mergedBytes)
	if err != nil {
		t.Fatal(err)
	}
	var recipes []config.RecipeName
	for _, recipe := range merged.Recipes {
		recipes = append(recipes, recipe.Name)
	}
	if !reflect.DeepEqual(recipes, []config.RecipeName{"main"}) || len(merged.Entrypoints) != len(original.Entrypoints) {
		t.Fatalf("recipes %v and entrypoints %+v after the round trip:\n%s", recipes, merged.Entrypoints, text)
	}
}
