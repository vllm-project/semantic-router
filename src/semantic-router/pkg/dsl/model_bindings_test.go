package dsl

import (
	"reflect"
	"strings"
	"testing"
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
