package dsl

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const entrypointRulesDSL = `
MODEL model_a {
  loras: ["reasoning-adapter"]
}
MODEL model_b {}

ENTRYPOINT {
  name: "shared"
  aliases: ["router/shared"]
  rules: [
    { name: "default", recipe: "shared",
      assignments: [
        { decision: "choose", models: [
          { model: "model_b", priority: 0, weight: "0.6", reasoning: { enabled: false } },
          { model: "model_a", priority: 1, weight: "0.4", lora: "reasoning-adapter", reasoning: { enabled: true, effort: "high", description: "Think step by step" } },
        ], fallback: { strategy: "priority", on: ["unavailable", "overloaded"] } },
      ]
    },
  ]
}

RECIPE shared {
  ROUTE choose {}
}
`

func TestEntrypointRulesRoundTripAcrossDSLRepresentations(t *testing.T) {
	cfg, errs := Compile(entrypointRulesDSL)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	wantModels := []string{"model_b", "model_a"}
	decisionID := cfg.Recipes[0].Profile.Decisions[0].ID
	assertAssignmentModels(t, cfg.Entrypoints[0].Rules[0].Action.Assignments[decisionID].Models, wantModels)
	effective, ok := cfg.RecipeForRequestModel("router/shared")
	if !ok {
		t.Fatal("compiled entrypoint did not resolve")
	}
	// Only the active priority tier becomes the decision's initial ModelRefs;
	// later tiers remain in the assignment set for Router-owned fallback.
	assertModelRefs(t, effective.Profile.Decisions[0].ModelRefs, []string{"model_b"})

	decompiled, testEntrypointRulesRoundTripAcrossDSLRepresentationsErr := Decompile(cfg)
	if testEntrypointRulesRoundTripAcrossDSLRepresentationsErr != nil {
		t.Fatalf("decompile: %v", testEntrypointRulesRoundTripAcrossDSLRepresentationsErr)
	}
	if strings.Contains(decompiled, "model_bindings") ||
		!strings.Contains(decompiled, `assignments:`) ||
		!strings.Contains(decompiled, `fallback: { strategy: "priority", on: ["unavailable", "overloaded"] }`) ||
		!strings.Contains(decompiled, `description: "Think step by step"`) {
		t.Fatalf("decompiled DSL lost v0.4 rule fields:\n%s", decompiled)
	}
	recompiled, errs := Compile(decompiled)
	if len(errs) > 0 {
		t.Fatalf("recompile errors: %v\n%s", errs, decompiled)
	}
	if !reflect.DeepEqual(cfg.Entrypoints[0].Rules[0].Action, recompiled.Entrypoints[0].Rules[0].Action) {
		t.Fatalf("DSL round-trip changed rule action:\nbefore: %+v\nafter: %+v", cfg.Entrypoints[0].Rules[0].Action, recompiled.Entrypoints[0].Rules[0].Action)
	}

	yamlBytes, testEntrypointRulesRoundTripAcrossDSLRepresentationsErr := EmitRoutingYAMLFromConfig(cfg)
	if testEntrypointRulesRoundTripAcrossDSLRepresentationsErr != nil {
		t.Fatalf("emit routing YAML: %v", testEntrypointRulesRoundTripAcrossDSLRepresentationsErr)
	}
	var document routingYAMLDocument
	if err := yaml.Unmarshal(yamlBytes, &document); err != nil {
		t.Fatalf("unmarshal routing YAML: %v", err)
	}
	if len(document.Document.Decisions) != 1 || document.Document.Decisions[0].Name != "choose" || document.Document.Decisions[0].ID != "" {
		t.Fatalf("emitted Recipe document must keep only human Decision identity: %+v", document.Document.Decisions)
	}
	if len(document.Document.Decisions[0].ModelRefs) != 0 {
		t.Fatalf("emitted Recipe document leaked physical Model selection: %+v", document.Document.Decisions[0].ModelRefs)
	}

	program, parseErrs := Parse(entrypointRulesDSL)
	if len(parseErrs) > 0 {
		t.Fatalf("parse errors: %v", parseErrs)
	}
	jsonProgram := ProgramToJSON(program)
	jsonSet := jsonProgram.Entrypoints[0].Rules[0].Assignments["choose"]
	jsonRefs := jsonSet.Models
	if len(jsonRefs) != 2 || jsonRefs[0].Model != "model_b" || jsonRefs[1].Priority != 1 || jsonRefs[1].Reasoning.Description != "Think step by step" || jsonSet.Fallback == nil || len(jsonSet.Fallback.On) != 2 {
		t.Fatalf("AST JSON lost ordered assignments: %+v", jsonRefs)
	}
	encoded, testEntrypointRulesRoundTripAcrossDSLRepresentationsErr := MarshalProgramJSON(program)
	if testEntrypointRulesRoundTripAcrossDSLRepresentationsErr != nil {
		t.Fatalf("marshal AST JSON: %v", testEntrypointRulesRoundTripAcrossDSLRepresentationsErr)
	}
	if strings.Contains(string(encoded), `"modelBindings"`) || strings.Contains(string(encoded), `"modelId"`) || !strings.Contains(string(encoded), `"assignments"`) {
		t.Fatalf("marshaled AST JSON does not use the v0.4 rule contract: %s", encoded)
	}
}

func TestCompileRejectsRemovedEntrypointIdentitySyntax(t *testing.T) {
	for _, input := range []string{
		`ENTRYPOINT { id: "ep_old", name: "shared", recipe: "shared", assignments: [] }`,
		`ENTRYPOINT { name: "shared", model_names: ["router/shared"], recipe: "shared", assignments: [] }`,
		`ENTRYPOINT { name: "shared", rules: [{ name: "default", action: {} }] }`,
	} {
		if _, errs := Compile(input); len(errs) == 0 || !strings.Contains(errs[0].Error(), `unknown field`) {
			t.Fatalf("removed Entrypoint identity syntax must fail closed, got %v", errs)
		}
	}
}

func assertAssignmentModels(t *testing.T, assignments []config.RoutingModelAssignment, want []string) {
	t.Helper()
	got := make([]string, 0, len(assignments))
	for _, assignment := range assignments {
		got = append(got, assignment.ModelName)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("assignment models = %v, want %v", got, want)
	}
}

func assertModelRefs(t *testing.T, refs []config.ModelRef, want []string) {
	t.Helper()
	got := make([]string, 0, len(refs))
	for _, ref := range refs {
		got = append(got, ref.Model)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("model refs = %v, want %v", got, want)
	}
}
