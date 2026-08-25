package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

const strictV03AuthoringYAML = `
version: v0.3
providers:
  defaults:
    reasoning_families:
      model-c: {type: reasoning_effort, parameter: reasoning_effort}
  models:
    - name: model-a
      provider_model_id: model-a
      backend_refs:
        - {provider: private-test, endpoint: http://model-a.example}
    - name: model-b
      provider_model_id: model-b
      backend_refs:
        - {provider: private-test, endpoint: http://model-b.example}
    - name: model-c
      provider_model_id: model-c
      reasoning_family: model-c
      backend_refs:
        - {provider: private-test, endpoint: http://model-c.example}
routing:
  modelCards:
    - {name: model-a}
    - {name: model-b}
    - name: model-c
      reasoning: {type: reasoning_effort, efforts: [high]}
recipes:
  - name: orchestration
    routing:
      decisions:
        - name: choose
          rules: {}
        - name: finish
          rules: {}
entrypoints:
  - model_names: [vllm-sr/edge, vllm-sr/edge-alias]
    recipe: orchestration
    assignments:
      choose:
        models:
          - model: model-c
            weight: "2"
            reasoning: {enabled: true, effort: high}
      finish: {models: [{model: model-a}]}
global:
  services:
    backend_egress: {policy_file: /app/config/backend-egress-policy.yaml}
  integrations:
    looper: {endpoint: "http://localhost:8899/v1/chat/completions"}
`

func TestEntrypointAssignmentsCompileStablePinnedView(t *testing.T) {
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(strictV03AuthoringYAML))
	if err != nil {
		t.Fatalf("parse v0.3 entrypoint: %v", err)
	}
	if len(cfg.Entrypoints) != 1 || len(cfg.Entrypoints[0].Rules) != 1 {
		t.Fatalf("unexpected normalized entrypoint: %+v", cfg.Entrypoints)
	}
	recipe, ok := cfg.RecipeForRequestModel("vllm-sr/edge")
	if !ok {
		t.Fatal("request-facing model did not resolve its Recipe")
	}
	assertDecisionModels(t, recipe.Profile.Decisions[0], "model-c")
	assertDecisionModels(t, recipe.Profile.Decisions[1], "model-a")

	ref := recipe.Profile.Decisions[0].ModelRefs[0]
	if ref.UseReasoning == nil || !*ref.UseReasoning || ref.ReasoningEffort != "high" || ref.Weight != 2 {
		t.Fatalf("assignment controls were not compiled: %+v", ref)
	}
	model := cfg.ModelConfig["model-c"]
	if model.Reasoning.Type != ReasoningFamilyTypeReasoningEffort ||
		!reflect.DeepEqual(model.Reasoning.Efforts, []string{"high"}) {
		t.Fatalf("Model reasoning support was not compiled: %+v", model.Reasoning)
	}
	rule := &cfg.Entrypoints[0].Rules[0]
	decisionID := recipe.Profile.Decisions[0].ID
	assignment := rule.Action.Assignments[decisionID].Models[0]
	if assignment.ModelRevision != initialRoutingResourceRevision ||
		rule.Action.RecipeRevision != initialRoutingResourceRevision {
		t.Fatalf("action did not pin exact revisions: %+v", rule.Action)
	}
}

func TestEntrypointAssignmentsCanonicalRoundTrip(t *testing.T) {
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(strictV03AuthoringYAML))
	if err != nil {
		t.Fatal(err)
	}
	exported, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	for _, machineField := range []string{"recipe_id:", "provider_catalog_revision:", "backends:"} {
		if strings.Contains(string(exported), machineField) {
			t.Fatalf("export leaked compiler-owned field %q:\n%s", machineField, exported)
		}
	}
	reparsed, err := testAuthoringParser(t).ParseYAMLBytes(exported)
	if err != nil {
		t.Fatalf("reparse export: %v\n%s", err, exported)
	}
	if !reflect.DeepEqual(canonicalEntrypointsFromRouterConfig(cfg), canonicalEntrypointsFromRouterConfig(reparsed)) {
		t.Fatal("v0.3 Entrypoint assignments changed across canonical round trip")
	}
}

func TestEntrypointAssignmentsFailClosed(t *testing.T) {
	tests := []struct {
		name    string
		replace string
		with    string
		want    string
	}{
		{"unsupported version", "version: v0.3", "version: v0.4", "version must be v0.3"},
		{"unknown recipe", "recipe: orchestration", "recipe: missing", "unknown Recipe"},
		{"unknown decision", "finish: {models: [{model: model-a}]}", "missing: {models: [{model: model-a}]}", "unknown Decision name"},
		{"missing decision", "      finish: {models: [{model: model-a}]}\n", "", "must assign every decision"},
		{"unknown model", "          - model: model-c", "          - model: missing", "unknown Model"},
		{"duplicate model target", "finish: {models: [{model: model-a}]}", "finish: {models: [{model: model-a}, {model: model-a}]}", "repeats the same model, LoRA, and reasoning target"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := testAuthoringParser(t).ParseYAMLBytes([]byte(strings.Replace(strictV03AuthoringYAML, test.replace, test.with, 1)))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}
}

func assertDecisionModels(t *testing.T, decision Decision, want ...string) {
	t.Helper()
	got := make([]string, 0, len(decision.ModelRefs))
	for _, ref := range decision.ModelRefs {
		got = append(got, ref.Model)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("decision models = %v, want %v", got, want)
	}
}
