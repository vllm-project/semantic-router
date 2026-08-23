package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

const entrypointRulesYAML = `
version: v0.4
models:
  - name: model-a
    card: {}
    connections:
      - {provider: private-test, endpoint: http://model-a.example, model: model-a}
  - name: model-b
    card: {}
    connections:
      - {provider: private-test, endpoint: http://model-b.example, model: model-b}
  - name: model-c
    card:
      reasoning: {type: reasoning_effort, efforts: [high]}
    connections:
      - {provider: private-test, endpoint: http://model-c.example, model: model-c}
recipes:
  - name: orchestration
    document:
      decisions:
        - name: choose
          rules: {}
        - name: finish
          rules: {}
entrypoints:
  - name: vllm-sr/edge
    aliases: [vllm-sr/edge-alias]
    rules:
      - name: premium
        matches:
          - claim: {name: routing_tier, exact: premium}
        recipe: orchestration
        assignments:
          choose:
            models:
              - model: model-c
                weight: "2"
                reasoning: {enabled: true, effort: high}
          finish: {models: [{model: model-b}]}
      - name: default
        matches: []
        recipe: orchestration
        assignments:
          choose: {models: [{model: model-b}]}
          finish: {models: [{model: model-a}]}
global:
  services:
    backend_egress: {policy_file: /app/config/backend-egress-policy.yaml}
  integrations:
    looper: {endpoint: "http://localhost:8899/v1/chat/completions"}
`

func TestEntrypointRulesCompileStablePinnedViews(t *testing.T) {
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(entrypointRulesYAML))
	if err != nil {
		t.Fatalf("parse v0.4 entrypoint: %v", err)
	}
	if len(cfg.Entrypoints) != 1 || len(cfg.Entrypoints[0].Rules) != 2 {
		t.Fatalf("unexpected normalized entrypoint: %+v", cfg.Entrypoints)
	}
	defaultRecipe, ok := cfg.RecipeForRequestModel("vllm-sr/edge")
	if !ok {
		t.Fatal("request-model-only resolution must select the unconditional rule")
	}
	assertDecisionModels(t, defaultRecipe.Profile.Decisions[0], "model-b")
	assertDecisionModels(t, defaultRecipe.Profile.Decisions[1], "model-a")

	var premium *RoutingRecipe
	var premiumRule *EntrypointRule
	for index := range cfg.Entrypoints[0].Rules {
		if cfg.Entrypoints[0].Rules[index].Name == "premium" {
			premiumRule = &cfg.Entrypoints[0].Rules[index]
			premium = premiumRule.derivedRecipe
			break
		}
	}
	if premium == nil || premium == defaultRecipe {
		t.Fatal("each rule must receive an independent immutable derived view")
	}
	assertDecisionModels(t, premium.Profile.Decisions[0], "model-c")
	ref := premium.Profile.Decisions[0].ModelRefs[0]
	if ref.UseReasoning == nil || !*ref.UseReasoning || ref.ReasoningEffort != "high" || ref.Weight != 2 {
		t.Fatalf("assignment controls were not compiled: %+v", ref)
	}
	decisionID := premium.Profile.Decisions[0].ID
	assignment := premiumRule.Action.Assignments[decisionID].Models[0]
	if assignment.ModelRevision != initialRoutingResourceRevision ||
		premiumRule.Action.RecipeRevision != initialRoutingResourceRevision {
		t.Fatalf("action did not pin exact revisions: %+v", premiumRule.Action)
	}
}

func TestEntrypointRulesCanonicalRoundTrip(t *testing.T) {
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(entrypointRulesYAML))
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
		t.Fatal("v0.4 Entrypoint rules changed across canonical round trip")
	}
}

func TestEntrypointRulesFailClosed(t *testing.T) {
	tests := []struct {
		name    string
		replace string
		with    string
		want    string
	}{
		{"old version", "version: v0.4", "version: v0.3", "version must be v0.4"},
		{"unknown recipe", "recipe: orchestration", "recipe: missing", "unknown Recipe"},
		{"unknown decision", "finish: {models: [{model: model-b}]}", "missing: {models: [{model: model-b}]}", "unknown Decision name"},
		{"missing decision", "          finish: {models: [{model: model-b}]}\n", "", "must assign every decision"},
		{"unknown model", "              - model: model-c", "              - model: missing", "unknown Model"},
		{"duplicate model target", "finish: {models: [{model: model-b}]}", "finish: {models: [{model: model-b}, {model: model-b}]}", "repeats the same model, LoRA, and reasoning target"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := testAuthoringParser(t).ParseYAMLBytes([]byte(strings.Replace(entrypointRulesYAML, test.replace, test.with, 1)))
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
