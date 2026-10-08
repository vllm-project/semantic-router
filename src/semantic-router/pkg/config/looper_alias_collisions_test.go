package config

import (
	"reflect"
	"strings"
	"testing"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"
)

// looperAliasTestYAML is the shape that broke the response-api-redis profile
// (#4651): the backend model is also a Flow alias, so its requests evaluate
// only the workflows decision.
const looperAliasTestYAML = `
version: v0.3
providers:
  defaults:
    model: gpt-oss
  models:
    - name: gpt-oss
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
    - name: helper
      backend_refs:
        - endpoint: 127.0.0.1:8001
          provider: vllm
routing:
  modelCards:
    - name: gpt-oss
    - name: helper
  signals:
    keywords:
      - name: plan_keywords
        operator: OR
        keywords: ["plan"]
  decisions:
    - name: workflow_route
      priority: 20
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: plan_keywords
      modelRefs:
        - model: gpt-oss
          use_reasoning: false
      algorithm:
        type: workflows
        workflows:
          mode: static
          roles:
            - name: worker
              models: [gpt-oss]
    - name: default_route
      priority: 10
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: gpt-oss
          use_reasoning: false
`

func observeConfigWarnings(t *testing.T) *observer.ObservedLogs {
	t.Helper()
	core, logs := observer.New(zapcore.WarnLevel)
	t.Cleanup(zap.ReplaceGlobals(zap.New(core)))
	return logs
}

func looperAliasWarnings(logs *observer.ObservedLogs) []map[string]interface{} {
	var warnings []map[string]interface{}
	for _, entry := range logs.All() {
		fields := entry.ContextMap()
		if fields["event"] == "looper_alias_shadows_model" {
			warnings = append(warnings, fields)
		}
	}
	return warnings
}

func TestFlowAliasNamingABackendModelWarnsAtLoad(t *testing.T) {
	logs := observeConfigWarnings(t)
	cfg, err := ParseYAMLBytes([]byte(looperAliasTestYAML + `
global:
  integrations:
    looper:
      flow:
        model_names: [gpt-oss]
`))
	if err != nil {
		t.Fatalf("a Flow alias that names a model must load, with a warning: %v", err)
	}
	if !cfg.IsFlowModelName("gpt-oss") {
		t.Fatal("the alias was not registered")
	}

	warnings := looperAliasWarnings(logs)
	if len(warnings) != 1 {
		t.Fatalf("got %d looper_alias_shadows_model warnings, want 1: %v", len(warnings), warnings)
	}
	warning := warnings[0]
	if warning["field"] != "global.integrations.looper.flow.model_names" || warning["alias"] != "gpt-oss" {
		t.Fatalf("the warning names the wrong alias: %v", warning)
	}
	if warning["served_by_providers"] != true {
		t.Fatalf("the warning does not say providers serve the model: %v", warning)
	}
	if got, want := warning["routed_by_decisions"], []interface{}{"workflow_route", "default_route"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("routed_by_decisions = %v, want %v", got, want)
	}
	if got, want := warning["evaluated_decisions"], []interface{}{"workflow_route"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("evaluated_decisions = %v, want %v", got, want)
	}
	reason, _ := warning["reason"].(string)
	if !strings.Contains(reason, "evaluate only workflows decisions") || !strings.Contains(reason, "no_route") {
		t.Fatalf("the reason does not say what the alias does to the model's requests: %q", reason)
	}
}

func TestDistinctLooperAliasesLoadWithoutWarnings(t *testing.T) {
	logs := observeConfigWarnings(t)
	if _, err := ParseYAMLBytes([]byte(looperAliasTestYAML + `
global:
  integrations:
    looper:
      flow:
        model_names: [vllm-sr/flow, team/flow]
`)); err != nil {
		t.Fatal(err)
	}
	if warnings := looperAliasWarnings(logs); len(warnings) != 0 {
		t.Fatalf("distinct aliases must not warn: %v", warnings)
	}
}

func TestLooperAliasCollisionsCoverEveryFamilyAndModelReference(t *testing.T) {
	cfg := &RouterConfig{
		BackendModels: BackendModels{ModelConfig: map[string]ModelParams{
			"base": {LoRAs: []LoRAAdapter{{Name: "base-sql"}}},
		}},
		Recipes: []RoutingRecipe{
			{Name: DefaultRecipeName},
			{Name: "privacy", Profile: RoutingProfile{Decisions: []Decision{{
				Name:      "private_route",
				ModelRefs: []ModelRef{{Model: "private-model"}},
			}}}},
		},
	}
	cfg.Looper.ReMoM.ModelNames = []string{"base-sql"}
	cfg.Looper.Fusion.ModelNames = []string{"private-model"}

	collisions := cfg.looperAliasCollisions()
	if len(collisions) != 2 {
		t.Fatalf("got %d collisions, want the LoRA adapter and the decision's model: %+v", len(collisions), collisions)
	}
	lora, decisionModel := collisions[0], collisions[1]
	if lora.alias != "base-sql" || lora.algorithm != DecisionAlgorithmReMoM || !lora.served || len(lora.routedBy) != 0 {
		t.Fatalf("a ReMoM alias naming a served LoRA adapter: %+v", lora)
	}
	if decisionModel.alias != "private-model" || decisionModel.algorithm != DecisionAlgorithmFusion || decisionModel.served {
		t.Fatalf("a Fusion alias naming a decision's model: %+v", decisionModel)
	}
	if want := []string{RoutingDecisionKey("privacy", "private_route")}; !reflect.DeepEqual(decisionModel.routedBy, want) {
		t.Fatalf("routedBy = %v, want the recipe-scoped decision %v", decisionModel.routedBy, want)
	}
}

func TestLooperAliasCollisionReportsOnlyTheFamilyThatCapturesTheName(t *testing.T) {
	cfg := &RouterConfig{BackendModels: BackendModels{ModelConfig: map[string]ModelParams{"shared": {}}}}
	cfg.Looper.Fusion.ModelNames = []string{"shared"}
	cfg.Looper.Flow.ModelNames = []string{"shared"}

	collisions := cfg.looperAliasCollisions()
	if len(collisions) != 1 || collisions[0].algorithm != DecisionAlgorithmFusion {
		t.Fatalf("request routing resolves Fusion before Flow, so only Fusion captures the name: %+v", collisions)
	}
}

func TestLooperAliasCollisionWarningSkipsRecipeViews(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(looperAliasTestYAML + `
global:
  integrations:
    looper:
      flow:
        model_names: [gpt-oss]
`))
	if err != nil {
		t.Fatal(err)
	}
	logs := observeConfigWarnings(t)
	if err := warnLooperAliasCollisions(cfg.ConfigForRecipe(cfg.DefaultRecipe())); err != nil {
		t.Fatal(err)
	}
	if warnings := looperAliasWarnings(logs); len(warnings) != 0 {
		t.Fatalf("a recipe view repeated the whole configuration's warning: %v", warnings)
	}
}
