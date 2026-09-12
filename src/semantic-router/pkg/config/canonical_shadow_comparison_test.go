package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func TestCanonicalExportIncludesShadowComparison(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			ShadowComparison: ShadowComparisonConfig{
				Enabled:   true,
				MaxWaitMS: 1500,
				Budget: ShadowBudgetConfig{
					MaxCalls:              2,
					MaxTokens:             1000,
					MaxCost:               0.5,
					MaxConcurrency:        1,
					PricePerMillionTokens: 3.0,
				},
				Judge: ShadowJudgeConfig{
					Enabled:        true,
					Model:          "judge-model",
					Endpoint:       "http://127.0.0.1:9001",
					RubricVersion:  "v2",
					TimeoutSeconds: 5,
				},
				Arms: []ShadowArmConfig{
					{Name: "a", Model: "m1", Endpoint: "http://127.0.0.1:9101"},
				},
			},
		},
	}

	encoded, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatalf("marshal canonical config: %v", err)
	}

	var document map[interface{}]interface{}
	if err := yaml.Unmarshal(encoded, &document); err != nil {
		t.Fatalf("unmarshal canonical config: %v", err)
	}
	global := requireYAMLMap(t, document["global"], "global")
	router := requireYAMLMap(t, global["router"], "global.router")
	comp := requireYAMLMap(t, router["shadow_comparison"], "global.router.shadow_comparison")
	if got, ok := comp["enabled"]; !ok || got != true {
		t.Fatalf("shadow_comparison.enabled = %#v, want true; YAML:\n%s", got, encoded)
	}
	budget := requireYAMLMap(t, comp["budget"], "shadow_comparison.budget")
	if got, ok := budget["max_calls"]; !ok || got != 2 {
		t.Fatalf("budget.max_calls = %#v, want 2; YAML:\n%s", got, encoded)
	}
	judge := requireYAMLMap(t, comp["judge"], "shadow_comparison.judge")
	if got, ok := judge["model"]; !ok || got != "judge-model" {
		t.Fatalf("judge.model = %#v, want judge-model; YAML:\n%s", got, encoded)
	}
	arms, ok := comp["arms"].([]interface{})
	if !ok || len(arms) != 1 {
		t.Fatalf("shadow_comparison.arms = %#v, want 1 entry; YAML:\n%s", comp["arms"], encoded)
	}
	first := arms[0].(map[interface{}]interface{})
	if first["model"] != "m1" {
		t.Fatalf("arms[0].model = %#v, want m1; YAML:\n%s", first["model"], encoded)
	}
}

func TestCanonicalLoadReadsShadowComparison(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.3
providers:
  defaults:
    model: private
  models:
    - name: private
      api_format: openai
      backend_refs:
        - endpoint: http://127.0.0.1:8000
routing: {}
global:
  router:
    shadow_comparison:
      enabled: true
      max_wait_ms: 2000
      budget:
        max_calls: 3
        max_tokens: 500
      judge:
        enabled: true
        model: judge-model
        endpoint: http://127.0.0.1:9001
        rubric_version: v3
      arms:
        - name: a
          model: arm-model
          endpoint: http://127.0.0.1:9101
`))
	if err != nil {
		t.Fatalf("parse canonical: %v", err)
	}
	sc := cfg.ShadowComparison
	if !sc.Enabled {
		t.Fatal("shadow_comparison.enabled must be true after canonical load")
	}
	if sc.MaxWaitMS != 2000 {
		t.Fatalf("max_wait_ms = %d, want 2000", sc.MaxWaitMS)
	}
	if sc.Budget.MaxCalls != 3 || sc.Budget.MaxTokens != 500 {
		t.Fatalf("budget = %+v, want MaxCalls=3 MaxTokens=500", sc.Budget)
	}
	if !sc.Judge.Enabled || sc.Judge.Model != "judge-model" || sc.Judge.RubricVersion != "v3" {
		t.Fatalf("judge = %+v, want enabled judge-model v3", sc.Judge)
	}
	if len(sc.Arms) != 1 || sc.Arms[0].Model != "arm-model" {
		t.Fatalf("arms = %+v, want one arm-model", sc.Arms)
	}
}

func TestCanonicalExportOmitsDisabledShadowComparison(t *testing.T) {
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{}}
	encoded, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatalf("marshal canonical config: %v", err)
	}
	if strings.Contains(string(encoded), "shadow_comparison") {
		t.Fatalf("disabled shadow_comparison should be omitted (omitempty); YAML:\n%s", encoded)
	}
}
