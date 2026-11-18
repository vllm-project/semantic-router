package extproc

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
)

const listenerRulesConfigYAML = `version: v0.5
providers:
  defaults:
    model: m
  models:
    - name: m
      backend_refs: [{endpoint: "127.0.0.1:8000", provider: vllm}]
global:
  model_catalog:
    deployments:
      gone:
        provider: systemone
        endpoint: http://127.0.0.1:1/v1/systemone
        model: gone
        timeout_ms: 500
        capabilities: {question_types: [noul]}
recipes:
  - name: default
    signals:
      keywords:
        - {name: prod_database, operator: OR, method: regex, keywords: ['prod-db\.internal']}
        - {name: employee_id, operator: OR, method: regex, keywords: ['\bE-\d{6}\b']}
        - {name: long_number, operator: OR, method: regex, keywords: ['\d{4,}']}
      systemone:
        - name: deletes_data
          deployment: gone
          question:
            type: noul
            instructions: Does this command delete data that cannot be restored?
`

func listenerRulesRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(listenerRulesConfigYAML))
	if err != nil {
		t.Fatal(err)
	}
	classifiers, err := classification.BuildRecipeClassifiers(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return &OpenAIRouter{Config: cfg, Classifier: classifiers.Default(), RecipeClassifiers: classifiers}
}

func bind(names ...string) []config.VSRRule {
	rules := make([]config.VSRRule, len(names))
	for i, name := range names {
		kind := config.SignalTypeKeyword
		if name == "deletes_data" {
			kind = config.SignalTypeSystemOne
		}
		rules[i] = config.VSRRule{SignalRef: config.SignalRef{Type: kind, Name: name}}
	}
	return rules
}

func asJSON(t *testing.T, value any) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}

// Guard reports the bound rules each part matches; a rule the Router could
// not check is unchecked with a reason that holds no text, and the part's
// coverage says so.
func TestListenerGuardRulesMatchEachPart(t *testing.T) {
	router := listenerRulesRouter(t)
	rules := &config.VSRRules{Recipe: "default", Guard: bind("prod_database", "deletes_data")}
	parts := []systemone.RulePart{{ID: "p1", Text: "psql --host=prod-db.internal -c 'drop table x'"}, {ID: "p2", Text: "ls"}}
	results, err := router.EvaluateListenerRules(t.Context(), "agents", rules, config.VSRAppGuard, parts)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 || results[0].Coverage != "partial" || asJSON(t, results[0].Matches) != `["prod_database"]` ||
		len(results[0].Unchecked) != 1 || results[0].Unchecked[0].Rule != "deletes_data" || len(results[1].Matches) != 0 {
		t.Fatalf("results = %s", asJSON(t, results))
	}
	if reason := results[0].Unchecked[0].Reason; reason == "" || strings.Contains(reason, "prod-db") || strings.Contains(reason, "drop table") {
		t.Fatalf("the unchecked reason %q is empty or carries the text", reason)
	}

	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	results, err = router.EvaluateListenerRules(cancelled, "agents", rules, config.VSRAppGuard, parts[:1])
	if err != nil || results[0].Coverage != "failed" || len(results[0].Matches) != 0 || len(results[0].Unchecked) != 2 {
		t.Fatalf("a call past its deadline answered %s, %v", asJSON(t, results), err)
	}
}

// Redact returns what its rules find in code points, overlapping spans of
// different rules merged into one under the rule that starts first.
func TestListenerRedactRulesMergeTheirSpans(t *testing.T) {
	router := listenerRulesRouter(t)
	rules := &config.VSRRules{Recipe: "default", Redact: bind("long_number", "employee_id")}
	parts := []systemone.RulePart{{ID: "out", Text: "계정 E-104233 paid 99887766"}}
	results, err := router.EvaluateListenerRules(t.Context(), "agents", rules, config.VSRAppRedact, parts)
	if err != nil {
		t.Fatal(err)
	}
	want := `[{"start":3,"end":11,"rule":"employee_id"},{"start":17,"end":25,"rule":"long_number"}]`
	if results[0].Coverage != "complete" || asJSON(t, results[0].Spans) != want {
		t.Fatalf("spans = %s, want %s", asJSON(t, results[0]), want)
	}
	if _, err := router.EvaluateListenerRules(t.Context(), "agents", &config.VSRRules{Recipe: "missing", Redact: rules.Redact}, config.VSRAppRedact, parts); err == nil {
		t.Fatal("rules of a recipe with no classifier were evaluated")
	}
}
