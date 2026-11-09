package config

import (
	"strings"
	"testing"
)

func TestReaskDemandRequiresExplicitDecisionBinding(t *testing.T) {
	for _, scope := range []string{"unbound", "global", "recipe", "global by name", "recipe by name"} {
		t.Run(scope, func(t *testing.T) {
			cfg := &RouterConfig{}
			cfg.DecisionModel = "primary"
			cfg.ReaskRules = []ReaskRule{{Name: "repeat", Threshold: .8}}
			cfg.Decisions = []Decision{{Rules: RuleCombination{Operator: "AND", Conditions: []RuleNode{{Type: SignalTypeReask, Name: "repeat"}}}}}
			binding := map[string]ModelBinding{"reask": {Deployment: "primary", Contract: DecisionTaskContract}}
			if strings.HasSuffix(scope, "by name") {
				// The form a configuration writes today: the signal implies
				// the contract.
				binding = map[string]ModelBinding{"reask": {Deployment: "primary"}}
			}
			switch {
			case strings.HasPrefix(scope, "global"):
				cfg.GlobalModelBindings = binding
			case strings.HasPrefix(scope, "recipe"):
				cfg.ModelBindings = binding
			}
			wantTask := scope != "unbound"
			if cfg.ReaskUsesDecisionTask() != wantTask || TaskConsumerInUse(cfg, DefaultRecipeName, "reask") != wantTask {
				t.Fatalf("native task selection did not follow %s binding", scope)
			}
			if needed := EmbeddingModelsNeeded(cfg, "mmbert", false); needed["mmbert"] == wantTask {
				t.Fatalf("embedding demand=%v, native=%v", needed, wantTask)
			}
			if _, _, found, err := cfg.ImplicitTaskDeployment("reask"); err != nil || found {
				t.Fatalf("reask has an implicit native deployment: found=%v err=%v", found, err)
			}
		})
	}
}

// A reask binding written with its contract keeps asking the bound deployment
// after the migration drops the contract the signal implies.
func TestMigratedReaskBindingStillAsksItsDeployment(t *testing.T) {
	document := []byte(`version: v0.3
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
routing:
  signals:
    reasks: [{name: repeat, threshold: 0.8, lookback_turns: 1}]
  decisions:
    - name: route
      priority: 100
      rules: {operator: OR, conditions: [{type: reask, name: repeat}]}
      modelRefs: [{model: general}]
global:
  model_catalog:
    bindings:
      reask: {deployment: primary, contract: decision.v1}
`)
	migrated, _, err := MigrateModelResources(document)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(migrated), "reask: primary") {
		t.Fatalf("the binding was not written by name:\n%s", migrated)
	}
	for name, data := range map[string][]byte{"before": document, "after": migrated} {
		cfg, err := ParseYAMLBytesWithoutEnvExpansion(data)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if !cfg.ReaskUsesDecisionTask() {
			t.Errorf("%s: reask fell back to cosine similarity", name)
		}
		task, ok, err := cfg.ResolveTask("reask")
		if !ok || err != nil || task.Name != "primary" {
			t.Errorf("%s: reask resolves to %q (ok=%v, err=%v), want primary", name, task.Name, ok, err)
		}
	}
}
