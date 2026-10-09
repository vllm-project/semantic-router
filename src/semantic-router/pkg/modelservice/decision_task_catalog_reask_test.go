package modelservice

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestTaskCatalogReaskRequiresAuthoredBinding(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.DecisionModel = "primary"
	cfg.ReaskRules = []config.ReaskRule{{Name: "repeat"}}
	cfg.Decisions = []config.Decision{{Rules: config.RuleCombination{Operator: "AND", Conditions: []config.RuleNode{{Type: config.SignalTypeReask, Name: "repeat"}}}}}
	for _, explicit := range []bool{false, true} {
		if explicit {
			cfg.ModelBindings = map[string]config.ModelBinding{"reask": {Deployment: "primary", Contract: config.DecisionTaskContract}}
		}
		catalog := ProjectTaskCatalog(cfg, nil)
		if _, ok := catalog.DefaultBindings["reask"]; ok {
			t.Fatal("cosine reask was advertised as an implicit native binding")
		}
		found := false
		for _, binding := range catalog.Bindings {
			if binding.Consumer == "reask" {
				found = true
				if binding.Source != "recipe" || binding.Binding.Contract != config.DecisionTaskContract {
					t.Fatalf("native reask binding=%+v", binding)
				}
			}
		}
		if found != explicit {
			t.Fatalf("native binding=%v, explicit=%v", found, explicit)
		}
	}
}
