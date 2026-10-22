package config

import "testing"

func TestReaskDemandRequiresExplicitDecisionBinding(t *testing.T) {
	for _, scope := range []string{"unbound", "global", "recipe"} {
		t.Run(scope, func(t *testing.T) {
			cfg := &RouterConfig{}
			cfg.DecisionModel = "primary"
			cfg.ReaskRules = []ReaskRule{{Name: "repeat", Threshold: .8}}
			cfg.Decisions = []Decision{{Rules: RuleCombination{Operator: "AND", Conditions: []RuleNode{{Type: SignalTypeReask, Name: "repeat"}}}}}
			binding := map[string]ModelBinding{"reask": {Deployment: "primary", Contract: DecisionTaskContract}}
			switch scope {
			case "global":
				cfg.GlobalModelBindings = binding
			case "recipe":
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
