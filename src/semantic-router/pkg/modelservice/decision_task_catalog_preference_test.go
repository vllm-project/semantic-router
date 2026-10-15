package modelservice

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestTaskCatalogPreferencePreservesExecutionContract(t *testing.T) {
	for _, test := range []struct {
		name, source       string
		examples, external bool
	}{
		{name: "generic native", source: "default"},
		{name: "authored prototypes", examples: true},
		{name: "external classifier", examples: true, external: true},
		{name: "global native override", examples: true, source: "global"},
		{name: "recipe native override", examples: true, source: "recipe"},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := config.DefaultGlobalConfig()
			cfg.PreferenceRules = []config.PreferenceRule{{Name: "brief", Description: "Short responses"}}
			if test.examples {
				cfg.PreferenceRules[0].Examples = []string{"Please be brief"}
			}
			if test.external {
				cfg.ExternalModels = []config.ExternalModelConfig{{ModelRole: config.ModelRolePreference}}
			}
			binding := map[string]config.ModelBinding{"preference": {Deployment: "primary", Contract: config.DecisionTaskContract}}
			switch test.source {
			case "global":
				cfg.GlobalModelBindings = binding
			case "recipe":
				cfg.ModelBindings = binding
			}
			cfg.Decisions = []config.Decision{{Rules: config.RuleCombination{Operator: "AND", Conditions: []config.RuleNode{{Type: config.SignalTypePreference, Name: "brief"}}}}}
			catalog := ProjectTaskCatalog(&cfg, nil)
			var active []TaskCatalogBinding
			for _, row := range catalog.Bindings {
				if row.Consumer == "preference" {
					active = append(active, row)
				}
			}
			if test.source == "" {
				if len(active) != 0 {
					t.Fatalf("non-native preference advertised as native task: %+v", active)
				}
			} else if len(active) != 1 || active[0].Source != test.source || active[0].Deployment != "primary" || active[0].Binding.Contract != config.DecisionTaskContract {
				t.Fatalf("native preference lost its resource or source: %+v", active)
			}
		})
	}
}
