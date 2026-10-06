package modeldownload

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDefaultHazardIsServedOnlyForAReachableConsumer(t *testing.T) {
	for _, state := range []string{"unused", "dormant", "reachable"} {
		t.Run(state, func(t *testing.T) {
			cfg := config.DefaultGlobalConfig()
			cfg.MoMRegistry = config.ToLegacyRegistry()
			cfg.Recipes = []config.RoutingRecipe{{Name: config.DefaultRecipeName}}
			if state != "unused" {
				cfg.Recipes = append(cfg.Recipes, config.RoutingRecipe{Name: "care", Profile: config.RoutingProfile{
					Signals: config.Signals{ClassifierRules: []config.ClassifierSignalRule{{Name: "content-risk", Type: "local", Labels: []string{"violence", "criminal_activity", "sexual_content", "child_exploitation", "hate", "harassment_abuse", "regulated_substances", "weapons", "self_harm", "privacy", "specialized_advice", "misinformation"}}}},
					ModelBindings: map[string]config.ModelBinding{"classifier.content-risk": {
						Deployment: "hazard", Adapter: "modernbert", Contract: config.RemoteClassifierContractLabelScores,
					}},
					Decisions: []config.Decision{{Name: "risky", Rules: config.RuleNode{Type: config.SignalTypeClassifier, Name: "content-risk"}}},
				}})
			}
			if state == "reachable" {
				cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{"care"}, Recipe: "care"}}
			}
			specs, err := BuildModelSpecs(&cfg)
			if err != nil {
				t.Fatal(err)
			}
			if spec, ok := findSpecByPath(specs, "models/Vela-1.0-Encoder-307M-Hazard"); ok {
				t.Fatalf("router downloads the runtime-served Hazard: %+v", spec)
			}
			_, served := config.ModelRuntimeDeploymentsInUse(&cfg)["hazard"]
			if served != (state == "reachable") {
				t.Fatalf("Hazard served=%v for %s", served, state)
			}
			if served {
				assertRuntimeServed(t, &cfg, specs, "models/Vela-1.0-Encoder-307M-Hazard")
			}
		})
	}
}
