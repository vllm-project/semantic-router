package config

import (
	"math"
	"reflect"
	"testing"
)

func safetyTestConfig() *RouterConfig {
	cfg := &RouterConfig{}
	cfg.SafetyRules = []SafetyRule{{Name: "unsafe", Threshold: 0.5, Hazard: &SafetyHazardRule{Labels: []string{"privacy", "violence"}, Categories: []string{"privacy"}, Threshold: 0.6}}}
	cfg.SafetyModels = SafetyModelsConfig{Safety: SequenceHeadModelConfig{ModelID: "models/test-safety", UseCPU: true, MaxSequenceLength: 32768}, Hazard: SequenceHeadModelConfig{ModelID: "models/test-hazard", UseCPU: true, MaxSequenceLength: 32768}}
	return cfg
}

func TestSafetyValidationAndCanonicalRoundTrip(t *testing.T) {
	cfg := safetyTestConfig()
	if err := validateSafetySignalContracts(cfg); err != nil {
		t.Fatal(err)
	}
	canonical := CanonicalConfigFromRouterConfig(cfg)
	if !reflect.DeepEqual(canonical.Routing.Signals.Safety, cfg.SafetyRules) {
		t.Fatal("safety rules lost on export")
	}
	restored := &RouterConfig{}
	if err := applyCanonicalGlobal(restored, canonical.Global); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(restored.SafetyModels, cfg.SafetyModels) {
		t.Fatalf("local heads lost: %+v", restored.SafetyModels)
	}
	for name, mutate := range map[string]func(*RouterConfig){
		"empty model":         func(c *RouterConfig) { c.SafetyModels.Safety.ModelID = "" },
		"invalid context":     func(c *RouterConfig) { c.SafetyModels.Hazard.MaxSequenceLength = -1 },
		"unknown external":    func(c *RouterConfig) { c.SafetyRules[0].Model = "missing" },
		"nonfinite threshold": func(c *RouterConfig) { c.SafetyRules[0].Threshold = math.NaN() },
		"unknown hazard":      func(c *RouterConfig) { c.SafetyRules[0].Hazard.Categories = []string{"missing"} },
		"duplicate label":     func(c *RouterConfig) { c.SafetyRules[0].Hazard.Labels = []string{"privacy", "privacy"} },
		"all unsafe":          func(c *RouterConfig) { c.SafetyRules[0].UnsafeLabels = []string{"safe", "unsafe"} },
	} {
		t.Run(name, func(t *testing.T) {
			c := safetyTestConfig()
			mutate(c)
			if err := validateSafetySignalContracts(c); err == nil {
				t.Fatal("invalid contract accepted")
			}
		})
	}
}

func TestSafetyModelsFollowRecipeReachabilityAndHeadOverrides(t *testing.T) {
	cfg := safetyTestConfig()
	cfg.Recipes = []RoutingRecipe{{Name: DefaultRecipeName}, {Name: "private", Profile: RoutingProfile{Signals: cfg.Signals, Decisions: []Decision{{Name: "block", Rules: RuleNode{Type: SignalTypeSafety, Name: "unsafe"}}}}}}
	cfg.Signals = Signals{}
	if cfg.NeedsLocalSafetyHeadForRouting(false) || cfg.NeedsLocalSafetyHeadForRouting(true) {
		t.Fatal("unreachable recipe provisioned models")
	}
	cfg.Entrypoints = []EntrypointMapping{{ModelNames: []string{"private"}, Recipe: "private"}}
	if !cfg.NeedsLocalSafetyHeadForRouting(false) || !cfg.NeedsLocalSafetyHeadForRouting(true) {
		t.Fatal("reachable recipe lost local models")
	}
	cfg.Recipes[1].Profile.Signals.SafetyRules[0].Hazard.Model = "remote-hazard"
	if cfg.NeedsLocalSafetyHeadForRouting(true) {
		t.Fatal("external hazard still provisions local hazard")
	}
	cfg.Recipes[1].Profile.Signals.SafetyRules[0].Model = "remote-safety"
	if cfg.NeedsLocalSafetyHeadForRouting(false) {
		t.Fatal("external safety still provisions local safety")
	}
}
